"""Procedural UDMF WAD generator for multi-tier DOOM RL training and held-out evaluation.

Generates Doom-compatible UDMF PWADs (`MAP01`) with multi-room layouts, corners,
regular doors, keycard-locked doors requiring backtracking, enemies, pickups,
and exits. Generated maps are compatible with `doom_env.make_env` and Freedoom2.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from doom_env.wad import Lump, write_wad

CELL_SIZE = 64.0
LOCK_MAP = {"R": 129, "B": 130, "Y": 131}
LOCK_TEX = {"R": "DOORRED", "B": "DOORBLU", "Y": "DOORYEL", "D": "BIGDOOR2"}
THING_TYPES = {
    "P": 1,     # DoomPlayer
    "z": 3004,  # Zombieman
    "s": 9,     # ShotgunGuy
    "i": 3001,  # DoomImp
    "d": 3002,  # Demon
    "r": 13,    # RedCard
    "b": 5,     # BlueCard
    "y": 6,     # YellowCard
    "h": 2011,  # Stimpack
    "m": 2012,  # Medikit
    "c": 2007,  # Clip
    "g": 2001,  # Shotgun
    "a": 2018,  # GreenArmor
}
FLOOR_CHARS = set(".Pzsidrbymhcga")
DOOR_CHARS = set("DRBY")


@dataclass(frozen=True)
class MapSpec:
    name: str
    tier: int
    grid: str
    player_angle: int = 0
    description: str = ""


def grid_to_udmf(grid: str, *, player_angle: int = 0) -> str:
    """Compile an ASCII grid into a ZDoom UDMF TEXTMAP string."""
    rows = [line for line in grid.strip("\n").splitlines() if line]
    if not rows:
        raise ValueError("Grid cannot be empty")
    height = len(rows)
    width = max(len(r) for r in rows)
    board = [list(r.ljust(width, "#")) for r in rows]

    # Map grid (row=0 is top/north -> highest y)
    def cell_xy(r: int, c: int) -> tuple[float, float]:
        return float(c * CELL_SIZE), float((height - 1 - r) * CELL_SIZE)

    sector_grid = [[-1] * width for _ in range(height)]
    sectors: list[dict] = []

    def add_sector(floor_tex: str, ceil_tex: str, h_floor: int, h_ceil: int) -> int:
        idx = len(sectors)
        sectors.append({
            "texturefloor": floor_tex,
            "textureceiling": ceil_tex,
            "heightfloor": h_floor,
            "heightceiling": h_ceil,
            "lightlevel": 192,
        })
        return idx

    # Assign room sectors by flood-filling contiguous walkable floor cells
    room_textures = ("FLOOR0_1", "FLOOR4_8", "FLAT14", "MFLR8_1")
    room_wall_textures = ("STARTAN2", "STONE2", "BROWN1", "GRAY1")
    sector_wall_tex: dict[int, str] = {}
    room_count = 0

    for r in range(height):
        for c in range(width):
            ch = board[r][c]
            if sector_grid[r][c] != -1:
                continue
            if ch in FLOOR_CHARS:
                sec_id = add_sector(
                    room_textures[room_count % len(room_textures)],
                    "CEIL1_1",
                    0,
                    128,
                )
                sector_wall_tex[sec_id] = room_wall_textures[room_count % len(room_wall_textures)]
                room_count += 1
                q = deque([(r, c)])
                sector_grid[r][c] = sec_id
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and sector_grid[nr][nc] == -1:
                            if board[nr][nc] in FLOOR_CHARS:
                                sector_grid[nr][nc] = sec_id
                                q.append((nr, nc))
            elif ch in DOOR_CHARS:
                sec_id = add_sector("FLAT5_4", "CEIL1_1", 0, 0)
                sector_wall_tex[sec_id] = "DOORTRAK"
                sector_grid[r][c] = sec_id
            elif ch == "E":
                sec_id = add_sector("GATE3", "CEIL1_1", 0, 128)
                sector_wall_tex[sec_id] = "GRAY1"
                sector_grid[r][c] = sec_id

    vertex_map: dict[tuple[float, float], int] = {}
    vertices: list[tuple[float, float]] = []

    def get_v(x: float, y: float) -> int:
        key = (float(x), float(y))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append(key)
        return vertex_map[key]

    sidedefs: list[dict] = []
    linedefs: list[dict] = []

    def add_sidedef(sector: int, top: str = "-", mid: str = "-", bot: str = "-") -> int:
        idx = len(sidedefs)
        sidedefs.append({
            "sector": sector,
            "texturetop": top,
            "texturemiddle": mid,
            "texturebottom": bot,
        })
        return idx

    def emit_edge(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        right_sec: int,
        left_sec: int,
        right_ch: str,
        left_ch: str,
    ) -> None:
        if right_sec == left_sec:
            return
        # Ensure right_sec >= 0 (if right is void, flip edge direction so front is solid sector)
        if right_sec < 0:
            x1, y1, x2, y2 = x2, y2, x1, y1
            right_sec, left_sec = left_sec, right_sec
            right_ch, left_ch = left_ch, right_ch

        v1 = get_v(x1, y1)
        v2 = get_v(x2, y2)

        if left_sec < 0:
            # One-sided solid wall facing inward toward right_sec
            mid_tex = sector_wall_tex.get(right_sec, "STARTAN2")
            sf = add_sidedef(right_sec, mid=mid_tex)
            linedefs.append({
                "v1": v1,
                "v2": v2,
                "sidefront": sf,
                "blocking": True,
                "dontpegbottom": True,
            })
            return

        # Two-sided boundary between right_sec and left_sec
        door_char = right_ch if right_ch in DOOR_CHARS else (left_ch if left_ch in DOOR_CHARS else None)
        is_exit = (right_ch == "E") or (left_ch == "E")

        if door_char is not None:
            # MapInfo expects sideback to point to the door sector (arg0 == 0).
            if right_ch in DOOR_CHARS and left_ch not in DOOR_CHARS:
                x1, y1, x2, y2 = x2, y2, x1, y1
                right_sec, left_sec = left_sec, right_sec
                right_ch, left_ch = left_ch, right_ch
                v1 = get_v(x1, y1)
                v2 = get_v(x2, y2)
            tex = LOCK_TEX[door_char]
            top_f = "-" if right_ch in DOOR_CHARS else tex
            top_b = "-" if left_ch in DOOR_CHARS else tex
            sf = add_sidedef(right_sec, top=top_f)
            sb = add_sidedef(left_sec, top=top_b)
            # Orient so player outside door triggers front/back cleanly
            if door_char == "D":
                linedefs.append({
                    "v1": v1,
                    "v2": v2,
                    "sidefront": sf,
                    "sideback": sb,
                    "twosided": True,
                    "special": 12,
                    "arg0": 0,
                    "arg1": 16,
                    "arg2": 150,
                    "playeruse": True,
                    "monsteruse": False,
                    "repeatspecial": True,
                })
            else:
                linedefs.append({
                    "v1": v1,
                    "v2": v2,
                    "sidefront": sf,
                    "sideback": sb,
                    "twosided": True,
                    "special": 13,
                    "arg0": 0,
                    "arg1": 16,
                    "arg2": 0,
                    "arg3": LOCK_MAP[door_char],
                    "playeruse": True,
                    "monsteruse": False,
                    "repeatspecial": True,
                })
        elif is_exit:
            sf = add_sidedef(right_sec)
            sb = add_sidedef(left_sec)
            linedefs.append({
                "v1": v1,
                "v2": v2,
                "sidefront": sf,
                "sideback": sb,
                "twosided": True,
                "special": 243,
                "playercross": True,
            })
        else:
            sf = add_sidedef(right_sec)
            sb = add_sidedef(left_sec)
            linedefs.append({
                "v1": v1,
                "v2": v2,
                "sidefront": sf,
                "sideback": sb,
                "twosided": True,
            })

    # Emit horizontal edges (between row r and row r-1, and bottom boundary of last row)
    for r in range(height + 1):
        for c in range(width):
            # Edge goes East: (c*64, y) -> ((c+1)*64, y)
            # Right side of Eastward vector is South (row r), Left side is North (row r-1)
            y = float((height - r) * CELL_SIZE)
            x1 = float(c * CELL_SIZE)
            x2 = float((c + 1) * CELL_SIZE)
            right_sec = sector_grid[r][c] if r < height else -1
            left_sec = sector_grid[r - 1][c] if r > 0 else -1
            right_ch = board[r][c] if r < height else "#"
            left_ch = board[r - 1][c] if r > 0 else "#"
            emit_edge(x1, y, x2, y, right_sec, left_sec, right_ch, left_ch)

    # Emit vertical edges (between col c-1 and col c, and right boundary of last col)
    for r in range(height):
        for c in range(width + 1):
            # Edge goes North: (x, y0) -> (x, y1)
            # Right side of Northward vector is East (col c), Left side is West (col c-1)
            x = float(c * CELL_SIZE)
            y0 = float((height - 1 - r) * CELL_SIZE)
            y1 = float((height - r) * CELL_SIZE)
            right_sec = sector_grid[r][c] if c < width else -1
            left_sec = sector_grid[r][c - 1] if c > 0 else -1
            right_ch = board[r][c] if c < width else "#"
            left_ch = board[r][c - 1] if c > 0 else "#"
            emit_edge(x, y0, x, y1, right_sec, left_sec, right_ch, left_ch)

    # Collect things
    things: list[dict] = []
    player_count = 0
    for r in range(height):
        for c in range(width):
            ch = board[r][c]
            if ch in THING_TYPES:
                x0, y0 = cell_xy(r, c)
                cx = x0 + CELL_SIZE * 0.5
                cy = y0 + CELL_SIZE * 0.5
                angle = player_angle if ch == "P" else 180
                if ch == "P":
                    player_count += 1
                things.append({"x": cx, "y": cy, "angle": angle, "type": THING_TYPES[ch]})

    if player_count != 1:
        raise ValueError(f"Expected exactly 1 player start 'P', found {player_count}")

    lines: list[str] = ['namespace = "zdoom";']
    for x, y in vertices:
        lines.append(f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}")
    for s in sectors:
        lines.append(
            f'sector {{ texturefloor = "{s["texturefloor"]}"; textureceiling = "{s["textureceiling"]}"; '
            f'heightfloor = {s["heightfloor"]}; heightceiling = {s["heightceiling"]}; lightlevel = {s["lightlevel"]}; }}'
        )
    for sd in sidedefs:
        lines.append(
            f'sidedef {{ sector = {sd["sector"]}; texturetop = "{sd["texturetop"]}"; '
            f'texturemiddle = "{sd["texturemiddle"]}"; texturebottom = "{sd["texturebottom"]}"; }}'
        )
    for ld in linedefs:
        parts = [f"v1 = {ld['v1']}", f"v2 = {ld['v2']}", f"sidefront = {ld['sidefront']}"]
        if "sideback" in ld:
            parts.append(f"sideback = {ld['sideback']}")
        for k, v in ld.items():
            if k in ("v1", "v2", "sidefront", "sideback"):
                continue
            if isinstance(v, bool):
                parts.append(f"{k} = {'true' if v else 'false'}")
            else:
                parts.append(f"{k} = {v}")
        lines.append("linedef { " + "; ".join(parts) + "; }")
    for th in things:
        lines.append(
            f"thing {{ x = {th['x']:.1f}; y = {th['y']:.1f}; angle = {th['angle']}; type = {th['type']}; "
            f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}"
        )
    return "\n".join(lines) + "\n"


def write_udmf_wad(path: str | Path, grid: str, *, player_angle: int = 0, map_name: str = "MAP01") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    textmap = grid_to_udmf(grid, player_angle=player_angle).encode("utf-8")
    write_wad(out, [Lump(map_name.upper(), b""), Lump("TEXTMAP", textmap), Lump("ENDMAP", b"")])
    return out


def _rotate_grid(grid: str, k: int) -> str:
    rows = [list(line) for line in grid.strip("\n").splitlines() if line]
    w = max(len(r) for r in rows)
    arr = np.array([list("".join(r).ljust(w, "#")) for r in rows])
    rot = np.rot90(arr, k=k % 4)
    return "\n".join("".join(row) for row in rot)


def build_curriculum_specs() -> tuple[list[MapSpec], list[MapSpec]]:
    """Return (training_specs, heldout_specs) with strictly disjoint layouts."""
    train_templates = [
        # --- Tier 1: Combat & Survival ---
        MapSpec(
            "proc_t1_arena_imp", 1,
            "#########\n"
            "#P.....i#\n"
            "#.......#\n"
            "#..#.#..#\n"
            "#.....hE#\n"
            "#########",
            player_angle=0,
            description="Open arena with cover pillars, imp, stimpack, and exit.",
        ),
        MapSpec(
            "proc_t1_crossfire_zombies", 1,
            "#########\n"
            "#..z....#\n"
            "#P....E.#\n"
            "#..z..h.#\n"
            "#########",
            player_angle=0,
            description="Two zombiemen flanking a room with stimpack and exit.",
        ),
        MapSpec(
            "proc_t1_turn_combat", 1,
            "#######\n"
            "#..i..#\n"
            "#.....#\n"
            "#..P..#\n"
            "#..h.E#\n"
            "#######",
            player_angle=180,
            description="Enemy is off-axis/behind initial facing; requires turning and shooting.",
        ),
        MapSpec(
            "proc_t1_shotgun_duel", 1,
            "#########\n"
            "#P.g....#\n"
            "#..#....#\n"
            "#....i.E#\n"
            "#########",
            player_angle=0,
            description="Shotgun pickup behind cover with an imp guarding the exit.",
        ),
        # --- Tier 2: Multi-Room Exploration & Doors ---
        MapSpec(
            "proc_t2_l_bend_door", 2,
            "#########\n"
            "#P..#..E#\n"
            "#...D...#\n"
            "#...#.h.#\n"
            "#########",
            player_angle=0,
            description="Two rooms with a central door and offset exit.",
        ),
        MapSpec(
            "proc_t2_three_room_turn", 2,
            "###########\n"
            "#P..#..z..#\n"
            "#...D.....#\n"
            "#...###D###\n"
            "#####.....#\n"
            "#####.h..E#\n"
            "###########",
            player_angle=0,
            description="Three rooms requiring 90-degree turns and two doors.",
        ),
        MapSpec(
            "proc_t2_branch_choice", 2,
            "###########\n"
            "#h..#...i.#\n"
            "#...D.....#\n"
            "#P..###D###\n"
            "###...#...#\n"
            "###...D..E#\n"
            "###########",
            player_angle=90,
            description="Branching hub with dead-end room, enemy room, and exit branch.",
        ),
        MapSpec(
            "proc_t2_u_corridor", 2,
            "#########\n"
            "#P.....z#\n"
            "#####D###\n"
            "#E..#...#\n"
            "#...D...#\n"
            "#########",
            player_angle=0,
            description="U-shaped multi-room loop with enemy and two doors.",
        ),
        # --- Tier 3: Long-Horizon Keycard -> Backtrack -> Locked Door -> Exit ---
        MapSpec(
            "proc_t3_red_branch", 3,
            "###########\n"
            "#r..#....E#\n"
            "#...D..#R##\n"
            "#...#..#..#\n"
            "#...#P....#\n"
            "###########",
            player_angle=180,
            description="Hub with locked red door; red key behind side door, backtrack to unlock.",
        ),
        MapSpec(
            "proc_t3_blue_backtrack", 3,
            "###########\n"
            "#P...#...E#\n"
            "#....B....#\n"
            "###D#######\n"
            "#..b.#\n"
            "#....#\n"
            "######",
            player_angle=0,
            description="See blue locked door, explore south door for blue key, return to unlock.",
        ),
        MapSpec(
            "proc_t3_yellow_hub", 3,
            "###########\n"
            "#E.Y......#\n"
            "#####..#..#\n"
            "#y..D..P..#\n"
            "#......#..#\n"
            "###########",
            player_angle=90,
            description="Yellow key in west room behind door; yellow locked door in north-west.",
        ),
        MapSpec(
            "proc_t3_two_doors_key", 3,
            "#############\n"
            "#P...#...#..#\n"
            "#....D...R.E#\n"
            "#....#...#..#\n"
            "###D##...####\n"
            "#..r.#\n"
            "######",
            player_angle=270,
            description="Multi-room key hunt requiring turning, normal door, key pickup, and backtrack.",
        ),
        # --- Tier 4: Full Gauntlet (Combat + Exploration + Keycard + Doors + Exit) ---
        MapSpec(
            "proc_t4_gauntlet_red", 4,
            "#############\n"
            "#P..#...z#..#\n"
            "#...D....R.E#\n"
            "#...#.#..#..#\n"
            "###D#.#h.####\n"
            "#r.z#\n"
            "#####",
            player_angle=0,
            description="Enemies in hub and key room, locked red door guarding exit.",
        ),
        MapSpec(
            "proc_t4_gauntlet_blue", 4,
            "#############\n"
            "#b..i.#....E#\n"
            "#.....D..#B##\n"
            "###.###..#..#\n"
            "#P..h....z..#\n"
            "#############",
            player_angle=0,
            description="Zombieman in hub, imp guarding blue key, backtrack through door to blue exit.",
        ),
        MapSpec(
            "proc_t4_gauntlet_pillars", 4,
            "#############\n"
            "#..#....#..E#\n"
            "#P.D.i..Y...#\n"
            "#..#....#...#\n"
            "###D#########\n"
            "#..y.h#\n"
            "#######",
            player_angle=0,
            description="Pillar arena with imp + yellow key room + locked yellow exit.",
        ),
        MapSpec(
            "proc_t4_gauntlet_double", 4,
            "#############\n"
            "#P..z.#r...c#\n"
            "#.....D.....#\n"
            "###D###.#####\n"
            "#h....#i....#\n"
            "#.....R....E#\n"
            "#############",
            player_angle=90,
            description="Full multi-room loop with ammo/health, zombieman, imp, red key, locked door.",
        ),
    ]

    # Add 90-degree rotated + re-oriented variants of training maps so turns aren't memorized
    expanded_train: list[MapSpec] = []
    for i, spec in enumerate(train_templates):
        expanded_train.append(spec)
        rot_k = (i % 3) + 1
        rot_angle = (spec.player_angle + 90 * rot_k) % 360
        expanded_train.append(
            MapSpec(
                f"{spec.name}_rot{rot_k * 90}",
                spec.tier,
                _rotate_grid(spec.grid, rot_k),
                player_angle=rot_angle,
                description=f"{spec.description} (rotated {rot_k * 90} deg)",
            )
        )

    # Strictly held-out evaluation maps (distinct geometries never seen in training)
    heldout_specs = [
        MapSpec(
            "heldout_t1_combat", 1,
            "#########\n"
            "#...i...#\n"
            "#..#.#..#\n"
            "#P..h..z#\n"
            "#......E#\n"
            "#########",
            player_angle=90,
            description="Held-out Tier 1: imp + zombieman + cover + stimpack + exit.",
        ),
        MapSpec(
            "heldout_t1_ambush", 1,
            "#########\n"
            "#z..#..E#\n"
            "#...#...#\n"
            "#..P..i.#\n"
            "#..h....#\n"
            "#########",
            player_angle=180,
            description="Held-out Tier 1: player facing away from imp/exit.",
        ),
        MapSpec(
            "heldout_t2_explore", 2,
            "###########\n"
            "#E..#....z#\n"
            "#...D.....#\n"
            "#####.###D#\n"
            "#P..h.#...#\n"
            "#.....D...#\n"
            "###########",
            player_angle=0,
            description="Held-out Tier 2: 4-room exploration loop with 3 doors and zombieman.",
        ),
        MapSpec(
            "heldout_t2_branch", 2,
            "###########\n"
            "#P....#..E#\n"
            "#..#..D...#\n"
            "#..#..#####\n"
            "###D###\n"
            "#h...z#\n"
            "#######",
            player_angle=270,
            description="Held-out Tier 2: branch room + pillar room + exit door.",
        ),
        MapSpec(
            "heldout_t3_key_backtrack", 3,
            "###########\n"
            "#P....#..r#\n"
            "#.....D...#\n"
            "###R#######\n"
            "#...#..E#\n"
            "#...D...#\n"
            "#########",
            player_angle=0,
            description="Held-out Tier 3: red key behind east door, backtrack to south red door + exit door.",
        ),
        MapSpec(
            "heldout_t3_blue_maze", 3,
            "###########\n"
            "#E.B......#\n"
            "###.###.###\n"
            "#P....#..b#\n"
            "#.....D...#\n"
            "###########",
            player_angle=180,
            description="Held-out Tier 3: player starts facing west, blue key east, blue exit north-west.",
        ),
        MapSpec(
            "heldout_t4_gauntlet_a", 4,
            "#############\n"
            "#P..h.#i...E#\n"
            "#.....B.....#\n"
            "###D#########\n"
            "#z..b.#\n"
            "#######",
            player_angle=270,
            description="Held-out Tier 4: zombieman guards blue key, imp guards exit behind blue door.",
        ),
        MapSpec(
            "heldout_t4_gauntlet_b", 4,
            "#############\n"
            "#r.z.#......#\n"
            "#....D...#R.#\n"
            "#####.###.#.#\n"
            "#P..h.i...#E#\n"
            "#############",
            player_angle=90,
            description="Held-out Tier 4: hub combat + side red key + locked red exit corridor.",
        ),
    ]
    return expanded_train, heldout_specs


def generate_all(
    train_dir: str | Path = "maps/training",
    heldout_dir: str | Path = "maps/heldout",
) -> tuple[list[Path], list[Path]]:
    train_specs, heldout_specs = build_curriculum_specs()
    train_root = Path(train_dir)
    heldout_root = Path(heldout_dir)
    train_paths = [
        write_udmf_wad(train_root / f"{spec.name}.wad", spec.grid, player_angle=spec.player_angle)
        for spec in train_specs
    ]
    heldout_paths = [
        write_udmf_wad(heldout_root / f"{spec.name}.wad", spec.grid, player_angle=spec.player_angle)
        for spec in heldout_specs
    ]
    return train_paths, heldout_paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate procedural UDMF training and held-out maps.")
    parser.add_argument("--train-dir", default="maps/training")
    parser.add_argument("--heldout-dir", default="maps/heldout")
    args = parser.parse_args(list(argv) if argv is not None else None)
    train_paths, heldout_paths = generate_all(args.train_dir, args.heldout_dir)
    print(f"Generated {len(train_paths)} training WADs in {args.train_dir} and {len(heldout_paths)} held-out WADs in {args.heldout_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _bfs_reachable(board: list[list[str]], start: tuple[int, int], blocked: set[str]) -> set[tuple[int, int]]:
    h, w = len(board), len(board[0])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and board[nr][nc] not in blocked:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def generate_random_map(rng: np.random.Generator, tier: int | None = None) -> MapSpec:
    """Generate a guaranteed-solvable random multi-room ASCII layout for Tier 1-4."""
    if tier is None:
        # Weight toward multi-room exploration, keycard backtracking, and combat gauntlets
        tier = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.25, 0.30, 0.30]))

    for _attempt in range(30):
        if tier == 1:
            rh = int(rng.integers(5, 8))
            rw = int(rng.integers(6, 10))
            board = [["#"] * (rw + 2) for _ in range(rh + 2)]
            for r in range(1, rh + 1):
                for c in range(1, rw + 1):
                    board[r][c] = "."
            # Optional pillars away from borders
            if rh >= 5 and rw >= 6:
                for _ in range(int(rng.integers(0, 3))):
                    pr, pc = int(rng.integers(2, rh)), int(rng.integers(2, rw))
                    board[pr][pc] = "#"
            free = [(r, c) for r in range(1, rh + 1) for c in range(1, rw + 1) if board[r][c] == "."]
            rng.shuffle(free)
            if len(free) < 8:
                continue
            pr, pc = free[0]
            # Put exit far from player
            free_sorted = sorted(free[1:], key=lambda rc: abs(rc[0] - pr) + abs(rc[1] - pc), reverse=True)
            er, ec = free_sorted[0]
            board[pr][pc] = "P"
            board[er][ec] = "E"
            rem = [rc for rc in free_sorted[1:] if abs(rc[0] - pr) + abs(rc[1] - pc) >= 3]
            n_enemies = int(rng.integers(1, 4))
            for idx in range(min(n_enemies, len(rem))):
                er_i, ec_i = rem[idx]
                board[er_i][ec_i] = str(rng.choice(["i", "z", "s"]))
            if len(rem) > n_enemies:
                ir, ic = rem[n_enemies]
                board[ir][ic] = str(rng.choice(["h", "c", "g", "m"]))
            reach = _bfs_reachable(board, (pr, pc), {"#"})
            if (er, ec) in reach:
                grid = "\n".join("".join(row) for row in board)
                return MapSpec(f"rand_t1", 1, grid, player_angle=int(rng.integers(0, 360)))
            continue

        # Tiers 2, 3, 4: macro-grid of rooms separated by 1-cell walls
        macro_r = int(rng.choice([2, 3]))
        macro_c = 3 if macro_r == 2 else int(rng.choice([2, 3]))
        cell_h = int(rng.choice([3, 4]))
        cell_w = int(rng.choice([3, 4]))
        H = macro_r * cell_h + (macro_r + 1)
        W = macro_c * cell_w + (macro_c + 1)
        board = [["#"] * W for _ in range(H)]

        target_rooms = int(rng.integers(3, min(6, macro_r * macro_c) + 1))
        if tier >= 3:
            target_rooms = max(target_rooms, 3)

        start_node = (int(rng.integers(0, macro_r)), int(rng.integers(0, macro_c)))
        selected = {start_node}
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {start_node: []}
        frontier = [start_node]
        while len(selected) < target_rooms and frontier:
            u = frontier[int(rng.integers(0, len(frontier)))]
            nbrs = [
                (u[0] + dr, u[1] + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= u[0] + dr < macro_r and 0 <= u[1] + dc < macro_c and (u[0] + dr, u[1] + dc) not in selected
            ]
            if not nbrs:
                frontier.remove(u)
                continue
            v = nbrs[int(rng.integers(0, len(nbrs)))]
            selected.add(v)
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
            frontier.append(v)

        if len(selected) < 3:
            continue

        def room_box(mr: int, mc: int) -> tuple[int, int, int, int]:
            r0 = 1 + mr * (cell_h + 1)
            c0 = 1 + mc * (cell_w + 1)
            return r0, r0 + cell_h, c0, c0 + cell_w

        room_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for node in selected:
            r0, r1, c0, c1 = room_box(*node)
            cells = []
            for r in range(r0, r1):
                for c in range(c0, c1):
                    board[r][c] = "."
                    cells.append((r, c))
            # Occasional center pillar in 4x4 rooms
            if cell_h == 4 and cell_w == 4 and rng.random() < 0.35:
                pr_p = r0 + int(rng.integers(1, 3))
                pc_p = c0 + int(rng.integers(1, 3))
                board[pr_p][pc_p] = "#"
                cells.remove((pr_p, pc_p))
            room_cells[node] = cells

        # Choose leaf node for Exit (and non-exit node for key in Tier 3/4)
        leaves = [n for n, nbrs in adj.items() if len(nbrs) == 1]
        if not leaves:
            leaves = list(selected)
        exit_node = leaves[int(rng.integers(0, len(leaves)))]
        other_nodes = [n for n in selected if n != exit_node]
        # Prefer a non-adjacent / distant leaf or node for player & key
        player_node = other_nodes[int(rng.integers(0, len(other_nodes)))]
        key_candidates = [n for n in other_nodes if n != player_node]
        key_node = key_candidates[int(rng.integers(0, len(key_candidates)))] if key_candidates else player_node

        lock_pairs = [("R", "r"), ("B", "b"), ("Y", "y")]
        locked_char, key_char = lock_pairs[int(rng.integers(0, len(lock_pairs)))]

        # Carve doorways along spanning tree edges
        emitted_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for u, nbrs in adj.items():
            for v in nbrs:
                edge = (min(u, v), max(u, v))
                if edge in emitted_edges:
                    continue
                emitted_edges.add(edge)
                ur0, ur1, uc0, uc1 = room_box(*u)
                vr0, vr1, vc0, vc1 = room_box(*v)
                is_exit_edge = exit_node in (u, v)
                if tier >= 3 and is_exit_edge:
                    door_ch = locked_char
                else:
                    door_ch = "D" if rng.random() < 0.8 else "."
                if u[0] == v[0]:
                    # Horizontal neighbors -> shared vertical wall
                    wc = uc1 if v[1] > u[1] else vc1
                    wr = int(rng.integers(ur0, ur1))
                else:
                    # Vertical neighbors -> shared horizontal wall
                    wr = ur1 if v[0] > u[0] else vr1
                    wc = int(rng.integers(uc0, uc1))
                board[wr][wc] = door_ch

        # Place P, E, key, enemies, items inside their assigned room cells
        p_rc = room_cells[player_node][int(rng.integers(0, len(room_cells[player_node])))]
        e_rc = room_cells[exit_node][int(rng.integers(0, len(room_cells[exit_node])))]
        board[p_rc[0]][p_rc[1]] = "P"
        board[e_rc[0]][e_rc[1]] = "E"

        if tier >= 3:
            k_pool = [rc for rc in room_cells[key_node] if board[rc[0]][rc[1]] == "."]
            if not k_pool:
                continue
            k_rc = k_pool[int(rng.integers(0, len(k_pool)))]
            board[k_rc[0]][k_rc[1]] = key_char
        else:
            k_rc = None

        # Enemies & pickups in Tier 2/4 (and occasional single enemy in Tier 3)
        enemy_budget = 0 if tier == 3 and rng.random() < 0.5 else (1 if tier in (2, 3) else int(rng.integers(2, 5)))
        all_free = [
            (r, c)
            for node in selected
            for (r, c) in room_cells[node]
            if board[r][c] == "." and abs(r - p_rc[0]) + abs(c - p_rc[1]) >= 2
        ]
        rng.shuffle(all_free)
        for idx in range(min(enemy_budget, len(all_free))):
            er, ec = all_free[idx]
            board[er][ec] = str(rng.choice(["z", "i", "s"]))
        if len(all_free) > enemy_budget and rng.random() < 0.7:
            hr, hc = all_free[enemy_budget]
            board[hr][hc] = str(rng.choice(["h", "m", "c", "a"]))

        # Verify solvability with 2-phase BFS
        reach_no_lock = _bfs_reachable(board, p_rc, {"#", "R", "B", "Y", "E"})
        if tier >= 3 and k_rc not in reach_no_lock:
            continue
        reach_all = _bfs_reachable(board, p_rc, {"#"})
        if e_rc not in reach_all:
            continue

        grid = "\n".join("".join(row) for row in board)
        return MapSpec(f"rand_t{tier}", tier, grid, player_angle=int(rng.integers(0, 360)))

    raise RuntimeError(f"Failed to generate solvable random map for tier {tier}")
