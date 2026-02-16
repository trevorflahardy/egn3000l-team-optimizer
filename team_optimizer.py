"""
EGN3000L Team Formation Optimizer
=================================
Reads student survey responses and creates optimal teams of 5.

Optimization criteria:
  1. Role preferences — place students in roles they ranked highest,
     weighted by their self-reported interest level.
  2. Robot theme matching — group students wanting similar robot themes.
  3. Major similarity — group students in similar majors (lower priority).
  4. 3D-printer cap — at most one printer-owner per team (unless every
     team already has one).
  5. Worst offenders — students with >= ABSENCE_THRESHOLD absences are
     placed together in the last team(s).

Output: CSV with one row per team, columns = the five role names,
        cells = student full names.
"""

import csv
import random
import os
import sys
import time
from itertools import permutations as _py_permutations
from collections import Counter, defaultdict

import numpy as np
from numba import njit, types, int32, int64, float64, boolean
from numba.typed import List as NumbaList

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
ROLES = [
    "Design Engineering Lead",
    "Project Engineering Lead",
    "Test Engineering Lead",
    "Product Development (hardware) Lead",
    "Software Engineering Lead",
]

TEAM_SIZE = 5
ABSENCE_THRESHOLD = 3  # >= this  ->  worst offender

# Weights for the combined team-quality score
W_ROLE = 10.0  # role-preference satisfaction
W_THEME = 8.0  # robot-theme cohesion
W_MAJOR = 2.0  # major similarity

SWAP_ITERATIONS = 10_000_000  # local-search budget
NUM_RESTARTS = 10  # full greedy+search restarts
RANDOM_SEED = 42


def _int(val):
    """Safely parse an integer, returning 0 on failure."""
    try:
        return int(str(val).strip())
    except (ValueError, AttributeError):
        return 0


def normalize_major(major: str) -> str:
    """Collapse related major names into a single key."""
    m = major.upper()
    for kw in (
        "MECHANICAL",
        "ELECTRICAL",
        "CIVIL",
        "INDUSTRIAL",
        "CHEMICAL",
        "BIOMEDICAL",
        "ENVIRONMENTAL",
        "AEROSPACE",
        "COMPUTER",
    ):
        if kw in m:
            return kw
    return m or "UNKNOWN"


def load_students(path: str) -> list[dict]:
    students = []
    with open(path, "r", encoding="cp1252") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            s = dict(
                id=i,
                first_name=row.get("What is your first name?", "").strip(),
                last_name=row.get("What is your last name?", "").strip(),
                uid=row.get("What is your U#?", "").strip(),
                major=row.get("What is your current major?", "").strip(),
                choice1=row.get("First Appealing", "").strip(),
                choice2=row.get("Second Appealing", "").strip(),
                choice3=row.get("Third Appealing", "").strip(),
                interest1=_int(row.get("Your first choice interest level (1-10)", "")),
                interest2=_int(row.get("Your second choice interest level (1-10)", "")),
                interest3=_int(row.get("Your third choice interest level (1-10)", "")),
                theme=row.get("Robot Design Theme", "").strip(),
                has_printer=row.get("Has 3D Printer?", "").strip().lower() == "yes",
                absences=_int(row.get("Total Absences", "0")),
            )
            s["name"] = f"{s['first_name']} {s['last_name']}"
            s["has_data"] = bool(s["choice1"])
            students.append(s)
    return students


# ═══════════════════════════════════════════════════════════════
# Precomputed numerical arrays  (built once, used by Numba)
# ═══════════════════════════════════════════════════════════════
# All 120 permutations of [0..4] — shape (120, 5), dtype int32
_PERMS = np.array(list(_py_permutations(range(TEAM_SIZE))), dtype=np.int32)


def build_arrays(students: list[dict]):
    """
    Convert the list-of-dicts into flat NumPy arrays for the JIT kernel.
    Returns a dict with arrays keyed by name.
    """
    n = len(students)
    # role_score_matrix[i, r] = score if student i is placed in role r
    rsm = np.zeros((n, TEAM_SIZE), dtype=np.int32)
    role_idx = {r: j for j, r in enumerate(ROLES)}

    # theme / major  → integer id  (0 = none)
    theme_map: dict[str, int] = {}
    major_map: dict[str, int] = {}
    theme_ids = np.zeros(n, dtype=np.int32)
    major_ids = np.zeros(n, dtype=np.int32)
    printer = np.zeros(n, dtype=np.int8)

    for s in students:
        i = s["id"]
        # role score matrix
        for r_idx, role in enumerate(ROLES):
            if not s["has_data"]:
                rsm[i, r_idx] = 1
            elif s["choice1"] == role:
                rsm[i, r_idx] = s["interest1"] * 3
            elif s["choice2"] == role:
                rsm[i, r_idx] = s["interest2"] * 2
            elif s["choice3"] == role:
                rsm[i, r_idx] = s["interest3"] * 1
            else:
                rsm[i, r_idx] = 0

        # theme → int
        th = s["theme"]
        if th:
            if th not in theme_map:
                theme_map[th] = len(theme_map) + 1
            theme_ids[i] = theme_map[th]

        # major → int
        m = normalize_major(s["major"])
        if m and m != "UNKNOWN":
            if m not in major_map:
                major_map[m] = len(major_map) + 1
            major_ids[i] = major_map[m]

        printer[i] = 1 if s["has_printer"] else 0

    return dict(
        rsm=rsm, theme_ids=theme_ids, major_ids=major_ids, printer=printer, perms=_PERMS
    )


# ═══════════════════════════════════════════════════════════════
# Numba-accelerated scoring
# ═══════════════════════════════════════════════════════════════
@njit(int32(int32[:, :], int32[:], int32[:, :]), cache=True)
def _fast_best_role_score(rsm, team_ids, perms):
    """Return the best role-assignment score for a team of 5 student ids."""
    best = -1
    for pi in range(perms.shape[0]):  # 120 iterations
        s = int32(0)
        for r in range(5):
            s += rsm[team_ids[perms[pi, r]], r]
        if s > best:
            best = s
    return best


@njit(int32(int32[:], int32[:]), cache=True)
def _count_pairs(ids, attr):
    """Count matching-attribute pairs in a team of 5."""
    pairs = int32(0)
    for i in range(5):
        ai = attr[ids[i]]
        if ai == 0:
            continue
        for j in range(i + 1, 5):
            if attr[ids[j]] == ai:
                pairs += 1
    return pairs


@njit(
    float64(
        int32[:, :],
        int32[:],
        int32[:],
        int32[:],
        int32[:, :],
        float64,
        float64,
        float64,
    ),
    cache=True,
)
def _fast_team_score(
    rsm, team_ids, theme_ids, major_ids, perms, w_role, w_theme, w_major
):
    rs = _fast_best_role_score(rsm, team_ids, perms)
    tp = _count_pairs(team_ids, theme_ids)
    mp = _count_pairs(team_ids, major_ids)
    return w_role * rs + w_theme * tp + w_major * mp


@njit(cache=True)
def _fast_local_search(
    team_arr,
    rsm,
    theme_ids,
    major_ids,
    printer,
    perms,
    w_role,
    w_theme,
    w_major,
    iters,
    allow_double,
    seed,
):
    """
    Numba-accelerated pairwise-swap hill climber.
    team_arr : int32[n_teams, 5]  — student IDs per team
    Returns  : (team_arr, improved_count, total_score)
    """
    np.random.seed(seed)
    n_teams = team_arr.shape[0]
    if n_teams < 2:
        return team_arr, 0, 0.0

    # pre-compute team scores
    scores = np.empty(n_teams, dtype=np.float64)
    for t in range(n_teams):
        scores[t] = _fast_team_score(
            rsm, team_arr[t], theme_ids, major_ids, perms, w_role, w_theme, w_major
        )

    improved = 0
    for _ in range(iters):
        # pick two distinct teams
        t1 = np.random.randint(0, n_teams)
        t2 = np.random.randint(0, n_teams - 1)
        if t2 >= t1:
            t2 += 1
        i1 = np.random.randint(0, 5)
        i2 = np.random.randint(0, 5)

        a = team_arr[t1, i1]
        b = team_arr[t2, i2]

        # printer guard
        if not allow_double:
            if printer[b] == 1:
                cnt = int32(0)
                for k in range(5):
                    sid = b if k == i1 else team_arr[t1, k]
                    cnt += printer[sid]
                if cnt > 1:
                    continue
            if printer[a] == 1:
                cnt = int32(0)
                for k in range(5):
                    sid = a if k == i2 else team_arr[t2, k]
                    cnt += printer[sid]
                if cnt > 1:
                    continue

        # tentative swap
        team_arr[t1, i1] = b
        team_arr[t2, i2] = a
        ns1 = _fast_team_score(
            rsm, team_arr[t1], theme_ids, major_ids, perms, w_role, w_theme, w_major
        )
        ns2 = _fast_team_score(
            rsm, team_arr[t2], theme_ids, major_ids, perms, w_role, w_theme, w_major
        )

        if ns1 + ns2 > scores[t1] + scores[t2]:
            scores[t1] = ns1
            scores[t2] = ns2
            improved += 1
        else:
            team_arr[t1, i1] = a
            team_arr[t2, i2] = b

    total = 0.0
    for t in range(n_teams):
        total += scores[t]
    return team_arr, improved, total


# ═══════════════════════════════════════════════════════════════
# Python wrappers (used outside the hot loop)
# ═══════════════════════════════════════════════════════════════
def role_score(student: dict, role: str) -> int:
    """
    How well *role* matches *student*'s preferences.
    1st choice x 3, 2nd x 2, 3rd x 1.  Filler students get 1.
    """
    if not student["has_data"]:
        return 1
    if student["choice1"] == role:
        return student["interest1"] * 3
    if student["choice2"] == role:
        return student["interest2"] * 2
    if student["choice3"] == role:
        return student["interest3"] * 1
    return 0


def best_role_assignment(team: list[dict]):
    """
    Brute-force over 5! = 120 permutations.
    Returns (score, {role: student}).
    """
    if len(team) != TEAM_SIZE:
        asgn = {ROLES[j]: team[j] for j in range(len(team))}
        return 0, asgn

    best, best_p = -1, None
    for p in _py_permutations(range(TEAM_SIZE)):
        s = sum(role_score(team[p[r]], ROLES[r]) for r in range(TEAM_SIZE))
        if s > best:
            best, best_p = s, p
    asgn = {ROLES[r]: team[best_p[r]] for r in range(TEAM_SIZE)}
    return best, asgn


def _theme_pairs(team):
    """# matching-theme pairs (max C(5,2)=10)."""
    themes = [s["theme"] for s in team if s["theme"]]
    if not themes:
        return 0
    return sum(c * (c - 1) // 2 for c in Counter(themes).values())


def _major_pairs(team):
    """# matching-major pairs."""
    majors = [normalize_major(s["major"]) for s in team if s["major"]]
    if not majors:
        return 0
    return sum(c * (c - 1) // 2 for c in Counter(majors).values())


def team_score(team):
    """Combined quality metric."""
    if len(team) != TEAM_SIZE:
        return -1e9
    rs, _ = best_role_assignment(team)
    return W_ROLE * rs + W_THEME * _theme_pairs(team) + W_MAJOR * _major_pairs(team)


# ═══════════════════════════════════════════════════════════════
# Team formation — greedy with role-diversity heuristic
# ═══════════════════════════════════════════════════════════════
def _greedy_form_teams(pool: list[dict], allow_printer_double: bool):
    """
    Greedily partition *pool* into teams of TEAM_SIZE.
    Returns (teams, leftover).
    """
    pool = list(pool)
    random.shuffle(pool)
    teams: list[list[dict]] = []
    consecutive_fails = 0
    MAX_FAILS = len(pool) + 5  # upper-bound on retries

    while len(pool) >= TEAM_SIZE and consecutive_fails < MAX_FAILS:
        # --- pick seed: prefer student whose 1st-choice is rarest in pool ---
        role_pop = Counter(s["choice1"] for s in pool if s["has_data"])

        def _rarity(s):
            if not s["has_data"]:
                return 999  # lower count = rarer = picked first
            return role_pop.get(s["choice1"], 0)

        pool.sort(key=_rarity)
        seed = pool.pop(0)
        team = [seed]

        # --- greedily add 4 more members ---
        for _ in range(TEAM_SIZE - 1):
            best_j, best_sc = None, -1e9
            for j, cand in enumerate(pool):
                # printer guard (skip if constraint active and would double up)
                if (
                    not allow_printer_double
                    and cand["has_printer"]
                    and any(m["has_printer"] for m in team)
                ):
                    continue

                # role diversity: does this candidate bring a new 1st-choice?
                covered = {m["choice1"] for m in team if m["has_data"]}
                new_role = (
                    cand["has_data"]
                    and cand["choice1"]
                    and cand["choice1"] not in covered
                )
                sc = 12 if new_role else 0

                # theme match with existing members
                sc += sum(
                    3
                    for m in team
                    if m["theme"] and cand["theme"] and m["theme"] == cand["theme"]
                )

                # major match
                sc += sum(
                    1
                    for m in team
                    if m["major"]
                    and cand["major"]
                    and normalize_major(m["major"]) == normalize_major(cand["major"])
                )

                if sc > best_sc:
                    best_sc, best_j = sc, j

            if best_j is not None:
                team.append(pool.pop(best_j))
            else:
                # can't fill — put seed back, shuffle, retry
                pool.append(seed)
                team = []
                break

        if len(team) == TEAM_SIZE:
            teams.append(team)
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            random.shuffle(pool)

    # --- fallback: if printer constraint blocked everything, relax it ----
    if len(pool) >= TEAM_SIZE and not allow_printer_double:
        fb_teams, pool = _greedy_form_teams(pool, allow_printer_double=True)
        teams.extend(fb_teams)

    return teams, pool


# ═══════════════════════════════════════════════════════════════
# Local search — Numba-accelerated pairwise swap hill-climber
# ═══════════════════════════════════════════════════════════════
def _local_search(teams, iters, allow_printer_double, arrays, rng_seed=0):
    """
    Delegates to the @njit kernel for speed.
    *teams* is a list of list[dict].  Only full-sized teams are optimised;
    partial/bad teams pass through untouched.
    """
    full_teams = [t for t in teams if len(t) == TEAM_SIZE]
    other_teams = [t for t in teams if len(t) != TEAM_SIZE]
    if len(full_teams) < 2:
        return teams

    # build int32 team array  (n_full × 5)
    team_arr = np.array([[s["id"] for s in t] for t in full_teams], dtype=np.int32)

    t0 = time.perf_counter()
    team_arr, improved, total = _fast_local_search(
        team_arr,
        arrays["rsm"],
        arrays["theme_ids"],
        arrays["major_ids"],
        arrays["printer"],
        arrays["perms"],
        W_ROLE,
        W_THEME,
        W_MAJOR,
        iters,
        allow_printer_double,
        rng_seed,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"    swaps accepted: {improved},  total score: {total:.0f}"
        f"  ({elapsed:.2f}s, {iters/elapsed/1e6:.1f}M iter/s)"
    )

    # rebuild list-of-dicts teams from the optimised int array
    id_to_student = {}
    for t in full_teams:
        for s in t:
            id_to_student[s["id"]] = s

    new_full = []
    for ti in range(team_arr.shape[0]):
        new_full.append([id_to_student[int(team_arr[ti, j])] for j in range(TEAM_SIZE)])

    return new_full + other_teams


# ═══════════════════════════════════════════════════════════════
# Full pipeline (one run)
# ═══════════════════════════════════════════════════════════════
def _build_solution(students, arrays, rng_seed=0, verbose=True):
    """
    Execute the full team-formation pipeline once for the current
    random state.  Returns (all_teams, all_assignments, total_score).
    """
    n = len(students)
    total_teams = n // TEAM_SIZE
    remainder = n % TEAM_SIZE

    # -- 3D-printer math -------------------------------------------------
    printer_count = sum(1 for s in students if s["has_printer"])
    allow_double = printer_count >= total_teams

    # -- Step 1: worst offenders -----------------------------------------
    offenders = sorted(
        [s for s in students if s["absences"] >= ABSENCE_THRESHOLD],
        key=lambda s: -s["absences"],
    )
    rest = [s for s in students if s["absences"] < ABSENCE_THRESHOLD]

    # -- Step 2: filler students (no survey data, not offenders) ----------
    filler = [s for s in rest if not s["has_data"]]
    regular = [s for s in rest if s["has_data"]]

    # -- Step 3: bad team(s) ----------------------------------------------
    # Worst offenders form their own team.  If there are fewer than
    # TEAM_SIZE offenders the team is intentionally undersized — these
    # students are not meaningfully participating.
    bad_teams: list[list[dict]] = []
    off_remain = list(offenders)
    filler_remain = list(filler)

    while len(off_remain) >= TEAM_SIZE:
        bad_teams.append(off_remain[:TEAM_SIZE])
        off_remain = off_remain[TEAM_SIZE:]

    if off_remain:
        # Keep the bad team undersized rather than sacrificing a
        # filler/regular student.  All fillers feed back into the
        # general pool so real teams each get 5 members.
        bad_teams.append(off_remain)

    if verbose:
        for i, t in enumerate(bad_teams):
            print(
                f"  Bad team {i+1}: "
                + ", ".join(f"{s['name']}(abs={s['absences']})" for s in t)
            )

    # -- Step 4: cluster by theme -----------------------------------------
    clusters: dict[str, list[dict]] = defaultdict(list)
    no_theme: list[dict] = []
    for s in regular:
        if s["theme"]:
            clusters[s["theme"]].append(s)
        else:
            no_theme.append(s)

    leftover_pool: list[dict] = list(no_theme) + list(filler_remain)

    if verbose:
        for th in sorted(clusters, key=lambda k: -len(clusters[k])):
            print(f"  Theme '{th}': {len(clusters[th])} students")
        if leftover_pool:
            print(f"  No theme / filler: {len(leftover_pool)}")

    # -- Step 5: form teams per theme cluster -----------------------------
    good_teams: list[list[dict]] = []
    for th in sorted(clusters, key=lambda k: -len(clusters[k])):
        formed, left = _greedy_form_teams(clusters[th], allow_double)
        good_teams.extend(formed)
        leftover_pool.extend(left)
        if verbose:
            print(f"    {th}: {len(formed)} teams, {len(left)} leftover")

    # -- Step 6: teams from the leftover pool -----------------------------
    if leftover_pool:
        formed, final_left = _greedy_form_teams(leftover_pool, allow_double)
        good_teams.extend(formed)
        if final_left:
            if verbose:
                print(
                    f"    {len(final_left)} students still unplaced — "
                    "forming partial team"
                )
            good_teams.append(final_left)

    # -- Step 7: local search (Numba-accelerated) -------------------------
    good_teams = _local_search(
        good_teams, SWAP_ITERATIONS, allow_double, arrays, rng_seed=rng_seed
    )

    # -- Combine (good first, bad last) -----------------------------------
    all_teams = good_teams + bad_teams

    # -- Role assignment --------------------------------------------------
    all_assignments = []
    for team in all_teams:
        _, asgn = best_role_assignment(team)
        all_assignments.append(asgn)

    total = sum(team_score(t) for t in all_teams if len(t) == TEAM_SIZE)
    return all_teams, all_assignments, total


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════
def validate(teams, all_students, verbose=True):
    errors: list[str] = []

    # duplicates / missing
    placed_ids = []
    for t in teams:
        for s in t:
            placed_ids.append(s["id"])
    if len(placed_ids) != len(set(placed_ids)):
        dup = [x for x in placed_ids if placed_ids.count(x) > 1]
        errors.append(f"Duplicate IDs: {set(dup)}")

    all_ids = {s["id"] for s in all_students}
    missing = all_ids - set(placed_ids)
    if missing:
        names = [s["name"] for s in all_students if s["id"] in missing]
        errors.append(f"Missing students ({len(missing)}): {names}")

    extra = set(placed_ids) - all_ids
    if extra:
        errors.append(f"Unknown IDs in teams: {extra}")

    # team sizes (last team(s) may be undersized — bad teams)
    num_bad = sum(
        1
        for t in teams
        if any(s["absences"] >= ABSENCE_THRESHOLD for s in t) and len(t) < TEAM_SIZE
    )
    for i, t in enumerate(teams):
        is_bad = any(s["absences"] >= ABSENCE_THRESHOLD for s in t)
        if len(t) != TEAM_SIZE and not is_bad:
            errors.append(f"Team {i+1} has {len(t)} members (expected {TEAM_SIZE})")
        elif len(t) != TEAM_SIZE and is_bad and verbose:
            print(
                f"  NOTE   Bad team {i+1} has {len(t)} members (offenders-only, intentional)"
            )

    # 3D printer (at most 1 per team unless every team has one)
    total_printers = sum(1 for s in all_students if s["has_printer"])
    if total_printers < len(teams):
        for i, t in enumerate(teams):
            pc = sum(1 for s in t if s["has_printer"])
            if pc > 1:
                names = [s["name"] for s in t if s["has_printer"]]
                errors.append(f"Team {i+1} has {pc} 3D-printer owners: {names}")

    if verbose:
        for e in errors:
            print(f"  ERROR  {e}")
        if not errors:
            print("  All checks passed!")

    return errors


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════
def write_output(teams, assignments, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ROLES)
        for asgn in assignments:
            w.writerow([asgn.get(r, {}).get("name", "") for r in ROLES])


def print_team_detail(teams, assignments):
    for i, (team, asgn) in enumerate(zip(teams, assignments)):
        themes = Counter(s["theme"] for s in team if s["theme"])
        top_th = themes.most_common(1)[0][0] if themes else "N/A"
        rs, _ = best_role_assignment(team) if len(team) == TEAM_SIZE else (0, None)
        is_bad = any(s["absences"] >= ABSENCE_THRESHOLD for s in team)
        label = f"Team {i+1}" + (" [ABSENT GROUP]" if is_bad else "")
        print(f"\n  {label}  (role_score={rs}, theme={top_th})")
        for role in ROLES:
            if role not in asgn:
                continue
            s = asgn[role]
            sc = role_score(s, role)
            pref = (
                "1st"
                if s.get("choice1") == role
                else (
                    "2nd"
                    if s.get("choice2") == role
                    else "3rd" if s.get("choice3") == role else "—"
                )
            )
            printer = "3DP" if s["has_printer"] else ""
            print(
                f"    {role:<42s} {s['name']:<28s} "
                f"pref={pref:<3s}  score={sc:>3d}  "
                f"theme={s.get('theme') or 'N/A':<22s}  {printer}"
            )


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "S.26 EGN3000L Teams.csv")
    output_path = os.path.join(script_dir, "S.26 EGN3000L Teams - Output.csv")

    # ── Load ────────────────────────────────────────────────────
    print("=" * 65)
    print("EGN3000L TEAM OPTIMIZER")
    print("=" * 65)
    students = load_students(input_path)
    n = len(students)
    total_teams = n // TEAM_SIZE
    remainder = n % TEAM_SIZE
    print(f"Students loaded       : {n}")
    print(
        f"Teams to form         : {total_teams}"
        + (f"  (+1 partial of {remainder})" if remainder else "")
    )
    printer_count = sum(1 for s in students if s["has_printer"])
    print(f"3D-printer owners     : {printer_count}")
    print(f"Allow doubles         : {printer_count >= total_teams}")
    print(f"Worst-offender cutoff : {ABSENCE_THRESHOLD}+ absences")
    offender_count = sum(1 for s in students if s["absences"] >= ABSENCE_THRESHOLD)
    print(f"Worst offenders       : {offender_count}")
    no_data_count = sum(
        1
        for s in students
        if not bool(s["choice1"]) and s["absences"] < ABSENCE_THRESHOLD
    )
    print(f"No-data fillers       : {no_data_count}")

    # ── Build numerical arrays & JIT warm-up ─────────────────────────────
    print("\nPrecomputing arrays & compiling Numba kernels …", end=" ", flush=True)
    t_jit = time.perf_counter()
    arrays = build_arrays(students)
    # Warm up the JIT with a tiny dummy run so compilation time
    # doesn't count against the first restart.
    _dummy = np.zeros((2, 5), dtype=np.int32)
    _fast_local_search(
        _dummy,
        arrays["rsm"],
        arrays["theme_ids"],
        arrays["major_ids"],
        arrays["printer"],
        arrays["perms"],
        W_ROLE,
        W_THEME,
        W_MAJOR,
        1,
        True,
        0,
    )
    print(f"done ({time.perf_counter() - t_jit:.1f}s)")

    # ── Multi-restart ─────────────────────────────────────────────
    print(f"\nRunning {NUM_RESTARTS} restarts × {SWAP_ITERATIONS:,} swaps …")
    best_teams, best_asgn, best_total = None, None, -1e18

    for restart in range(NUM_RESTARTS):
        random.seed(RANDOM_SEED + restart)
        print(f"\n  — Restart {restart + 1}/{NUM_RESTARTS} —")
        teams, asgns, total = _build_solution(
            students,
            arrays,
            rng_seed=RANDOM_SEED + restart,
            verbose=(restart == 0),
        )
        print(f"    Total score: {total:.0f}")
        if total > best_total:
            best_total = total
            best_teams = [list(t) for t in teams]
            best_asgn = asgns

    print(f"\n{'=' * 65}")
    print(f"Best total score across restarts: {best_total:.0f}")
    print("=" * 65)

    # ── Detailed output ─────────────────────────────────────────
    print("\nFinal Teams:")
    print_team_detail(best_teams, best_asgn)

    # ── Validate ────────────────────────────────────────────────
    print("\n\nValidation:")
    errors = validate(best_teams, students)

    # ── Summary stats ───────────────────────────────────────────
    with_data = [s for s in students if s["has_data"]]
    total_first = sum(
        1
        for asgn in best_asgn
        for r in ROLES
        if r in asgn and asgn[r].get("choice1") == r
    )
    total_top3 = sum(
        1
        for asgn in best_asgn
        for r in ROLES
        if r in asgn
        and r
        in (asgn[r].get("choice1"), asgn[r].get("choice2"), asgn[r].get("choice3"))
    )
    total_role_pts = sum(
        role_score(asgn[r], r) for asgn in best_asgn for r in ROLES if r in asgn
    )

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print("=" * 65)
    print(f"  Students placed         : {sum(len(t) for t in best_teams)}/{n}")
    print(f"  Teams                   : {len(best_teams)}")
    print(f"  Students in 1st-choice  : {total_first}/{len(with_data)}")
    print(f"  Students in top-3 choice: {total_top3}/{len(with_data)}")
    print(f"  Total role-pref points  : {total_role_pts}")
    print(f"  Errors                  : {len(errors)}")

    # ── Write CSV ───────────────────────────────────────────────
    write_output(best_teams, best_asgn, output_path)
    print(f"\n  Output written to: {output_path}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
