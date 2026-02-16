# EGN3000L Team Formation Optimizer

Automatically forms optimal student teams for EGN3000L from survey CSV data. The optimizer uses Numba-accelerated stochastic local search to maximize role satisfaction, theme cohesion, and major similarity across all teams simultaneously.

## Why this exists

Forming teams by hand is slow, biased, and produces mediocre results. With 65 students and 13 teams there are roughly $\frac{65!}{(5!)^{13} \cdot 13!}$ possible partitions — an astronomically large space. This tool explores it systematically so every student gets the best role assignment and most compatible teammates the data allows.

---

## How the optimization works

### Objective function

Each team of 5 receives a **quality score** that combines three sub-scores:

$$S_{\text{team}} = w_r \cdot R + w_t \cdot T + w_m \cdot M$$

| Symbol | Meaning | Default weight |
|--------|---------|----------------|
| $R$ | **Role-preference score** — how well the role assignment matches each student's ranked preferences | $w_r = 10$ |
| $T$ | **Theme-cohesion score** — number of student pairs on the team who want the same robot theme | $w_t = 8$ |
| $M$ | **Major-similarity score** — number of student pairs sharing the same engineering major | $w_m = 2$ |

The **global objective** is the sum of team scores across all teams:

$$S_{\text{total}} = \sum_{k=1}^{K} S_{\text{team}_k}$$

### Role-preference scoring ($R$)

Each student ranks 3 of the 5 roles and rates their interest (1–10) for each. The score for placing student $i$ in role $r$ is:

$$\text{score}(i, r) = \begin{cases} \text{interest}_1 \times 3 & \text{if } r = \text{1st choice} \\ \text{interest}_2 \times 2 & \text{if } r = \text{2nd choice} \\ \text{interest}_3 \times 1 & \text{if } r = \text{3rd choice} \\ 0 & \text{otherwise} \end{cases}$$

For a team of 5, the optimizer tries all $5! = 120$ permutations of role assignments and picks the one with the highest total:

$$R = \max_{\pi \in S_5} \sum_{r=0}^{4} \text{score}(\pi(r), r)$$

This brute-force over permutations is feasible because $5! = 120$ is small, and the inner loop is JIT-compiled with Numba.

### Theme cohesion ($T$) and major similarity ($M$)

These count the number of **matching pairs** within a team:

$$T = \sum_{i < j} \mathbb{1}[\text{theme}_i = \text{theme}_j], \quad M = \sum_{i < j} \mathbb{1}[\text{major}_i = \text{major}_j]$$

A team of 5 has $\binom{5}{2} = 10$ possible pairs. A perfect theme match (all 5 want the same theme) contributes 10 pairs.

### Constraints

- **3D-printer cap**: At most one 3D-printer owner per team (relaxed only if there are more printer owners than teams).
- **Worst-offender isolation**: Students with absences ≥ the threshold are grouped into their own team(s) at the end, so they don't drag down engaged students.

### Search algorithm

1. **Greedy initialization** — Students are sorted by rarest first-choice role. Teams are built one at a time, greedily adding candidates that maximize role diversity, theme overlap, and major overlap.

2. **Stochastic pairwise-swap local search** — A Numba-compiled hill climber runs millions of random swap trials:
   - Pick two random teams, pick one random member from each.
   - Swap them tentatively.
   - If the combined score of the two affected teams improves, keep the swap; otherwise revert.
   - The printer constraint is enforced during swaps.

3. **Multi-restart** — Steps 1–2 repeat with different random seeds. The best solution across all restarts is kept.

The default configuration runs **15 restarts × 50 million swaps each**, which typically takes ~2 minutes on a modern CPU and produces near-optimal results.

---

## Input CSV format

The input is a CSV file exported from a student survey (e.g. Google Forms or Qualtrics). It must have **exactly these column headers** in any order:

| Column | Type | Description |
|--------|------|-------------|
| `What is your first name?` | text | Student first name |
| `What is your last name?` | text | Student last name |
| `What is your U#?` | text | University ID (e.g. `U12345678`) |
| `What is your current major?` | text | e.g. `MECHANICAL ENGINEERING` |
| `CAD Software Skill Level (1-10)` | int | Self-reported skill (informational) |
| `3D Printing Skill Level (1-10)` | int | Self-reported skill (informational) |
| `Coding skill level (1-10)` | int | Self-reported skill (informational) |
| `Circuitry Skill Level (1-10)` | int | Self-reported skill (informational) |
| `Organization Skill Level (1-10)` | int | Self-reported skill (informational) |
| `First Appealing` | text | 1st-choice role (must match a role name exactly) |
| `Second Appealing` | text | 2nd-choice role |
| `Third Appealing` | text | 3rd-choice role |
| `Your first choice interest level (1-10)` | int | Interest in 1st-choice role |
| `Your second choice interest level (1-10)` | int | Interest in 2nd-choice role |
| `Your third choice interest level (1-10)` | int | Interest in 3rd-choice role |
| `Robot Design Theme` | text | Preferred theme (e.g. `Vehicle theme`, `Animals`) |
| `Has 3D Printer?` | text | `Yes` or `No` |
| `Total Absences` | int | Number of absences so far |

### Valid role names

The role choice columns (`First Appealing`, `Second Appealing`, `Third Appealing`) must contain one of:

```
Design Engineering Lead
Project Engineering Lead
Test Engineering Lead
Product Development (hardware) Lead
Software Engineering Lead
```

A reference example file is provided at [`data/example.csv`](data/example.csv).

---

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/egn3000l-team-optimizer.git
cd egn3000l-team-optimizer

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -e .
# or: pip install numpy numba
```

> **Note:** Numba requires a compatible NumPy version. Python 3.10–3.12 are well-supported.

---

## Usage

```bash
# Basic — reads survey CSV and writes output alongside it
python team_optimizer.py data/my_survey.csv

# Specify output path
python team_optimizer.py data/my_survey.csv -o data/teams_output.csv

# Tune the optimizer
python team_optimizer.py data/my_survey.csv \
    --restarts 25 \
    --swaps 100000000 \
    --w-role 12 --w-theme 6 --w-major 3

# Change absence threshold
python team_optimizer.py data/my_survey.csv --absence-threshold 4
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `input` (positional) | *required* | Path to input CSV |
| `-o`, `--output` | `<input> - Output.csv` | Path for output CSV |
| `--absence-threshold` | `3` | Absences ≥ this → worst-offender team |
| `--w-role` | `10.0` | Weight for role-preference satisfaction |
| `--w-theme` | `8.0` | Weight for theme cohesion |
| `--w-major` | `2.0` | Weight for major similarity |
| `--swaps` | `50,000,000` | Swap iterations per restart |
| `--restarts` | `15` | Number of random restarts |
| `--seed` | `42` | Random seed for reproducibility |

---

## Output

The output CSV has one row per team. Columns are the five role names; cells contain the assigned student's full name:

```
Design Engineering Lead,Project Engineering Lead,Test Engineering Lead,Product Development (hardware) Lead,Software Engineering Lead
Jane Doe,Alice Park,Bob Chen,Carlos Ruiz,Diana Lee
...
```

The console also prints detailed per-team breakdowns, validation checks, and summary statistics (% of students in their 1st choice, etc.).

---

## Data privacy

**Student CSV files contain PII** (names, university IDs). The `data/` directory is git-ignored — only the empty `data/.gitkeep` and the synthetic `data/example.csv` are tracked. Never commit real student data.

---

## License

MIT
