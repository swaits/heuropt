# `compare` example — reference output

Snapshot from `cargo run --release --example compare`, refreshed 2026-05-14
for heuropt v0.10.0. 10 seeds per algorithm per problem.

Each table is **sorted best-first** by its primary quality metric. The
live terminal output uses ASCII `+/-` for the mean ± std cells (so column
alignment can't be broken by a terminal that renders `±` at an odd
width); this doc uses `±` since markdown renders it fine.

The **continuous-problem quality metrics** are bit-identical to the
v0.3.0–v0.4.0 snapshots — every optimization pass so far (including the
Phase B CPU work) has been verified bit-identical by the `run()` snapshot
tests. The **ms columns** are the post-Phase-B numbers; SMS-EMOA on DTLZ2
in particular fell ~2.7× from the `hypervolume_nd` rework.

This refresh also adds three **combinatorial / sequencing** problems —
TSP, job-shop scheduling, and a bi-objective knapsack — which exercise the
permutation and bitstring operators and a different algorithm roster (the
real-vector methods can't run them).

Wall-clock numbers are from the development machine and will vary; the
*relative* numbers across algorithms are the interesting part.

---

## ZDT1 (dim=30, 25000 evals/run × 10 seeds)

Zitzler-Deb-Thiele 2-objective benchmark: 30 real variables, one smooth
convex Pareto front `f₂ = 1 − √f₁`. Hard because 29 of 30 variables must
collapse to 0 before the front is even reachable, and only then can the
population spread along it. Optimum: mean L2 → 0 (the front is known
exactly). Sorted by hypervolume (reference `[11, 11]`).

| algorithm    | hypervolume ↑       | spacing ↓       | mean L2 ↓       | front | ms  |
|---|---|---|---|---|---|
| MOPSO        | **120.6149 ± 0.0529** | 0.0125 ± 0.0025 | **0.0005 ± 0.0001** | 100 | 80  |
| IBEA         | 120.0167 ± 0.3112   | 0.0130 ± 0.0027 | 0.0448 ± 0.0168 | 73    | 130 |
| MOEA/D       | 119.9450 ± 0.4953   | 0.0118 ± 0.0013 | 0.0065 ± 0.0020 | 96    | 27  |
| PESA-II      | 119.3670 ± 0.3261   | **0.0095 ± 0.0011** | 0.0802 ± 0.0354 | 100 | 67 |
| eps-MOEA     | 118.8742 ± 0.6835   | 0.0167 ± 0.0058 | 0.0493 ± 0.0227 | 45    | 46  |
| NSGA-II      | 118.3336 ± 0.7750   | 0.0112 ± 0.0022 | 0.1891 ± 0.0599 | 96    | 40  |
| SPEA2        | 118.0823 ± 0.5973   | 0.0111 ± 0.0023 | 0.2408 ± 0.0509 | 97    | 226 |
| NSGA-III     | 115.1612 ± 0.4745   | 0.0139 ± 0.0029 | 0.4314 ± 0.0582 | 86    | 47  |
| RVEA         | 111.7151 ± 1.8195   | 0.0308 ± 0.0099 | 0.8399 ± 0.1569 | 47    | 62  |
| HypE         | 105.6489 ± 0.9789   | 0.0266 ± 0.0053 | 1.4820 ± 0.1003 | 72    | 30  |
| PAES         | 104.1887 ± 0.8953   | 0.0351 ± 0.0067 | 1.3195 ± 0.0558 | 33    | 27  |
| SMS-EMOA     | 102.8871 ± 1.0543   | 0.0263 ± 0.0039 | 1.4937 ± 0.1192 | 40    | 54  |
| RandomSearch | 99.5691 ± 0.9383    | 0.0937 ± 0.0347 | 2.3621 ± 0.1428 | 28    | 88  |

**MOPSO and MOEA/D dominate** convergence (mean L2 to true front ≤ 0.01).
PESA-II edges spacing.

## ZDT3 (dim=30, 25000 evals × 10 seeds)

Zitzler-Deb-Thiele 2-objective with a **disconnected** front: five
separate arcs rather than one curve. Hard because an algorithm has to
discover and populate every arc while not stranding solutions in the
dominated gaps between them.

| algorithm | hypervolume ↑       | spacing ↓       | front | ms  |
|---|---|---|---|---|
| **IBEA**  | **126.2072 ± 1.2280** | 0.0164 ± 0.0036 | 48  | 131 |
| MOEA/D    | 125.2413 ± 2.1647   | 0.0198 ± 0.0043 | 92    | 27  |
| NSGA-II   | 123.1826 ± 1.5829   | **0.0092 ± 0.0020** | 98 | 40  |
| AGE-MOEA  | 119.5132 ± 1.2732   | 0.0136 ± 0.0023 | 90    | 171 |

## DTLZ2 (3-obj, dim=12, 30000 evals/run × 10 seeds)

Deb-Thiele-Laumanns-Zitzler 3-objective; the Pareto front is the
unit-sphere octant (`Σf² = 1, all f ≥ 0`) — a curved 2-D surface embedded
in 3-D objective space. `mean dist = |‖f‖ − 1|`, so 0 means perfectly on
the sphere (the known optimum).

| algorithm    | mean dist ↓         | spacing ↓       | front | ms  |
|---|---|---|---|---|
| **IBEA**     | **0.0014 ± 0.0002** | 0.0607 ± 0.0047 | 87    | 148 |
| MOEA/D       | 0.0037 ± 0.0003     | 0.0886 ± 0.0024 | 78    | 23  |
| HypE         | 0.0113 ± 0.0033     | **0.0269 ± 0.0172** | 80 | 41  |
| NSGA-III     | 0.0197 ± 0.0015     | 0.0735 ± 0.0052 | 92    | 91  |
| eps-MOEA     | 0.0325 ± 0.0104     | 0.0572 ± 0.0170 | 136   | 88  |
| NSGA-II      | 0.0332 ± 0.0068     | 0.0577 ± 0.0109 | 92    | 60  |
| SPEA2        | 0.0368 ± 0.0021     | 0.0288 ± 0.0038 | 92    | 530 |
| PESA-II      | 0.0395 ± 0.0033     | 0.0616 ± 0.0051 | 100   | 372 |
| SMS-EMOA     | 0.0484 ± 0.0134     | 0.0764 ± 0.0081 | 40    | 483 |
| RVEA         | 0.0510 ± 0.0044     | 0.0631 ± 0.0024 | 68    | 66  |
| MOPSO        | 0.0566 ± 0.0048     | 0.0687 ± 0.0084 | 100   | 66  |
| RandomSearch | 0.3949 ± 0.0152     | 0.0797 ± 0.0083 | 239   | 530 |

**IBEA wins decisively** (14× closer to the true front than NSGA-III).
SMS-EMOA's wall-clock fell ~2.7× from the v0.4.0 snapshot — the
`hypervolume_nd` rework.

## DTLZ1 (3-obj, dim=7, 30000 evals × 10 seeds)

Deb-Thiele-Laumanns-Zitzler 3-objective; the Pareto front is the linear
simplex `Σf = 0.5` in the positive octant. Hard because a deceptive
multimodal `g` term riddles the approach with a huge number of local
fronts — only fully-converged runs land on the simplex.

| algorithm | mean dist ↓         | spacing ↓       | front | ms  |
|---|---|---|---|---|
| **GrEA**  | **1.7725 ± 0.9897** | **0.0719 ± 0.0438** | 72 | 62  |
| MOEA/D    | 2.8022 ± 1.7807     | 0.2279 ± 0.2247 | 78    | 22  |
| AGE-MOEA  | 4.5395 ± 2.2114     | 0.3930 ± 0.2864 | 90    | 193 |
| NSGA-III  | 5.9130 ± 2.8212     | 0.4375 ± 0.2212 | 92    | 81  |

**GrEA shines on linear fronts** — the grid-based niching matches the
geometry better than reference points.

## Rastrigin (dim=5, 50000 evals/run × 10 seeds)

Highly multimodal trap: `f = 10n + Σ(xᵢ² − 10·cos(2π·xᵢ))`. Hard because a
near-quadratic global bowl is overlaid with ~10⁵ regularly spaced local
minima — any greedy step lands in the nearest dimple. Global optimum
`f = 0` at the origin.

| algorithm        | best f                | ms  |
|---|---|---|
| **(1+1)-ES**     | **0.0000e0 ± 0.00e0** | 4   |
| **DE**           | **0.0000e0 ± 0.00e0** | 6   |
| GA               | 7.0913e-8 ± 5.50e-8   | 15  |
| NSGA-II          | 4.9270e-5 ± 5.04e-5   | 60  |
| IPOP-CMA-ES      | 1.3423e-1 ± 2.71e-1   | 61  |
| PSO              | 7.9598e-1 ± 8.67e-1   | 5   |
| CMA-ES           | 2.3453e0 ± 1.49e0     | 10  |
| SimulatedAnneal  | 3.8540e0 ± 1.48e0     | 7   |
| RandomSearch     | 1.1064e1 ± 2.54e0     | 14  |
| HillClimber      | 1.5966e1 ± 6.25e0     | 6   |
| PAES             | 1.5966e1 ± 6.25e0     | 10  |

(1+1)-ES and DE tie for `f = 0`. **IPOP-CMA-ES drops vanilla CMA-ES from
2.35 → 0.13** — the restart logic does what it should.

## Rosenbrock (dim=5, 30000 evals × 10 seeds)

Rosenbrock's banana valley: `f = Σ(100·(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²)`. Hard
because the minimum sits in a long, bent, near-flat valley — easy to
enter, very slow to crawl along to the tip. Global optimum `f = 0` at the
all-ones point.

| algorithm     | best f                  | ms |
|---|---|---|
| **Nelder-Mead** | **0.0000e0 ± 0.00e0** | 1  |
| CMA-ES        | 3.6207e-29 ± 2.35e-29   | 5  |
| TLBO          | 1.8458e-3 ± 1.91e-3     | 1  |
| DE            | 3.3345e-1 ± 3.01e-1     | 2  |
| PSO           | 8.2124e-1 ± 1.58e0      | 2  |
| (1+1)-ES      | 2.2115e0 ± 2.70e0       | 1  |
| BO (60 evals) | 3.1725e3 ± 2.92e3       | 39 |

Nelder-Mead **= 0 exactly**, CMA-ES at machine epsilon. BO at only 60
evaluations is honestly bad on 5-D Rosenbrock (no kernel hyperparameter
tuning) — included as a reminder that BO needs more evaluations than a
smooth problem actually requires for these other methods.

## Ackley (dim=5, 30000 evals × 10 seeds)

Ackley's function: a near-flat outer plateau with shallow ripples
surrounding a single deep, narrow global basin. Hard because the gradient
is almost zero far from the optimum, giving local search little to
follow. Global optimum `f = 0` at the origin.

| algorithm     | best f                  | ms |
|---|---|---|
| **DE**        | **4.4409e-16 ± 0.00e0** | 3  |
| PSO           | 1.5099e-15 ± 1.63e-15   | 3  |
| CMA-ES        | 1.5099e-15 ± 1.63e-15   | 5  |
| TLBO          | 2.2204e-15 ± 1.78e-15   | 2  |
| BO (60 evals) | 1.9622e1 ± 1.23e0       | 38 |

All conventional methods reach machine precision. BO at 60 evals
struggles — same caveat as Rosenbrock.

---

## TSP ring-15 (8000 evals/run × 10 seeds)

15 equally-spaced cities on the unit circle; minimize the closed tour
length. The space is `(15−1)!/2` distinct tours, but cities in convex
position have no 2-opt local optima — so this instance cleanly separates
methods with good neighbourhood moves (inversion = 2-opt) from blind
recombination / sampling. Known optimum (the polygon perimeter):
**6.2374**.

| algorithm       | tour length ↓       | ms |
|---|---|---|
| **HillClimber** | **6.2374 ± 0.0000** | 0  |
| **SimulatedAnneal** | **6.2374 ± 0.0000** | 0 |
| **TabuSearch**  | **6.2374 ± 0.0000** | 0  |
| **AntColony**   | **6.2374 ± 0.0000** | 8  |
| GA              | 7.0133 ± 0.6725     | 2  |
| RandomSearch    | 12.1474 ± 0.6797    | 1  |

Every local-search method (and Ant Colony) hits the exact optimum — as
theory predicts for convex-position TSP under 2-opt. The GA's order
crossover drifts off the optimum, and random sampling is hopeless.

## JSS FT06 (8000 evals/run × 10 seeds)

Fisher & Thompson 1963 6-job × 6-machine job-shop; minimize makespan.
Hard because every job has a fixed machine order, so swapping two
operations can ripple delays across the whole schedule. Known optimum:
**55**.

| algorithm       | makespan ↓          | ms |
|---|---|---|
| **SimulatedAnneal** | **55.2000 ± 0.6000** | 1 |
| TabuSearch      | 55.9000 ± 1.4457    | 1  |
| GA              | 56.0000 ± 1.5492    | 4  |
| RandomSearch    | 58.5000 ± 1.2042    | 4  |
| HillClimber     | 62.5000 ± 4.3186    | 0  |

Simulated annealing gets within 0.4% of the known optimum on average;
greedy hill-climbing stalls in operation-order local optima.

## Knapsack (30 items, bi-objective, 20000 evals/run × 10 seeds)

Zitzler-Thiele style 0/1 knapsack: two profit vectors, one capacity (half
the total weight). Hard because the two profit objectives conflict and
the capacity constraint carves feasible regions out of the `2³⁰`
bitstrings. No closed-form optimum; scored by hypervolume vs reference
`[0, 0]` (higher is better).

| algorithm    | hypervolume ↑           | front | ms  |
|---|---|---|---|
| **NSGA-II**  | **1360468.3 ± 11619.6** | 100   | 39  |
| SPEA2        | 1355615.5 ± 9266.8      | 100   | 213 |
| IBEA         | 1352595.5 ± 10183.6     | 99    | 101 |
| NSGA-III     | 1346446.0 ± 6922.1      | 100   | 39  |
| RandomSearch | 1118233.1 ± 34150.3     | 9     | 17  |

The three Pareto EAs land within ~1% of each other; random search finds a
front of only ~9 points and trails badly.
