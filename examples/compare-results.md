# `compare` example — reference output

Snapshot from `cargo run --release --example compare` after the v0.3.0
algorithm cohort landed (2026-05-05). 10 seeds per algorithm per problem.

Wall-clock numbers are from the development machine and will vary;
the *relative* numbers across algorithms are the interesting part.

---

## ZDT1 (dim=30, 25000 evals/run × 10 seeds)

Two-objective benchmark with a smooth Pareto front along
`f₂ = 1 − √f₁`. Hypervolume reference point: `[11, 11]`.

| algorithm    | hypervolume ↑     | spacing ↓     | mean L2 ↓     | front | ms  |
|---|---|---|---|---|---|
| RandomSearch | 99.5691 ± 0.94    | 0.0937 ± 0.03 | 2.3621 ± 0.14 | 28    | 111 |
| PAES         | 104.1887 ± 0.90   | 0.0351 ± 0.01 | 1.3195 ± 0.06 | 33    | 16  |
| MOPSO        | **120.6149 ± 0.05** | 0.0125 ± 0.00 | **0.0005 ± 0.00** | 100 | 106 |
| SPEA2        | 118.0823 ± 0.60   | 0.0111 ± 0.00 | 0.2408 ± 0.05 | 97    | 461 |
| PESA-II      | 119.3670 ± 0.33   | **0.0095 ± 0.00** | 0.0802 ± 0.04 | 100 | 86 |
| ε-MOEA       | 118.8742 ± 0.68   | 0.0167 ± 0.01 | 0.0493 ± 0.02 | 45    | 55  |
| IBEA         | 120.0167 ± 0.31   | 0.0130 ± 0.00 | 0.0448 ± 0.02 | 73    | 141 |
| HypE         | 105.6489 ± 0.98   | 0.0266 ± 0.01 | 1.4820 ± 0.10 | 72    | 66  |
| SMS-EMOA     | 102.8871 ± 1.05   | 0.0263 ± 0.00 | 1.4937 ± 0.12 | 40    | 181 |
| RVEA         | 111.7151 ± 1.82   | 0.0308 ± 0.01 | 0.8399 ± 0.16 | 47    | 60  |
| NSGA-II      | 118.3336 ± 0.78   | 0.0112 ± 0.00 | 0.1891 ± 0.06 | 96    | 287 |
| NSGA-III     | 115.1612 ± 0.47   | 0.0139 ± 0.00 | 0.4314 ± 0.06 | 86    | 244 |
| MOEA/D       | 119.9450 ± 0.50   | 0.0118 ± 0.00 | 0.0065 ± 0.00 | 96    | 30  |

**MOPSO and MOEA/D dominate** convergence (mean L2 to true front ≤ 0.01).
PESA-II edges spacing.

## ZDT3 (dim=30, 25000 evals × 10 seeds)

Disconnected Pareto front; tests an algorithm's ability to maintain
spread across gaps.

| algorithm | hypervolume ↑   | spacing ↓     | front | ms   |
|---|---|---|---|---|
| NSGA-II   | 123.1826 ± 1.58 | 0.0092 ± 0.00 | 98    | 284  |
| MOEA/D    | 125.2413 ± 2.16 | 0.0198 ± 0.00 | 92    | 31   |
| **IBEA**  | **126.2072 ± 1.23** | 0.0164 ± 0.00 | 48 | 140 |
| AGE-MOEA  | 119.5132 ± 1.27 | 0.0136 ± 0.00 | 90    | 1028 |

## DTLZ2 (3-obj, dim=12, 30000 evals × 10 seeds)

Spherical Pareto front. Mean dist = `|‖f‖ − 1|`.

| algorithm   | mean dist ↓       | spacing ↓     | front | ms   |
|---|---|---|---|---|
| RandomSearch | 0.3949 ± 0.02    | 0.0797 ± 0.01 | 239   | 621  |
| MOPSO       | 0.0566 ± 0.00     | 0.0687 ± 0.01 | 100   | 48   |
| NSGA-II     | 0.0332 ± 0.01     | 0.0577 ± 0.01 | 92    | 380  |
| SPEA2       | 0.0368 ± 0.00     | **0.0288 ± 0.00** | 92 | 4723 |
| PESA-II     | 0.0395 ± 0.00     | 0.0616 ± 0.01 | 100   | 530  |
| ε-MOEA      | 0.0325 ± 0.01     | 0.0572 ± 0.02 | 136   | 104  |
| **IBEA**    | **0.0014 ± 0.00** | 0.0607 ± 0.00 | 87    | 158  |
| HypE        | 0.0113 ± 0.00     | 0.0269 ± 0.02 | 80    | 82   |
| SMS-EMOA    | 0.0484 ± 0.01     | 0.0764 ± 0.01 | 40    | 6002 |
| RVEA        | 0.0510 ± 0.00     | 0.0631 ± 0.00 | 68    | 68   |
| NSGA-III    | 0.0197 ± 0.00     | 0.0735 ± 0.01 | 92    | 325  |
| MOEA/D      | 0.0037 ± 0.00     | 0.0886 ± 0.00 | 78    | 25   |

**IBEA wins decisively** (15× closer to the true front than NSGA-III).

## DTLZ1 (3-obj, dim=7, 30000 evals × 10 seeds)

Linear simplex Pareto front (`Σf = 0.5`).

| algorithm | mean dist ↓     | spacing ↓     | front | ms   |
|---|---|---|---|---|
| NSGA-III  | 5.9130 ± 2.82   | 0.4375 ± 0.22 | 92    | 310  |
| MOEA/D    | 2.8022 ± 1.78   | 0.2279 ± 0.22 | 78    | 21   |
| AGE-MOEA  | 4.5395 ± 2.21   | 0.3930 ± 0.29 | 90    | 2327 |
| **GrEA**  | **1.7725 ± 0.99** | **0.0719 ± 0.04** | 72 | 286 |

**GrEA shines on linear fronts** — the grid-based niching matches the
geometry better than reference points.

## Rastrigin (dim=5, 50000 evals/run × 10 seeds)

Multimodal trap. Global minimum f = 0 at the origin.

| algorithm        | best f          | ms  |
|---|---|---|
| RandomSearch     | 1.1064e1 ± 2.54 | 16  |
| HillClimber      | 1.5966e1 ± 6.25 | 7   |
| **(1+1)-ES**     | **0.0000e0 ± 0.00** | 4 |
| SimulatedAnneal  | 3.8540e0 ± 1.48 | 7   |
| PAES             | 1.5966e1 ± 6.25 | 10  |
| GA               | 7.0913e-8 ± 5.50e-8 | 16 |
| PSO              | 7.9598e-1 ± 8.67e-1 | 5 |
| NSGA-II          | 4.9270e-5 ± 5.04e-5 | 269 |
| **DE**           | **0.0000e0 ± 0.00** | 6 |
| CMA-ES           | 2.3453e0 ± 1.49 | 12  |
| **IPOP-CMA-ES**  | 1.3423e-1 ± 2.71e-1 | 69 |

(1+1)-ES and DE tie for f = 0. **IPOP-CMA-ES drops vanilla CMA-ES from
2.35 → 0.13** — the restart logic does what it should.

## Rosenbrock (dim=5, 30000 evals × 10 seeds)

Smooth non-convex valley.

| algorithm     | best f                | ms  |
|---|---|---|
| DE            | 3.3345e-1 ± 3.01e-1   | 3   |
| PSO           | 8.2124e-1 ± 1.58e0    | 2   |
| **CMA-ES**    | **3.6207e-29 ± 2.35e-29** | 6 |
| TLBO          | 1.8458e-3 ± 1.91e-3   | 1   |
| (1+1)-ES      | 2.2115e0 ± 2.70e0     | 1   |
| **Nelder-Mead** | **0.0000e0 ± 0.00** | 1 |
| BO (60 evals) | 3.1725e3 ± 2.92e3     | 40  |

Nelder-Mead **= 0 exactly**, CMA-ES at machine epsilon. BO at only 60
evaluations is honestly bad on 5-D Rosenbrock (no kernel
hyperparameter tuning) — included as a reminder that BO needs more
evaluations than a smooth problem actually requires for these other
methods.

## Ackley (dim=5, 30000 evals × 10 seeds)

Smoother multimodal landscape than Rastrigin.

| algorithm     | best f                  | ms  |
|---|---|---|
| DE            | 4.4409e-16 ± 0.00       | 3   |
| PSO           | 1.5099e-15 ± 1.63e-15   | 3   |
| CMA-ES        | 1.5099e-15 ± 1.63e-15   | 7   |
| TLBO          | 2.2204e-15 ± 1.78e-15   | 2   |
| BO (60 evals) | 1.9622e1 ± 1.23         | 40  |

All conventional methods reach machine precision. BO at 60 evals
struggles — same caveat as Rosenbrock.
