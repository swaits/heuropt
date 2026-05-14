# Multi-objective combinatorial problems

Real combinatorial problems usually have more than one cost. A TSP
where every edge has both *distance* and *time*; a job-shop where you
care about *makespan*, *flow time*, *and* *tardiness*; a knapsack
with two profit metrics and a single weight budget. The decision
type is still combinatorial — a permutation, a bitstring — but the
objective is a vector, and the answer is a Pareto front rather than
a single best.

heuropt's NSGA-II and NSGA-III are fully generic over the decision
type. You don't need a separate "combinatorial NSGA" — just plug in
the right initializer and variation operators for your encoding.

This recipe walks through three patterns:

- **Bi-objective TSP** with NSGA-II (Pareto front of two distance
  matrices over the same cities)
- **Bi-objective 0/1 knapsack** with NSGA-II (binary encoding)
- **3-objective JSS** with NSGA-III (the many-objective successor)

For the single-objective permutation toolkit it builds on, see
[Optimize a permutation](./permutation.md).

## Bi-objective TSP

This is the canonical multi-objective combinatorial benchmark
(Lust–Teghem 2010). Two TSP instances on the **same** city set define
two distance matrices A and B; the search trades off length under A
versus length under B.

```rust,no_run
use heuropt::prelude::*;
use heuropt::metrics::hypervolume_2d;

struct BiObjectiveTsp {
    dist_a: Vec<Vec<f64>>,
    dist_b: Vec<Vec<f64>>,
}

impl BiObjectiveTsp {
    fn tour_length(d: &[Vec<f64>], tour: &[usize]) -> f64 {
        let n = tour.len();
        let mut total = 0.0;
        for i in 0..n {
            total += d[tour[i]][tour[(i + 1) % n]];
        }
        total
    }
}

impl Problem for BiObjectiveTsp {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("length_A"),
            Objective::minimize("length_B"),
        ])
    }

    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        Evaluation::new(vec![
            Self::tour_length(&self.dist_a, tour),
            Self::tour_length(&self.dist_b, tour),
        ])
    }
}

fn main() {
    let n: usize = 25;
    let dist_a = vec![vec![0.0_f64; n]; n]; // your matrix A
    let dist_b = vec![vec![0.0_f64; n]; n]; // your matrix B
    let problem = BiObjectiveTsp { dist_a, dist_b };

    let mut optimizer = Nsga2::new(
        Nsga2Config {
            population_size: 200,
            generations: 600,
            seed: 11,
        },
        ShuffledPermutation { n },
        CompositeVariation {
            crossover: EdgeRecombinationCrossover,
            mutation: InversionMutation,
        },
    );
    let result = optimizer.run(&problem);

    println!("Pareto-front size: {}", result.pareto_front.len());

    // Hypervolume against a generous reference point (larger than any
    // length you'd reasonably see). Use this as the single-number
    // quality metric for the run.
    let ref_point = [40_000.0, 40_000.0];
    let hv = hypervolume_2d(&result.pareto_front, &problem.objectives(), ref_point);
    println!("Hypervolume vs. {:?}: {:.0}", ref_point, hv);
}
```

[`EdgeRecombinationCrossover`] (ERX) is the standout crossover for
TSP. On a 25-city bi-objective instance it produces about twice the
front diversity of OX, PMX, or CX — see
`examples/tsp_operators_compare.rs` for a head-to-head benchmark.

## Bi-objective 0/1 knapsack — `Vec<bool>` decisions

NSGA-II works over `Vec<bool>` the same way. The Zitzler–Thiele
bi-objective knapsack is the textbook benchmark: each item has two
profit values and a single weight; you maximize both profits under
one capacity constraint.

```rust,no_run
use heuropt::prelude::*;
use rand::Rng as _;

const N_ITEMS: usize = 30;

struct BiKnapsack {
    profits_a: Vec<f64>,
    profits_b: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
}

impl Problem for BiKnapsack {
    type Decision = Vec<bool>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::maximize("profit_A"),
            Objective::maximize("profit_B"),
        ])
    }

    fn evaluate(&self, take: &Vec<bool>) -> Evaluation {
        let (pa, pb, w) = take.iter().enumerate().fold(
            (0.0_f64, 0.0_f64, 0.0_f64),
            |(pa, pb, w), (i, &t)| {
                if t {
                    (pa + self.profits_a[i], pb + self.profits_b[i], w + self.weights[i])
                } else {
                    (pa, pb, w)
                }
            },
        );
        // Standard heuristic-MO constraint handling: penalize weight
        // overruns heavily so the recovered front is feasible.
        let penalty = 1000.0 * (w - self.capacity).max(0.0);
        Evaluation::new(vec![pa - penalty, pb - penalty])
    }
}

/// Each bit 50/50 independently.
#[derive(Clone, Copy)]
struct RandomBinary { n: usize }
impl Initializer<Vec<bool>> for RandomBinary {
    fn initialize(&mut self, size: usize, rng: &mut Rng) -> Vec<Vec<bool>> {
        (0..size).map(|_| (0..self.n).map(|_| rng.random_bool(0.5)).collect()).collect()
    }
}

/// One-point crossover for binary chromosomes.
#[derive(Default)]
struct OnePointCrossoverBool;
impl Variation<Vec<bool>> for OnePointCrossoverBool {
    fn vary(&mut self, parents: &[Vec<bool>], rng: &mut Rng) -> Vec<Vec<bool>> {
        let (p1, p2) = (&parents[0], &parents[1]);
        let n = p1.len();
        let cut = rng.random_range(1..n);
        let mut c1 = Vec::with_capacity(n);
        let mut c2 = Vec::with_capacity(n);
        c1.extend_from_slice(&p1[..cut]); c1.extend_from_slice(&p2[cut..]);
        c2.extend_from_slice(&p2[..cut]); c2.extend_from_slice(&p1[cut..]);
        vec![c1, c2]
    }
}

fn main() {
#   let profits_a = vec![0.0; N_ITEMS];
#   let profits_b = vec![0.0; N_ITEMS];
#   let weights   = vec![0.0; N_ITEMS];
    let problem = BiKnapsack {
        profits_a, profits_b, weights,
        capacity: 750.0, // ~half the total weight
    };

    let mut optimizer = Nsga2::new(
        Nsga2Config {
            population_size: 120,
            generations: 400,
            seed: 19,
        },
        RandomBinary { n: N_ITEMS },
        CompositeVariation {
            crossover: OnePointCrossoverBool,
            mutation: BitFlipMutation { probability: 1.0 / N_ITEMS as f64 },
        },
    );
    let result = optimizer.run(&problem);
    println!("Pareto-front size: {}", result.pareto_front.len());
}
```

Two things worth noting:

- **`OnePointCrossoverBool` and `RandomBinary` are defined locally.**
  They're tiny and common — a future PR could lift them into the
  library, but for now you write them inline.
- **Constraint handling is a penalty.** The factor `1000.0` is chosen
  so that even a 1-unit overrun beats any feasible solution by more
  than the entire profit range; the recovered front is entirely
  feasible. This is the standard heuristic-MO pattern (Deb 2001) and
  cheaper than a hard repair operator.

## Three-objective JSS with NSGA-III

NSGA-III is designed for ≥ 3 objectives. NSGA-II's crowding distance
degrades when most of the population is mutually non-dominated, which
is the rule rather than the exception in higher dimensions; NSGA-III
uses reference-point niching instead.

The example below adds *tardiness* to the standard (makespan, flow
time) JSS pair. Tardiness needs due dates; the common heuristic is
`dⱼ = 1.3 × sum_of_processing_times(j)`.

```rust,no_run
use heuropt::prelude::*;
use rand::Rng as _;

const N_JOBS: usize = 10;
const N_MACHINES: usize = 5;

struct La01ThreeObjective {
    routing: [[usize; N_MACHINES]; N_JOBS],
    times:   [[f64;   N_MACHINES]; N_JOBS],
    due:     [f64; N_JOBS],
}

impl Problem for La01ThreeObjective {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("makespan"),
            Objective::minimize("total_flow_time"),
            Objective::minimize("total_tardiness"),
        ])
    }

    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        let mut job_next = [0_usize; N_JOBS];
        let mut job_clock = [0.0_f64; N_JOBS];
        let mut machine_clock = [0.0_f64; N_MACHINES];
        for &job in schedule {
            let k = job_next[job];
            let m = self.routing[job][k];
            let t = self.times[job][k];
            let start = job_clock[job].max(machine_clock[m]);
            let end = start + t;
            job_clock[job] = end;
            machine_clock[m] = end;
            job_next[job] = k + 1;
        }
        let makespan = machine_clock.iter().cloned().fold(0.0_f64, f64::max);
        let flow_time: f64 = job_clock.iter().sum();
        let tardiness: f64 = job_clock.iter().zip(self.due.iter())
            .map(|(&c, &d)| (c - d).max(0.0))
            .sum();
        Evaluation::new(vec![makespan, flow_time, tardiness])
    }
}

/// Mix Insertion and Scramble per call — both preserve the multiset,
/// giving the search access to two complementary neighborhood moves.
#[derive(Default)]
struct InsertionOrScramble;
impl Variation<Vec<usize>> for InsertionOrScramble {
    fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
        if rng.random_bool(0.5) {
            InsertionMutation.vary(parents, rng)
        } else {
            ScrambleMutation.vary(parents, rng)
        }
    }
}

fn main() {
#   let routing = [[0; N_MACHINES]; N_JOBS];
#   let times   = [[0.0; N_MACHINES]; N_JOBS];
    let due = std::array::from_fn::<f64, N_JOBS, _>(
        |j| 1.3 * times[j].iter().sum::<f64>(),
    );
    let problem = La01ThreeObjective { routing, times, due };

    let mut optimizer = Nsga3::new(
        Nsga3Config {
            population_size: 120,
            generations: 600,
            reference_divisions: 12, // 91 Das-Dennis points in 3-D
            seed: 9,
        },
        ShuffledMultisetPermutation::new(vec![N_MACHINES; N_JOBS]),
        // Drop in a local PrecedenceOrderCrossover (POX) here for the
        // crossover slot if you want stronger mixing; see the
        // permutation recipe for the implementation.
        InsertionOrScramble,
    );
    let result = optimizer.run(&problem);

    println!("Pareto-front size: {}", result.pareto_front.len());
}
```

A few NSGA-III tips:

- **`reference_divisions` controls how many reference points the
  algorithm spreads across the front.** For M objectives, the Das–Dennis
  construction produces `C(divisions + M - 1, M - 1)` reference points.
  For M = 3 and divisions = 12 that's 91 points; pick a population size
  ≥ that.
- **`PrecedenceOrderCrossover` (POX)** belongs in the crossover slot
  for JSS. The strict-permutation crossovers (OX, PMX, CX, ERX) break
  the operation-string multiset. See
  [Optimize a permutation](./permutation.md#job-shop-scheduling-multiset-encodings)
  for the local POX definition.

## Comparing operators by hypervolume

For Pareto-front problems, single-objective fitness is the wrong
comparison metric. Use **hypervolume** instead — the dominated area
under the front, against a fixed reference point.

```rust,ignore
use heuropt::metrics::hypervolume_2d;

let ref_point = [40_000.0, 40_000.0]; // worse than anything you expect

for (name, crossover) in &[
    ("OX",  Box::new(OrderCrossover)             as Box<dyn Variation<Vec<usize>>>),
    ("PMX", Box::new(PartiallyMappedCrossover)   as _),
    ("CX",  Box::new(CycleCrossover)             as _),
    ("ERX", Box::new(EdgeRecombinationCrossover) as _),
] {
    let result = run_nsga2_with_crossover(crossover);
    let hv = hypervolume_2d(&result.pareto_front, &problem.objectives(), ref_point);
    println!("{name:>3}: hv = {hv:.0}");
}
```

This is the pattern in `examples/tsp_operators_compare.rs`. On the
KroAB-25 instance it ranks ERX > OX > PMX > CX by hypervolume.

For ≥ 3 objectives, hypervolume in N dimensions is exponentially
expensive; use [`hypervolume_2d`] when you can collapse to two
objectives for the metric, or sample-based hypervolume from [`HypE`]
otherwise.

## Pareto-front tips

| Problem | Algorithm | Notes |
|---|---|---|
| 2 objectives, permutation | [Nsga2][Nsga2] | Strong default |
| 2 objectives, binary | [Nsga2][Nsga2] | Same machinery, different encoding |
| 3 objectives | [Nsga3][Nsga3] | NSGA-II's crowding distance starts to degrade |
| 4+ objectives | [Nsga3][Nsga3] or [HypE][HypE] | NSGA-III if front is curved; HypE for indicator-based at scale |
| Many-objective with grid structure | [GrEA][Grea] | Wins linear / simplex fronts |

| Question | Use |
|---|---|
| Single-number quality metric for a run | `hypervolume_2d` against a fixed reference |
| "Is run A's front better than B's?" | Same reference point, compare hypervolume |
| "Pick one solution from the front" | See [Pick one answer off a Pareto front](./pick-one.md) |
| Interactive exploration / visualization | See [Explore your results in a webapp](./explorer.md) |

[Nsga2]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga2/struct.Nsga2.html
[Nsga3]: https://docs.rs/heuropt/latest/heuropt/algorithms/nsga3/struct.Nsga3.html
[HypE]: https://docs.rs/heuropt/latest/heuropt/algorithms/hype/struct.Hype.html
[Grea]: https://docs.rs/heuropt/latest/heuropt/algorithms/grea/struct.Grea.html
[`EdgeRecombinationCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.EdgeRecombinationCrossover.html
[`hypervolume_2d`]: https://docs.rs/heuropt/latest/heuropt/metrics/fn.hypervolume_2d.html
