# Optimize a permutation (TSP-style)

When your decision is "an ordering" — visiting cities, scheduling
jobs, routing — the natural representation is `Vec<usize>`. heuropt
ships three reasonable starting points:

- **A purpose-built algorithm** — [Ant Colony][AntColonyTsp] for TSP-shaped
  problems with a distance matrix.
- **A genetic algorithm** with the permutation operator toolkit —
  the most general option, and the right choice when you want to
  bring your own evaluator without a pheromone metaphor.
- **A trajectory method** — [Simulated Annealing][SimulatedAnnealing] +
  [`SwapMutation`] for a tiny baseline, or [Tabu Search][TabuSearch] when
  you have a custom neighbor function.

This recipe walks through all three, with the bulk of the page on
the GA toolkit, since it's the most flexible. For the multi-objective
versions (bi-objective TSP, bi-objective JSS, Pareto fronts) see
[Multi-objective combinatorial problems](./multi-objective-combinatorial.md).

## The permutation operator toolkit

heuropt ships a complete set of permutation operators in the prelude.
You compose them with [`CompositeVariation`] into a crossover-plus-mutation
pipeline and feed them to any GA-shaped algorithm.

### Initializers

| Operator | What it produces | Use for |
|---|---|---|
| [`ShuffledPermutation`] | Random shuffles of `[0..n)` | TSP, QAP, single-machine scheduling — strict permutations |
| [`ShuffledMultisetPermutation`] | Random shuffles of `[0]*r₀ ++ [1]*r₁ ++ …` | Job-shop scheduling operation strings (each job id repeated `n_machines` times) |

### Crossovers

All four take two parents and return two children. They assume *strict*
permutations — applying them to multiset encodings (like JSS) will
break the multiset.

| Operator | One-liner | Best at |
|---|---|---|
| [`OrderCrossover`] (OX) | Copy a random segment from A, fill the rest in B's order | General-purpose, fast |
| [`PartiallyMappedCrossover`] (PMX) | Slide A's segment into B via positional swaps | Classic; preserves more position info than OX |
| [`CycleCrossover`] (CX) | Partition positions into cycles, alternate parents | Preserves the most positional information |
| [`EdgeRecombinationCrossover`] (ERX) | Greedy walk through the union of both parents' edges | The gold standard for TSP — preserves adjacency, not position |

For TSP specifically, ERX usually wins on Pareto-front quality at the
cost of being ~70% slower per crossover. See
[Multi-objective combinatorial problems](./multi-objective-combinatorial.md)
for a head-to-head comparison.

### Mutations

All five take one parent and return one child. All four below preserve
both strict permutations *and* multiset encodings, so they're safe for
JSS too.

| Operator | What it does | Notes |
|---|---|---|
| [`SwapMutation`] | Swap two random positions | Smallest perturbation; canonical default |
| [`InversionMutation`] | Reverse a random sub-slice | The textbook 2-opt-style move for TSP |
| [`InsertionMutation`] | Remove an element, re-insert elsewhere | Strong for sequencing / scheduling |
| [`ScrambleMutation`] | Randomly permute a random sub-slice | Stronger diversification |

### Quick "what should I use?" guide

| Your problem | Initializer | Crossover | Mutation |
|---|---|---|---|
| TSP / routing | `ShuffledPermutation` | `EdgeRecombinationCrossover` | `InversionMutation` |
| Single-machine scheduling | `ShuffledPermutation` | `OrderCrossover` | `InsertionMutation` |
| Generic strict permutation | `ShuffledPermutation` | `OrderCrossover` | `InversionMutation` |
| Job-shop scheduling (multiset) | `ShuffledMultisetPermutation` | *example-local POX* (see below) | `InversionMutation` or `SwapMutation` |

## Single-objective TSP with a Genetic Algorithm

This is the toolkit's headline pattern. It mirrors the
`examples/tsp_ulysses16.rs` benchmark, which converges to the known
TSPLIB optimum for the 16-city Ulysses instance.

```rust,no_run
use heuropt::prelude::*;

struct Tsp {
    distances: Vec<Vec<f64>>,
}

impl Problem for Tsp {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("tour_length")])
    }

    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        let n = tour.len();
        let mut len = 0.0;
        for i in 0..n {
            len += self.distances[tour[i]][tour[(i + 1) % n]];
        }
        Evaluation::new(vec![len])
    }
}

fn main() {
    // Replace with your actual distance matrix.
    let n: usize = 16;
    let distances = vec![vec![0.0_f64; n]; n];
    let problem = Tsp { distances };

    let mut optimizer = GeneticAlgorithm::new(
        GeneticAlgorithmConfig {
            population_size: 150,
            generations: 1500,
            tournament_size: 3,
            elitism: 4,
            seed: 42,
        },
        ShuffledPermutation { n },
        CompositeVariation {
            crossover: OrderCrossover,
            mutation: InversionMutation,
        },
    );
    let r = optimizer.run(&problem);
    let best = r.best.unwrap();
    println!("best tour length: {:.0}", best.evaluation.objectives[0]);
    println!("tour: {:?}", best.decision);
}
```

A few things to notice:

- **Decision type is `Vec<usize>`.** Every operator in the toolkit is
  generic over the decision type via `Variation<Vec<usize>>`, so the
  whole pipeline composes naturally.
- **`CompositeVariation` is the wiring.** It runs the crossover first,
  then runs the mutation on each child. For a single-parent operator
  pair (e.g., two mutations stacked), it still works — the "crossover"
  slot just becomes a first-stage mutation.
- **Tournament size 3 and elitism 4** are slightly stronger than the
  defaults; small permutation GAs benefit from a touch more selection
  pressure.

## Job-shop scheduling — multiset encodings

JSS problems use a different encoding: a string of length
`n_jobs × n_machines` where each job id appears `n_machines` times. The
k-th occurrence of job `j` represents the k-th operation of job `j`.
This is a *multiset permutation*, not a strict permutation, and the
crossovers above (OX, PMX, CX, ERX) will break it because they assume
each value appears exactly once.

Use `ShuffledMultisetPermutation` for the initializer:

```rust,ignore
use heuropt::prelude::*;

// 6 jobs × 6 machines (FT06 layout): each job id 0..6 appears 6 times.
let initializer = ShuffledMultisetPermutation::new(vec![6; 6]);
```

For variation you have two options:

1. **Mutation only.** The four mutations above all preserve the
   multiset, so you can drive a GA with just `InversionMutation` or
   `SwapMutation` and skip crossover. This works on small JSS
   instances; on larger ones search becomes slow.

2. **Add a JSS-aware crossover.** The standard choice is **POX**
   (Precedence-preserving Order-based Crossover, Bierwirth 1996).
   It's not in the library because every JSS instance specifies its
   own number of distinct ids and POX needs that constant; defining
   it locally per example keeps the type clean:

   ```rust,no_run
   use heuropt::prelude::*;
   use rand::Rng as _;

   const N_JOBS: usize = 6;

   /// POX — partition job ids into two sets J1/J2; child takes positions
   /// of J1-jobs from parent A and fills the remaining positions with
   /// J2-jobs from parent B in B's order. Preserves the multiset.
   #[derive(Default)]
   struct PrecedenceOrderCrossover;

   impl Variation<Vec<usize>> for PrecedenceOrderCrossover {
       fn vary(&mut self, parents: &[Vec<usize>], rng: &mut Rng) -> Vec<Vec<usize>> {
           let p1 = &parents[0];
           let p2 = &parents[1];
           let mut in_j1 = [false; N_JOBS];
           loop {
               for slot in &mut in_j1 {
                   *slot = rng.random_bool(0.5);
               }
               let count = in_j1.iter().filter(|&&b| b).count();
               if count > 0 && count < N_JOBS { break; }
           }
           vec![pox_child(p1, p2, &in_j1), pox_child(p2, p1, &in_j1)]
       }
   }

   fn pox_child(donor: &[usize], filler: &[usize], in_donor_set: &[bool]) -> Vec<usize> {
       let n = donor.len();
       let mut child = vec![usize::MAX; n];
       for k in 0..n {
           if in_donor_set[donor[k]] {
               child[k] = donor[k];
           }
       }
       let mut idx = 0;
       for &v in filler {
           if !in_donor_set[v] {
               while idx < n && child[idx] != usize::MAX { idx += 1; }
               child[idx] = v;
               idx += 1;
           }
       }
       child
   }
   ```

   See `examples/jss_ft06_bi.rs` and `examples/mo_jss_la01.rs` for the
   complete worked examples.

A full JSS evaluator walks the schedule string left-to-right, tracking
per-job operation counters and per-machine clocks:

```rust,ignore
fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
    let mut job_next = [0_usize; N_JOBS];
    let mut job_clock = [0.0_f64; N_JOBS];
    let mut machine_clock = [0.0_f64; N_MACHINES];
    for &job in schedule {
        let k = job_next[job];
        let m = ROUTING[job][k];
        let t = PROCESSING_TIME[job][k];
        let start = job_clock[job].max(machine_clock[m]);
        let end = start + t;
        job_clock[job] = end;
        machine_clock[m] = end;
        job_next[job] = k + 1;
    }
    let makespan = machine_clock.iter().cloned().fold(0.0_f64, f64::max);
    Evaluation::new(vec![makespan])
}
```

## Comparing crossover operators

Tuning the right operator combo matters more than tuning population
size. The pattern is: hold everything constant, swap the operator,
record the metric:

```rust,ignore
use heuropt::prelude::*;
use heuropt::metrics::hypervolume_2d;

fn run_with<C: Variation<Vec<usize>>>(crossover: C) -> f64 {
    let mut opt = GeneticAlgorithm::new(
        GeneticAlgorithmConfig { /* identical config */ ..Default::default() },
        ShuffledPermutation { n: 25 },
        CompositeVariation { crossover, mutation: InversionMutation },
    );
    opt.run(&problem).best.unwrap().evaluation.objectives[0]
}

println!("OX:  {:.0}", run_with(OrderCrossover));
println!("PMX: {:.0}", run_with(PartiallyMappedCrossover));
println!("CX:  {:.0}", run_with(CycleCrossover));
println!("ERX: {:.0}", run_with(EdgeRecombinationCrossover));
```

For Pareto-front problems use hypervolume, not single-objective
fitness, as the comparison metric — see
[`examples/tsp_operators_compare.rs`][CompareExample] for the bi-objective
version.

## TSP with Ant Colony

When your problem is genuinely TSP-shaped — symmetric distance matrix,
visit-every-node — Ant Colony is purpose-built and worth a look. It
doesn't use crossover or mutation; instead it deposits pheromone trails
that bias future ants toward good edges.

```rust,no_run
use heuropt::prelude::*;

struct Tsp {
    distances: Vec<Vec<f64>>,
}

impl Problem for Tsp {
    type Decision = Vec<usize>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("length")])
    }

    fn evaluate(&self, tour: &Vec<usize>) -> Evaluation {
        let mut len = 0.0;
        for w in tour.windows(2) {
            len += self.distances[w[0]][w[1]];
        }
        len += self.distances[*tour.last().unwrap()][tour[0]];
        Evaluation::new(vec![len])
    }
}

fn main() {
    let cities = vec![(0.0, 0.0), (1.0, 5.0), (5.0, 2.0), (6.0, 6.0), (8.0, 3.0)];
    let n = cities.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let dx = cities[i].0 - cities[j].0;
            let dy = cities[i].1 - cities[j].1;
            distances[i][j] = (dx * dx + dy * dy).sqrt();
        }
    }
    let problem = Tsp { distances: distances.clone() };

    let mut opt = AntColonyTsp::new(AntColonyTspConfig {
        ants: 20,
        iterations: 200,
        alpha: 1.0,
        beta: 5.0,
        evaporation: 0.5,
        deposit: 1.0,
        distances,
        seed: 42,
    });

    let r = opt.run(&problem);
    let best = r.best.unwrap();
    println!("best tour length: {:.3}", best.evaluation.objectives[0]);
}
```

`alpha` weights pheromone influence and `beta` weights the heuristic
(1 / distance). `evaporation` is the per-iteration pheromone decay.
The classic Dorigo paper uses `alpha = 1`, `beta = 2..5`,
`evaporation = 0.1..0.5`.

## Tiny baseline: SA + SwapMutation

The smallest possible permutation optimizer — one starting decision,
no population, one mutation operator. Good as a sanity-check baseline.

```rust,no_run
use heuropt::prelude::*;

struct JobShop {
    process_times: Vec<f64>,
}
impl Problem for JobShop {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("weighted_completion")])
    }
    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        let cost: f64 = schedule.iter().enumerate()
            .map(|(i, &job)| (i as f64 + 1.0) * self.process_times[job])
            .sum();
        Evaluation::new(vec![cost])
    }
}

let times = vec![3.0, 1.5, 4.2, 2.7, 5.1];
let n = times.len();
let problem = JobShop { process_times: times };

// SimulatedAnnealing expects exactly one initial decision.
struct OneShuffle { n: usize }
impl Initializer<Vec<usize>> for OneShuffle {
    fn initialize(&mut self, _size: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
        use rand::seq::SliceRandom;
        let mut p: Vec<usize> = (0..self.n).collect();
        p.shuffle(rng);
        vec![p]
    }
}

let mut opt = SimulatedAnnealing::new(
    SimulatedAnnealingConfig {
        iterations: 2000,
        initial_temperature: 5.0,
        final_temperature: 1e-3,
        seed: 7,
    },
    OneShuffle { n },
    SwapMutation,
);
let r = opt.run(&problem);
let best = r.best.unwrap();
println!("best cost: {:.3}", best.evaluation.objectives[0]);
```

## Custom neighborhoods: Tabu Search

When you want full control of the move set (e.g., systematic 2-opt for
TSP, or insert-and-shift for scheduling), [Tabu Search][TabuSearch] takes
your own neighbor function.

```rust,ignore
use heuropt::prelude::*;
let neighbors = |x: &Vec<usize>, _rng: &mut Rng| -> Vec<Vec<usize>> {
    // All 2-opt neighbors of x.
    let mut out = Vec::new();
    for i in 0..x.len() {
        for j in (i + 2)..x.len() {
            let mut child = x.clone();
            child[i + 1..=j].reverse();
            out.push(child);
        }
    }
    out
};
// Pass `neighbors` to TabuSearch::new(...).
```

## When to use which approach

| Situation | Use |
|---|---|
| TSP-shaped with a distance matrix | [Ant Colony][AntColonyTsp] |
| Generic permutation, multi-seed budget | GA + `ShuffledPermutation` + OX + Inversion |
| Job-shop scheduling | GA + `ShuffledMultisetPermutation` + local POX + Inversion |
| Single-decision baseline | [SimulatedAnnealing][SimulatedAnnealing] + `SwapMutation` |
| Hand-crafted neighborhood (e.g. systematic 2-opt) | [Tabu Search][TabuSearch] |
| Bi-objective / many-objective permutation problem | See [Multi-objective combinatorial](./multi-objective-combinatorial.md) |

[AntColonyTsp]: https://docs.rs/heuropt/latest/heuropt/algorithms/ant_colony_tsp/struct.AntColonyTsp.html
[SimulatedAnnealing]: https://docs.rs/heuropt/latest/heuropt/algorithms/simulated_annealing/struct.SimulatedAnnealing.html
[TabuSearch]: https://docs.rs/heuropt/latest/heuropt/algorithms/tabu_search/struct.TabuSearch.html
[`ShuffledPermutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ShuffledPermutation.html
[`ShuffledMultisetPermutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ShuffledMultisetPermutation.html
[`OrderCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.OrderCrossover.html
[`PartiallyMappedCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.PartiallyMappedCrossover.html
[`CycleCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.CycleCrossover.html
[`EdgeRecombinationCrossover`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.EdgeRecombinationCrossover.html
[`SwapMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.SwapMutation.html
[`InversionMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.InversionMutation.html
[`InsertionMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.InsertionMutation.html
[`ScrambleMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.ScrambleMutation.html
[`CompositeVariation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.CompositeVariation.html
[CompareExample]: https://github.com/swaits/heuropt/blob/main/examples/tsp_operators_compare.rs
