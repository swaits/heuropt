# Optimize a permutation (TSP-style)

When your decision is "an ordering" — visiting cities, scheduling
jobs, routing — the natural representation is `Vec<usize>` and the
specialized algorithm is [Ant Colony][AntColonyTsp]. Generic alternatives are
[Simulated Annealing][SimulatedAnnealing] + [`SwapMutation`] for any permutation, and
[Tabu Search][TabuSearch] when you have a custom neighbor function.

## TSP with Ant Colony

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
    // 5-city Euclidean instance
    let cities = vec![
        (0.0, 0.0),
        (1.0, 5.0),
        (5.0, 2.0),
        (6.0, 6.0),
        (8.0, 3.0),
    ];
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
    println!("tour: {:?}", best.decision);
}
```

`alpha` weights pheromone influence and `beta` weights the
heuristic (1 / distance). `evaporation` is the per-iteration decay
of pheromone trails. The classic Dorigo paper uses `alpha = 1`,
`beta = 2..5`, `evaporation = 0.1..0.5`.

## Generic permutation: SA + SwapMutation

Use this when your problem isn't TSP-shaped (no distance matrix
makes sense) but you still want to optimize an ordering.

```rust,no_run
use heuropt::prelude::*;

struct JobShop {
    process_times: Vec<f64>,
}
impl Problem for JobShop {
    type Decision = Vec<usize>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("makespan")])
    }
    fn evaluate(&self, schedule: &Vec<usize>) -> Evaluation {
        // Pretend cumulative weighted-completion-time. Replace with your real cost.
        let cost: f64 = schedule.iter().enumerate()
            .map(|(i, &job)| (i as f64 + 1.0) * self.process_times[job])
            .sum();
        Evaluation::new(vec![cost])
    }
}

fn make_initial_perm(n: usize, seed: u64) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut rng = rng_from_seed(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    perm
}

let times = vec![3.0, 1.5, 4.2, 2.7, 5.1];
let problem = JobShop { process_times: times.clone() };

// SimulatedAnnealing needs a starting decision; pass a custom Initializer.
struct OnePerm(Vec<usize>);
impl Initializer<Vec<usize>> for OnePerm {
    fn initialize(&mut self, _size: usize, _rng: &mut Rng) -> Vec<Vec<usize>> {
        vec![self.0.clone()]
    }
}

let mut opt = SimulatedAnnealing::new(
    SimulatedAnnealingConfig {
        iterations: 2000,
        initial_temperature: 5.0,
        final_temperature: 1e-3,
        seed: 7,
    },
    OnePerm(make_initial_perm(times.len(), 7)),
    SwapMutation,
);
let r = opt.run(&problem);
let best = r.best.unwrap();
println!("best makespan: {:.3}", best.evaluation.objectives[0]);
println!("schedule: {:?}", best.decision);
```

`SwapMutation` swaps two random indices in the permutation —
preserves the "every element appears once" invariant for free.

## Custom neighborhoods: Tabu Search

When swap isn't the right move set (e.g., 2-opt for TSP, insert /
shift for scheduling), use [Tabu Search][TabuSearch] with your own neighbor
function.

```rust,ignore
use heuropt::prelude::*;
let neighbors = |x: &Vec<usize>, _rng: &mut Rng| -> Vec<Vec<usize>> {
    // Generate all 2-opt neighbors of x.
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

[AntColonyTsp]: https://docs.rs/heuropt/latest/heuropt/algorithms/ant_colony_tsp/struct.AntColonyTsp.html
[SimulatedAnnealing]: https://docs.rs/heuropt/latest/heuropt/algorithms/simulated_annealing/struct.SimulatedAnnealing.html
[`SwapMutation`]: https://docs.rs/heuropt/latest/heuropt/operators/struct.SwapMutation.html
[TabuSearch]: https://docs.rs/heuropt/latest/heuropt/algorithms/tabu_search/struct.TabuSearch.html
