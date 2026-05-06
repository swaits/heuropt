# Tune a model with expensive evaluations

Population-based EAs throw thousands of evaluations at a problem. If
each evaluation costs a minute (a model training run, a CFD solve, a
real-world measurement) you can't afford that. heuropt has three
algorithms aimed at this regime.

| Algorithm | Surrogate | Best for |
|---|---|---|
| [Bayesian Optimization][BayesianOpt] | Gaussian process + Expected Improvement | The textbook choice; needs kernel tuning to shine |
| [TPE] | Kernel-density estimate of good vs bad points | Cheaper per step; more robust without tuning |
| [Hyperband] | (none — it's a multi-fidelity scheduler) | When each eval has a tunable budget (epochs, MC samples) |

## When each is right

- **Black-box, fixed cost per eval, smooth-ish landscape** → BO.
- **Black-box, fixed cost per eval, no time to tune the surrogate** → TPE.
- **Each eval has a tunable fidelity** → Hyperband.

## Bayesian Optimization

A worked example with a synthetic 5-D problem and a 60-evaluation
budget — same configuration the `compare` harness uses.

```rust,no_run
use heuropt::prelude::*;

struct Rosenbrock5D;
impl Problem for Rosenbrock5D {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("f")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let f: f64 = x.windows(2).map(|w|
            100.0 * (w[1] - w[0].powi(2)).powi(2) + (1.0 - w[0]).powi(2)
        ).sum();
        Evaluation::new(vec![f])
    }
}

let bounds = vec![(-2.048_f64, 2.048_f64); 5];
let mut opt = BayesianOpt::new(
    BayesianOptConfig {
        evaluations: 60,
        initial_samples: 10,
        length_scale: 1.0,
        signal_variance: 1.0,
        noise_variance: 1e-6,
        seed: 42,
    },
    RealBounds::new(bounds),
);
let r = opt.run(&Rosenbrock5D);
println!("best f after 60 evals: {}", r.best.unwrap().evaluation.objectives[0]);
```

> **Honest disclosure.** On the comparison harness this default
> configuration produces **f ≈ 3170 ± 2920** on Rosenbrock 5-D — well
> below what a tuned BO can do. The default RBF kernel without
> per-problem hyperparameter tuning is the limitation. For real
> workloads, consider:
>
> - More evaluations (200+ instead of 60).
> - Tuning `length_scale` to a known scale of your problem
>   (lower for high-frequency landscapes, higher for smooth ones).
> - TPE instead of BO if you don't want to tune the kernel.

## Tree-structured Parzen Estimator

TPE keeps two density estimates — `l(x)` over historical good points
and `g(x)` over the rest — and picks new candidates that maximize the
ratio. Cheaper per step than a GP and famously robust without
hand-tuning.

```rust,no_run
use heuropt::prelude::*;
# struct Rosenbrock5D;
# impl Problem for Rosenbrock5D {
#     type Decision = Vec<f64>;
#     fn objectives(&self) -> ObjectiveSpace { ObjectiveSpace::new(vec![Objective::minimize("f")]) }
#     fn evaluate(&self, _x: &Vec<f64>) -> Evaluation { Evaluation::new(vec![0.0]) }
# }
let bounds = vec![(-2.048_f64, 2.048_f64); 5];
let mut opt = Tpe::new(
    TpeConfig {
        evaluations: 60,
        initial_samples: 10,
        gamma: 0.25,
        candidates_per_step: 24,
        bandwidth_factor: 1.06,
        seed: 42,
    },
    RealBounds::new(bounds),
);
let _r = opt.run(&Rosenbrock5D);
```

`gamma` is the fraction of best points used as `l(x)`; `0.25` is the
canonical Bergstra value.

## Hyperband

[Hyperband] needs your problem to implement [`PartialProblem`] —
that is, you can evaluate at a tunable fidelity (e.g. number of
training epochs). The algorithm schedules many cheap-fidelity runs
and promotes only the survivors to higher fidelity.

```rust,no_run
use heuropt::prelude::*;
use heuropt::core::partial_problem::PartialProblem;

struct ModelTuning;
impl Problem for ModelTuning {
    type Decision = Vec<f64>;
    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![Objective::minimize("val_loss")])
    }
    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        // Full-fidelity eval = train at max_epochs.
        self.evaluate_at_budget(x, 100.0)
    }
}
impl PartialProblem for ModelTuning {
    fn evaluate_at_budget(&self, x: &Vec<f64>, budget: f64) -> Evaluation {
        // Replace with: train your model for `budget` epochs, return val_loss.
        // For demo, pretend more budget = lower noisy loss.
        let lr = x[0];
        let wd = x[1];
        let loss = (lr - 0.001).powi(2) + (wd - 1e-4).powi(2)
                 + 1.0 / (budget + 1.0);
        Evaluation::new(vec![loss])
    }
}

let bounds = vec![(1e-5_f64, 1e-1), (1e-6_f64, 1e-2)];
let mut hyperband = Hyperband::new(
    HyperbandConfig {
        max_budget: 100.0,
        eta: 3.0,
        seed: 42,
    },
    RealBounds::new(bounds),
);
let _r = hyperband.run(&ModelTuning);
```

`max_budget` is the most epochs (or whatever your fidelity unit is)
you'd ever spend on a single config. `eta` controls how aggressive
the elimination is — `3.0` is the classic value; higher means more
aggressive culling.

## Strategy: combining surrogate + multi-fidelity

The state of the art (BOHB) combines BO with Hyperband: TPE picks the
configurations Hyperband then evaluates at increasing fidelity.
heuropt doesn't ship a unified BOHB but the building blocks are
there — wrap your `PartialProblem` with a TPE-driven sampler and
feed the picks into Hyperband. PRs welcome.

[BayesianOpt]: https://docs.rs/heuropt/latest/heuropt/algorithms/bayesian_opt/struct.BayesianOpt.html
[TPE]: https://docs.rs/heuropt/latest/heuropt/algorithms/tpe/struct.Tpe.html
[Hyperband]: https://docs.rs/heuropt/latest/heuropt/algorithms/hyperband/struct.Hyperband.html
[`PartialProblem`]: https://docs.rs/heuropt/latest/heuropt/core/partial_problem/trait.PartialProblem.html
