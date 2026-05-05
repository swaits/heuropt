//! Tune the four lifecycle constants of the `jiggly` USB-mouse-jiggler firmware
//! as a **multi-objective** optimization problem.
//!
//! The Python `tune_runtime.py` from ~/Code/jiggly grid-searches against a
//! single composite score that linearly combines several genuinely conflicting
//! goals. That's a workable workaround for grid search — you have to rank by
//! one number — but it bakes the user's weights into the search and hides the
//! tradeoffs.
//!
//! `heuropt` lets us optimize the goals as separate objectives and surface the
//! Pareto front of legitimate tradeoffs:
//!
//! 1. **minimize work-time failures** — the screen sleeping while the user is
//!    working is the worst outcome. (`mean_work_sleep`, minutes/day)
//! 2. **maximize lunch sleep** — the entire design goal. (`mean_lunch`,
//!    minutes/day, encoded as a Maximize objective)
//! 3. **minimize human interactions** — every button press is UX cost.
//!    (`mean_presses`, per day)
//! 4. **minimize after-hours waste** — keeping the screen alive past the end
//!    of the workday is screen burn for nothing. (`mean_after`, minutes/day)
//!
//! Decision: a 4-element `Vec<f64>` for `(RT, YELLOW_AT, RED_AT, FAST_RED_AT)`,
//! continuous-relaxed and rounded to integer minutes inside `evaluate`. The
//! firmware ordering constraint `YA > RA > FRA > 0` is encoded as
//! `constraint_violation` so the algorithm's feasible-beats-infeasible logic
//! handles it automatically.
//!
//! Solver: NSGA-III with 4 objectives and Das-Dennis H=6 → 84 reference
//! points, matching the population size. Each `evaluate` runs a 1,000-workday
//! Monte Carlo, so this example is also a deliberately meaty evaluator that
//! benefits from `--features parallel` (≈8× wall-clock with rayon enabled on
//! a typical laptop).
//!
//! ```bash
//! cargo run --release --example jiggly_tuning
//! cargo run --release --example jiggly_tuning --features parallel
//! ```
//!
//! Output is in jiggly's native units — `RT` as `Xh00m`, thresholds as plain
//! minutes, durations as `Xh00m` / `Mm`, probabilities as percentages.

use std::time::Instant;

use rand::Rng as _;
use rand::SeedableRng;
use rand::rngs::StdRng;

use heuropt::prelude::*;

const LUNCH_START: i32 = 12 * 60;
const LUNCH_END: i32 = 13 * 60;

const P_PRESS_YELLOW: f64 = 0.015;
const P_PRESS_RED: f64 = 0.040;
const P_PRESS_FAST_RED: f64 = 0.060;
const P_WARN10_BUMP: f64 = 0.04;
const P_WARN5_BUMP: f64 = 0.03;

// Sweet-spot lunch-sleep window (minutes spent dead during 12:00–13:00).
const SWEET_LO: u32 = 15;
const SWEET_HI: u32 = 45;

const N_DAYS: usize = 1000;

// -----------------------------------------------------------------------------
// Day model + Monte Carlo (same model as scripts/tune_runtime.py)
// -----------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct DayOutcome {
    presses: u32,
    slept_work: u32,
    slept_lunch: u32,
    after_hours: u32,
}

#[derive(Clone, Copy)]
struct Stats {
    /// Probability of landing in the 12:15–12:45 sweet spot.
    p_sweet: f64,
    mean_lunch: f64,
    mean_work_sleep: f64,
    mean_presses: f64,
    mean_after: f64,
}

fn sample_triangular(low: f64, mode: f64, high: f64, rng: &mut StdRng) -> f64 {
    let u: f64 = rng.random();
    let c = (mode - low) / (high - low);
    if u < c {
        low + ((high - low) * (mode - low) * u).sqrt()
    } else {
        high - ((high - low) * (high - mode) * (1.0 - u)).sqrt()
    }
}

/// Pre-sampled simulated workdays. Sampling once and reusing across all
/// `evaluate` calls is the standard SAA pattern: every parameter combination
/// is scored on the same days, so differences in objective values reflect the
/// parameters rather than Monte Carlo noise between evaluations.
struct JigglyTuning {
    days: Vec<(i32, i32, u64)>, // start_min, end_min, per-day RNG seed
}

impl JigglyTuning {
    fn new(n_days: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let days = (0..n_days)
            .map(|_| {
                let s = (sample_triangular(8.0, 8.5, 9.5, &mut rng) * 60.0) as i32;
                let e = (sample_triangular(16.0, 17.5, 19.0, &mut rng) * 60.0) as i32;
                let day_seed: u64 = rng.random();
                (s, e, day_seed)
            })
            .collect();
        Self { days }
    }

    fn simulate_one(
        s: i32,
        e: i32,
        day_seed: u64,
        rt: i32,
        ya: i32,
        ra: i32,
        fra: i32,
    ) -> DayOutcome {
        let mut rng = StdRng::seed_from_u64(day_seed);
        let mut expire = s + rt;
        let mut o = DayOutcome::default();
        let t_max = e.max(expire) + 1;
        for t in s..t_max {
            // Free re-tap when the user re-logs in at 13:00.
            if t == LUNCH_END && t < e {
                expire = t + rt;
            }
            let in_workday = t >= s && t < e;
            let at_lunch = (LUNCH_START..LUNCH_END).contains(&t);
            let device_running = t < expire;
            let device_dead = !device_running;

            if device_dead && in_workday {
                if at_lunch {
                    o.slept_lunch += 1;
                } else {
                    o.slept_work += 1;
                }
            }
            if t >= e && device_running {
                o.after_hours += 1;
            }

            if !at_lunch && in_workday && device_running {
                let remaining = expire - t;
                let mut p = 0.0;
                if remaining > ra && remaining <= ya {
                    p = P_PRESS_YELLOW;
                } else if remaining > fra && remaining <= ra {
                    p = P_PRESS_RED;
                } else if remaining > 0 && remaining <= fra {
                    p = P_PRESS_FAST_RED;
                }
                if remaining == 10 {
                    p += P_WARN10_BUMP;
                }
                if remaining == 5 {
                    p += P_WARN5_BUMP;
                }
                let roll: f64 = rng.random();
                if roll < p {
                    expire = t + rt;
                    o.presses += 1;
                }
            }
        }
        o
    }

    fn aggregate(&self, rt: i32, ya: i32, ra: i32, fra: i32) -> Stats {
        let n = self.days.len() as f64;
        let mut sweet = 0u32;
        let mut sum_lunch = 0.0_f64;
        let mut sum_work = 0.0_f64;
        let mut sum_presses = 0.0_f64;
        let mut sum_after = 0.0_f64;
        for &(s, e, ds) in &self.days {
            let o = Self::simulate_one(s, e, ds, rt, ya, ra, fra);
            if (SWEET_LO..=SWEET_HI).contains(&o.slept_lunch) {
                sweet += 1;
            }
            sum_lunch += o.slept_lunch as f64;
            sum_work += o.slept_work as f64;
            sum_presses += o.presses as f64;
            sum_after += o.after_hours as f64;
        }
        Stats {
            p_sweet: sweet as f64 / n,
            mean_lunch: sum_lunch / n,
            mean_work_sleep: sum_work / n,
            mean_presses: sum_presses / n,
            mean_after: sum_after / n,
        }
    }
}

impl Problem for JigglyTuning {
    type Decision = Vec<f64>;

    fn objectives(&self) -> ObjectiveSpace {
        ObjectiveSpace::new(vec![
            Objective::minimize("work_failure_min"),
            Objective::maximize("lunch_sleep_min"),
            Objective::minimize("presses_per_day"),
            Objective::minimize("after_hours_min"),
        ])
    }

    fn evaluate(&self, x: &Vec<f64>) -> Evaluation {
        let rt = x[0].round() as i32;
        let ya = x[1].round() as i32;
        let ra = x[2].round() as i32;
        let fra = x[3].round() as i32;

        // Soft constraint: YA > RA > FRA > 0 (any violation is positive).
        let mut violation = 0.0_f64;
        if ra >= ya {
            violation += (ra - ya + 1) as f64;
        }
        if fra >= ra {
            violation += (fra - ra + 1) as f64;
        }
        if fra <= 0 {
            violation += (1 - fra) as f64;
        }

        let stats = self.aggregate(rt, ya, ra, fra);
        Evaluation::constrained(
            vec![
                stats.mean_work_sleep,
                stats.mean_lunch, // Objective is Maximize → as_minimization will negate
                stats.mean_presses,
                stats.mean_after,
            ],
            violation.max(0.0),
        )
    }
}

// -----------------------------------------------------------------------------
// Output formatting (matches the units used in tune_runtime.py)
// -----------------------------------------------------------------------------

fn fmt_minutes(m: f64) -> String {
    let total = m.round() as i32;
    let h = total / 60;
    let mm = total % 60;
    if h > 0 {
        format!("{h}h{mm:02}m")
    } else {
        format!("{mm}m")
    }
}

fn fmt_rt(m: i32) -> String {
    let h = m / 60;
    let mm = m % 60;
    format!("{h}h{mm:02}m")
}

/// One row in the Pareto-front summary table.
struct Row {
    rt: i32,
    ya: i32,
    ra: i32,
    fra: i32,
    work_fail: f64,
    lunch: f64,
    presses: f64,
    after: f64,
    p_sweet: f64,
}

fn row_for(decision: &[f64], stats: &Stats) -> Row {
    Row {
        rt: decision[0].round() as i32,
        ya: decision[1].round() as i32,
        ra: decision[2].round() as i32,
        fra: decision[3].round() as i32,
        work_fail: stats.mean_work_sleep,
        lunch: stats.mean_lunch,
        presses: stats.mean_presses,
        after: stats.mean_after,
        p_sweet: stats.p_sweet,
    }
}

fn print_header() {
    println!(
        "{:<6} {:>3} {:>3} {:>3}   {:>9}   {:>9}   {:>8}   {:>8}   {:>7}",
        "RT", "YA", "RA", "FRA", "work fail↓", "lunch↑", "presses↓", "after↓", "p_sweet",
    );
    println!("{}", "-".repeat(78));
}

fn print_row(label: &str, r: &Row) {
    let prefix = if label.is_empty() { String::new() } else { format!("{label} ") };
    println!(
        "{}{:<6} {:>3} {:>3} {:>3}   {:>9}   {:>9}   {:>7.2}/d   {:>8}   {:>6.1}%",
        prefix,
        fmt_rt(r.rt),
        r.ya,
        r.ra,
        r.fra,
        fmt_minutes(r.work_fail),
        fmt_minutes(r.lunch),
        r.presses,
        fmt_minutes(r.after),
        r.p_sweet * 100.0,
    );
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn main() {
    let problem = JigglyTuning::new(N_DAYS, 2026);

    let bounds = vec![
        (230.0, 250.0), // RT
        (20.0, 70.0),   // YELLOW_AT
        (10.0, 40.0),   // RED_AT
        (4.0, 20.0),    // FAST_RED_AT
    ];
    let initializer = RealBounds::new(bounds.clone());
    // Canonical NSGA-II/-III operator pair (SBX + PolyMut) with bounds.
    let variation = CompositeVariation {
        crossover: SimulatedBinaryCrossover::new(bounds.clone(), 30.0, 1.0),
        mutation: PolynomialMutation::new(bounds, 20.0, 1.0 / 4.0),
    };
    // M=4, H=6 → C(9,3) = 84 reference points. Match the population size.
    let pop = 84;
    let gens = 25;
    let config = Nsga3Config {
        population_size: pop,
        generations: gens,
        reference_divisions: 6,
        seed: 42,
    };

    println!("Optimizing jiggly's 4 lifecycle constants — 4-objective Pareto search");
    println!("  algorithm: NSGA-III (84 ref points, M=4, H=6)");
    println!("  N_DAYS:    {N_DAYS} simulated workdays per evaluation");
    println!("  search:    RT∈[230,250], YA∈[20,70], RA∈[10,40], FRA∈[4,20]");
    println!(
        "  budget:    {pop} pop × {gens} gens = {} evaluations",
        pop * (gens + 1)
    );
    println!();

    let mut opt = Nsga3::new(config, initializer, variation);
    let t0 = Instant::now();
    let result = opt.run(&problem);
    let elapsed = t0.elapsed();

    println!(
        "NSGA-III finished in {:.2}s ({} evaluations, |front|={})",
        elapsed.as_secs_f64(),
        result.evaluations,
        result.pareto_front.len(),
    );
    println!();

    // Materialize each Pareto member's full Stats so we can print rich rows.
    // Multiple f64 decisions can round to the same integer combo — dedupe.
    let mut seen = std::collections::HashSet::new();
    let mut rows: Vec<Row> = result
        .pareto_front
        .iter()
        .filter_map(|c| {
            let rt = c.decision[0].round() as i32;
            let ya = c.decision[1].round() as i32;
            let ra = c.decision[2].round() as i32;
            let fra = c.decision[3].round() as i32;
            if !seen.insert((rt, ya, ra, fra)) {
                return None;
            }
            let stats = problem.aggregate(rt, ya, ra, fra);
            Some(row_for(&c.decision, &stats))
        })
        .collect();
    // Drop any infeasible front entries (shouldn't happen for a converged
    // run, but guard anyway).
    rows.retain(|r| r.ya > r.ra && r.ra > r.fra && r.fra > 0);

    println!("=== Pareto front (sorted by lunch sleep, descending) ===");
    print_header();
    rows.sort_by(|a, b| b.lunch.partial_cmp(&a.lunch).unwrap_or(std::cmp::Ordering::Equal));
    for r in rows.iter().take(15) {
        print_row("", r);
    }
    if rows.len() > 15 {
        println!("  ... ({} more on the front)", rows.len() - 15);
    }
    println!();

    // Re-rank by each individual objective to surface extreme tradeoffs.
    let best_by = |key: fn(&Row) -> f64, want_high: bool| -> Option<&Row> {
        rows.iter().min_by(|a, b| {
            let ka = key(a);
            let kb = key(b);
            let cmp = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
            if want_high { cmp.reverse() } else { cmp }
        })
    };
    println!("=== extreme tradeoffs ===");
    print_header();
    if let Some(r) = best_by(|r| r.work_fail, false) {
        print_row("FEWEST WORK FAILS    ", r);
    }
    if let Some(r) = best_by(|r| r.lunch, true) {
        print_row("MOST LUNCH SLEEP     ", r);
    }
    if let Some(r) = best_by(|r| r.presses, false) {
        print_row("FEWEST PRESSES       ", r);
    }
    if let Some(r) = best_by(|r| r.after, false) {
        print_row("LEAST AFTER-HOURS    ", r);
    }
    println!();

    let shipping = problem.aggregate(240, 30, 25, 20);
    let shipping_row = row_for(&[240.0, 30.0, 25.0, 20.0], &shipping);
    println!("=== firmware shipping default (RT=4h00m YEL=30 RED=25 FST=20) ===");
    print_header();
    print_row("", &shipping_row);
}
