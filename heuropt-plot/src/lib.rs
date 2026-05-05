//! Lightweight SVG plotting helpers for `heuropt` results.
//!
//! Two core primitives:
//!
//! - [`pareto_front_svg`] — render a 2-objective Pareto front as an
//!   SVG scatter plot with axes and labels.
//! - [`convergence_svg`] — render a per-generation "best-fitness so
//!   far" trace as an SVG line plot.
//!
//! Hand-rolled SVG output (no `plotters` / `tiny-skia` dep) so the
//! crate stays a tiny optional dependency. Output is a `String` of
//! valid SVG — write it to a file, embed it in HTML, or pipe it to a
//! browser.
//!
//! # Example
//!
//! ```
//! use heuropt::prelude::*;
//! use heuropt_plot::pareto_front_svg;
//!
//! let space = ObjectiveSpace::new(vec![
//!     Objective::minimize("f1"),
//!     Objective::minimize("f2"),
//! ]);
//! let front = vec![
//!     Candidate::new((), Evaluation::new(vec![0.0, 1.0])),
//!     Candidate::new((), Evaluation::new(vec![0.5, 0.5])),
//!     Candidate::new((), Evaluation::new(vec![1.0, 0.0])),
//! ];
//! let svg = pareto_front_svg(&front, &space, 600, 400, "Sample front");
//! assert!(svg.starts_with("<svg"));
//! assert!(svg.contains("</svg>"));
//! ```

use std::fmt::Write as _;

use heuropt::core::candidate::Candidate;
use heuropt::core::objective::ObjectiveSpace;

/// Render a 2-objective Pareto front as an SVG scatter plot.
///
/// `width` and `height` are the SVG viewport dimensions in pixels.
/// `title` is rendered at the top.
///
/// Points are plotted in minimization-oriented coordinates.
///
/// # Panics
///
/// If `objectives.len() != 2`.
pub fn pareto_front_svg<D>(
    front: &[Candidate<D>],
    objectives: &ObjectiveSpace,
    width: u32,
    height: u32,
    title: &str,
) -> String {
    assert_eq!(
        objectives.len(),
        2,
        "pareto_front_svg requires exactly 2 objectives",
    );
    let oriented: Vec<[f64; 2]> = front
        .iter()
        .map(|c| {
            let m = objectives.as_minimization(&c.evaluation.objectives);
            [m[0], m[1]]
        })
        .collect();
    let (xs_label, ys_label) = (
        objectives.objectives[0].name.as_str(),
        objectives.objectives[1].name.as_str(),
    );

    let (xmin, xmax) = bounds(oriented.iter().map(|p| p[0]));
    let (ymin, ymax) = bounds(oriented.iter().map(|p| p[1]));
    let xspan = (xmax - xmin).max(1e-12);
    let yspan = (ymax - ymin).max(1e-12);

    // Margins so axes/labels have room.
    let m_left = 60.0_f64;
    let m_right = 20.0_f64;
    let m_top = 40.0_f64;
    let m_bot = 50.0_f64;
    let plot_w = width as f64 - m_left - m_right;
    let plot_h = height as f64 - m_top - m_bot;

    let to_x = |v: f64| m_left + (v - xmin) / xspan * plot_w;
    // Y is inverted: lower minimization value → higher pixel.
    let to_y = |v: f64| m_top + plot_h - (v - ymin) / yspan * plot_h;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" \
        font-family=\"system-ui, sans-serif\" font-size=\"12\">",
    );
    let _ = writeln!(
        out,
        "  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\"/>",
    );
    let _ = writeln!(
        out,
        "  <text x=\"{x}\" y=\"22\" font-size=\"16\" font-weight=\"bold\">{title}</text>",
        x = m_left,
        title = escape_xml(title),
    );

    // Axes box.
    let _ = writeln!(
        out,
        "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#888\" />",
        m_left, m_top, plot_w, plot_h,
    );

    // X-axis ticks (3 ticks).
    for i in 0..=3 {
        let t = i as f64 / 3.0;
        let v = xmin + t * xspan;
        let x = to_x(v);
        let _ = writeln!(
            out,
            "  <line x1=\"{x}\" y1=\"{y0}\" x2=\"{x}\" y2=\"{y1}\" stroke=\"#888\" />",
            y0 = m_top + plot_h,
            y1 = m_top + plot_h + 5.0,
        );
        let _ = writeln!(
            out,
            "  <text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\">{v:.3}</text>",
            y = m_top + plot_h + 18.0,
        );
    }
    // Y-axis ticks.
    for i in 0..=3 {
        let t = i as f64 / 3.0;
        let v = ymin + t * yspan;
        let y = to_y(v);
        let _ = writeln!(
            out,
            "  <line x1=\"{x0}\" y1=\"{y}\" x2=\"{x1}\" y2=\"{y}\" stroke=\"#888\" />",
            x0 = m_left - 5.0,
            x1 = m_left,
        );
        let _ = writeln!(
            out,
            "  <text x=\"{x}\" y=\"{y}\" text-anchor=\"end\" dominant-baseline=\"middle\">{v:.3}</text>",
            x = m_left - 8.0,
        );
    }

    // Axis labels.
    let _ = writeln!(
        out,
        "  <text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\">{xs_label}</text>",
        x = m_left + plot_w / 2.0,
        y = height as f64 - 12.0,
        xs_label = escape_xml(xs_label),
    );
    let _ = writeln!(
        out,
        "  <text x=\"15\" y=\"{y}\" text-anchor=\"middle\" \
         transform=\"rotate(-90 15 {y})\">{ys_label}</text>",
        y = m_top + plot_h / 2.0,
        ys_label = escape_xml(ys_label),
    );

    // Points.
    for p in &oriented {
        let cx = to_x(p[0]);
        let cy = to_y(p[1]);
        let _ = writeln!(
            out,
            "  <circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"3\" fill=\"#1f77b4\" \
             stroke=\"#0d4a8a\" stroke-width=\"0.5\" />",
        );
    }

    out.push_str("</svg>");
    out
}

/// Render a per-generation "best fitness so far" trace as an SVG line
/// plot. `bests[i]` is the best fitness *after* generation `i`.
///
/// `direction_minimize` controls which way is "improvement": `true`
/// for minimize problems, `false` for maximize.
pub fn convergence_svg(
    bests: &[f64],
    width: u32,
    height: u32,
    title: &str,
    y_axis_label: &str,
    _direction_minimize: bool,
) -> String {
    let n = bests.len();
    if n == 0 {
        return format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">\
             <text x=\"10\" y=\"20\">{}</text></svg>",
            escape_xml(title)
        );
    }

    let (ymin, ymax) = bounds(bests.iter().copied());
    let yspan = (ymax - ymin).max(1e-12);
    let xspan = (n - 1).max(1) as f64;

    let m_left = 70.0_f64;
    let m_right = 20.0_f64;
    let m_top = 40.0_f64;
    let m_bot = 50.0_f64;
    let plot_w = width as f64 - m_left - m_right;
    let plot_h = height as f64 - m_top - m_bot;

    let to_x = |i: usize| m_left + (i as f64) / xspan * plot_w;
    let to_y = |v: f64| m_top + plot_h - (v - ymin) / yspan * plot_h;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" \
         font-family=\"system-ui, sans-serif\" font-size=\"12\">",
    );
    let _ = writeln!(
        out,
        "  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\"/>",
    );
    let _ = writeln!(
        out,
        "  <text x=\"{x}\" y=\"22\" font-size=\"16\" font-weight=\"bold\">{title}</text>",
        x = m_left,
        title = escape_xml(title),
    );
    let _ = writeln!(
        out,
        "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#888\" />",
        m_left, m_top, plot_w, plot_h,
    );

    // X axis: generation index.
    for i in 0..=4 {
        let t = i as f64 / 4.0;
        let g = (t * (n - 1) as f64).round() as usize;
        let x = to_x(g);
        let _ = writeln!(
            out,
            "  <line x1=\"{x}\" y1=\"{y0}\" x2=\"{x}\" y2=\"{y1}\" stroke=\"#888\" />",
            y0 = m_top + plot_h,
            y1 = m_top + plot_h + 5.0,
        );
        let _ = writeln!(
            out,
            "  <text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\">{g}</text>",
            y = m_top + plot_h + 18.0,
        );
    }
    // Y ticks.
    for i in 0..=3 {
        let t = i as f64 / 3.0;
        let v = ymin + t * yspan;
        let y = to_y(v);
        let _ = writeln!(
            out,
            "  <line x1=\"{x0}\" y1=\"{y}\" x2=\"{x1}\" y2=\"{y}\" stroke=\"#888\" />",
            x0 = m_left - 5.0,
            x1 = m_left,
        );
        let _ = writeln!(
            out,
            "  <text x=\"{x}\" y=\"{y}\" text-anchor=\"end\" dominant-baseline=\"middle\">{v:.3e}</text>",
            x = m_left - 8.0,
        );
    }

    // Axis labels.
    let _ = writeln!(
        out,
        "  <text x=\"{x}\" y=\"{y}\" text-anchor=\"middle\">generation</text>",
        x = m_left + plot_w / 2.0,
        y = height as f64 - 12.0,
    );
    let _ = writeln!(
        out,
        "  <text x=\"15\" y=\"{y}\" text-anchor=\"middle\" \
         transform=\"rotate(-90 15 {y})\">{label}</text>",
        y = m_top + plot_h / 2.0,
        label = escape_xml(y_axis_label),
    );

    // Polyline.
    let mut points = String::new();
    for (i, &v) in bests.iter().enumerate() {
        if i > 0 {
            points.push(' ');
        }
        let _ = write!(points, "{:.2},{:.2}", to_x(i), to_y(v));
    }
    let _ = writeln!(
        out,
        "  <polyline points=\"{points}\" fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"1.5\" />",
    );

    out.push_str("</svg>");
    out
}

fn bounds<I: IntoIterator<Item = f64>>(it: I) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in it {
        if v.is_finite() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    if lo.is_infinite() {
        (0.0, 1.0)
    } else if (hi - lo).abs() < f64::EPSILON {
        // All points equal — give a small artificial span.
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use heuropt::core::evaluation::Evaluation;
    use heuropt::core::objective::Objective;

    #[test]
    fn pareto_svg_well_formed() {
        let space = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
        let front = vec![
            Candidate::new((), Evaluation::new(vec![0.0, 1.0])),
            Candidate::new((), Evaluation::new(vec![1.0, 0.0])),
        ];
        let svg = pareto_front_svg(&front, &space, 400, 300, "test");
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn convergence_svg_well_formed() {
        let bests = vec![10.0, 5.0, 2.0, 1.0, 0.5];
        let svg = convergence_svg(&bests, 400, 300, "convergence", "best", true);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("polyline"));
    }

    #[test]
    fn convergence_empty_returns_valid_svg() {
        let svg = convergence_svg(&[], 200, 100, "empty", "y", true);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }
}
