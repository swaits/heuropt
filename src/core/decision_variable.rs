//! Optional schema describing a decision variable — name, label, unit,
//! and bounds. Returned by [`Problem::decision_schema`](super::Problem::decision_schema)
//! and consumed by the explorer JSON export so that the webapp can
//! render decision-variable axes with the user's preferred labels and
//! units.
//!
//! The `Problem` trait's default `decision_schema()` returns an empty
//! `Vec`, in which case the exporter generates fallback names like
//! `x[0]`, `x[1]`. Override `decision_schema()` to provide pretty
//! names, units, and bounds.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Schema for one decision variable. All fields except `name` are
/// optional; the explorer falls back to sensible defaults when
/// they're absent.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionVariable {
    /// Canonical short identifier (e.g. `"displacement"`).
    pub name: String,
    /// Human-readable display label (e.g. `"Engine size"`).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub label: Option<String>,
    /// Display unit (e.g. `"L"`, `"kg"`, `"Cd"`).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub unit: Option<String>,
    /// Lower bound, if known.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub min: Option<f64>,
    /// Upper bound, if known.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub max: Option<f64>,
}

impl DecisionVariable {
    /// Construct a `DecisionVariable` with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            label: None,
            unit: None,
            min: None,
            max: None,
        }
    }

    /// Attach a human-readable display label. Builder-style.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Attach a display unit string. Builder-style.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Attach lower / upper bounds. Builder-style.
    pub fn with_bounds(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_with_only_name() {
        let v = DecisionVariable::new("displacement");
        assert_eq!(v.name, "displacement");
        assert!(v.label.is_none());
        assert!(v.unit.is_none());
        assert!(v.min.is_none());
        assert!(v.max.is_none());
    }

    #[test]
    fn builder_methods_chain() {
        let v = DecisionVariable::new("displacement")
            .with_label("Engine size")
            .with_unit("L")
            .with_bounds(1.0, 6.0);
        assert_eq!(v.label.as_deref(), Some("Engine size"));
        assert_eq!(v.unit.as_deref(), Some("L"));
        assert_eq!(v.min, Some(1.0));
        assert_eq!(v.max, Some(6.0));
    }
}
