//! Objective directions, named objectives, and the objective space.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Whether an objective should be minimized or maximized.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Smaller objective values are better.
    Minimize,
    /// Larger objective values are better.
    Maximize,
}

/// A named objective and its optimization direction.
///
/// `name` is the canonical short identifier (used as a key). The
/// optional `label` is a human-readable display name (e.g. "Price"
/// vs the technical name `"price_thousand_dollars"`). The optional
/// `unit` is a display unit string (e.g. `"$k"`, `"s"`, `"dB"`).
/// Both flow through to the explorer JSON export so the webapp can
/// render axes with the user's preferred labels and units.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Objective {
    /// Canonical short identifier, used as a key.
    pub name: String,
    /// Whether to minimize or maximize.
    pub direction: Direction,
    /// Human-readable display name (defaults to `name` if not set).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub label: Option<String>,
    /// Display unit, e.g. `"$k"`, `"s"`, `"dB"`.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub unit: Option<String>,
}

impl Objective {
    /// Create a minimize objective with the given name.
    pub fn minimize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: Direction::Minimize,
            label: None,
            unit: None,
        }
    }

    /// Create a maximize objective with the given name.
    pub fn maximize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: Direction::Maximize,
            label: None,
            unit: None,
        }
    }

    /// Attach a human-readable display label.
    ///
    /// Builder-style; consumes and returns `self`.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Attach a display unit string (e.g. `"$k"`, `"seconds"`, `"dB"`).
    ///
    /// Builder-style; consumes and returns `self`.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

/// The collection of objectives that define a problem's objective space.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveSpace {
    /// Objectives in declaration order.
    pub objectives: Vec<Objective>,
}

impl ObjectiveSpace {
    /// Build an objective space from the given objectives.
    pub fn new(objectives: Vec<Objective>) -> Self {
        Self { objectives }
    }

    /// Number of objectives.
    pub fn len(&self) -> usize {
        self.objectives.len()
    }

    /// Returns `true` if there are zero objectives.
    pub fn is_empty(&self) -> bool {
        self.objectives.is_empty()
    }

    /// Returns `true` if there is exactly one objective.
    pub fn is_single_objective(&self) -> bool {
        self.objectives.len() == 1
    }

    /// Returns `true` if there are two or more objectives.
    pub fn is_multi_objective(&self) -> bool {
        self.objectives.len() >= 2
    }

    /// Convert objective values into minimization orientation.
    ///
    /// Minimize objectives are returned unchanged; Maximize objectives are
    /// negated. In v1, this zips to the shorter of the two lengths.
    pub fn as_minimization(&self, values: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            values.len(),
            self.objectives.len(),
            "objective value count must match ObjectiveSpace length",
        );
        self.objectives
            .iter()
            .zip(values.iter())
            .map(|(obj, &v)| match obj.direction {
                Direction::Minimize => v,
                Direction::Maximize => -v,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_constructor_sets_direction() {
        let o = Objective::minimize("cost");
        assert_eq!(o.name, "cost");
        assert_eq!(o.direction, Direction::Minimize);
    }

    #[test]
    fn maximize_constructor_sets_direction() {
        let o = Objective::maximize("accuracy");
        assert_eq!(o.name, "accuracy");
        assert_eq!(o.direction, Direction::Maximize);
    }

    #[test]
    fn label_and_unit_default_to_none_and_round_trip_through_builders() {
        let o = Objective::minimize("price");
        assert!(o.label.is_none());
        assert!(o.unit.is_none());

        let o = Objective::minimize("price")
            .with_label("Price")
            .with_unit("$k");
        assert_eq!(o.label.as_deref(), Some("Price"));
        assert_eq!(o.unit.as_deref(), Some("$k"));
        assert_eq!(o.direction, Direction::Minimize);
        assert_eq!(o.name, "price");
    }

    #[test]
    fn as_minimization_negates_maximize_only() {
        let space = ObjectiveSpace::new(vec![
            Objective::minimize("cost"),
            Objective::maximize("accuracy"),
        ]);
        assert_eq!(space.as_minimization(&[10.0, 0.8]), vec![10.0, -0.8]);
    }

    #[test]
    fn lengths_and_predicates() {
        let single = ObjectiveSpace::new(vec![Objective::minimize("f")]);
        assert!(single.is_single_objective());
        assert!(!single.is_multi_objective());
        assert!(!single.is_empty());
        assert_eq!(single.len(), 1);

        let multi = ObjectiveSpace::new(vec![Objective::minimize("f1"), Objective::minimize("f2")]);
        assert!(multi.is_multi_objective());
        assert!(!multi.is_single_objective());

        let empty = ObjectiveSpace::new(Vec::new());
        assert!(empty.is_empty());
    }
}
