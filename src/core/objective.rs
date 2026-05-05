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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Objective {
    /// Human-readable name of the objective.
    pub name: String,
    /// Whether to minimize or maximize.
    pub direction: Direction,
}

impl Objective {
    /// Create a minimize objective with the given name.
    pub fn minimize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: Direction::Minimize,
        }
    }

    /// Create a maximize objective with the given name.
    pub fn maximize(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: Direction::Maximize,
        }
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
