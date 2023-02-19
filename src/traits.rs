use crate::geometry::Rect;

pub trait PackingAlgorithm {
    /// Attempts to find a valid packing using the given rectangles.
    ///
    /// If `true` is returned, then a packing was found; otherwise, `false` is returned.
    fn pack(&self, packing: &mut [Rect]) -> bool;
}

pub trait RectChoiceAlgorithm {
    /// Chooses a rectangle from among the given `choices`, potentially taking into consideration
    /// the given partial `packing`.
    ///
    /// The result is implementation-defined if `choices` is empty.
    fn choose(&self, packing: &[Rect], choices: &[Rect]) -> usize;

    /// Returns `true` if this choice algorithm is hinted to be nondeterministic.
    fn nondeterministic_hint(&self) -> bool {
        false
    }
}

pub trait PackingHeuristic {
    type Score: PackingHeuristicScore<Self>;

    /// Returns the score of the given `packing`, which must be valid for the score to be valid.
    fn score(&self, packing: &[Rect]) -> Self::Score;
}

pub trait PackingHeuristicScore<H: PackingHeuristic + ?Sized> {
    /// Returns true if `self` is a better score than `other`.
    fn is_better_than(&self, other: &Self) -> bool;

    /// Returns the index of the best score. If `scores` is empty, returns `None`.
    fn best(scores: &[Self]) -> Option<usize>
    where
        Self: Sized;
}
