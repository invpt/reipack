use crate::geometry::Rect;

use crate::{PackingHeuristic, PackingHeuristicScore};

pub fn spread_score(packing: &[Rect]) -> f64 {
    const M: f64 = 0.5;
    const K: f64 = 0.75;
    let s = packing
        .iter()
        .map(|r| (r.width() + r.height()) as f64)
        .sum::<f64>()
        / (2 * packing.len()) as f64;
    // https://www.desmos.com/calculator/ak1wgpjjdo
    let mapping = |x: f64| {
        (2.0 * M * s.sqrt() * x.sqrt() + s * (K - 2.0 * M)).max(if x < s.sqrt() { x } else { 0.0 })
    };
    let inverted = Rect::inverse(packing);
    inverted.iter().map(|r| mapping(r.area() as f64)).sum()
}

pub struct SpreadPackingHeuristic;

impl PackingHeuristic for SpreadPackingHeuristic {
    type Score = f64;

    fn score(&self, packing: &[Rect]) -> f64 {
        spread_score(packing)
    }
}

impl PackingHeuristicScore<SpreadPackingHeuristic> for f64 {
    fn is_better_than(&self, other: &Self) -> bool {
        self < other
    }

    fn best(scores: &[Self]) -> Option<usize> {
        scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Greater))
            .map(|(i, _)| i)
    }
}
