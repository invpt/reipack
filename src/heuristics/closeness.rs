use crate::geometry::{Rect, Side};

use crate::{PackingHeuristic, PackingHeuristicScore};

pub fn closeness_score(packing: &[Rect]) -> f32 {
    const POW: i32 = 4;

    let mut acc = 0.0;
    for a in packing.iter() {
        let (mut top, mut bottom, mut left, mut right) = (0, 0, 0, 0);
        for b in packing.iter() {
            match a.amount_touching(b) {
                Some((Side::Left, amt)) => left += amt,
                Some((Side::Top, amt)) => top += amt,
                Some((Side::Right, amt)) => right += amt,
                Some((Side::Bottom, amt)) => bottom += amt,
                None => (),
            }
        }
        acc += (left as f32 / a.height() as f32).powi(POW);
        acc += (top as f32 / a.width() as f32).powi(POW);
        acc += (right as f32 / a.height() as f32).powi(POW);
        acc += (bottom as f32 / a.width() as f32).powi(POW);
    }
    acc
}

pub struct ClosenessPackingHeuristic;

impl PackingHeuristic for ClosenessPackingHeuristic {
    type Score = f32;

    fn score(&self, packing: &[Rect]) -> f32 {
        closeness_score(packing)
    }
}

impl PackingHeuristicScore<ClosenessPackingHeuristic> for f32 {
    fn is_better_than(&self, other: &Self) -> bool {
        self > other
    }

    fn best(scores: &[Self]) -> Option<usize> {
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Greater))
            .map(|(i, _)| i)
    }
}
