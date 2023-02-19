use crate::geometry::Rect;

use crate::{PackingHeuristic, PackingHeuristicScore};

pub fn score(packing: &[Rect]) -> i32 {
    let mut bounds = Rect {
        x1: i32::MAX,
        y1: i32::MAX,
        x2: i32::MIN,
        y2: i32::MIN,
    };

    let mut rects_area = 0;
    for rect in packing.iter() {
        bounds.x1 = bounds.x1.min(rect.x1);
        bounds.x2 = bounds.x2.max(rect.x2);
        bounds.y1 = bounds.y1.min(rect.y1);
        bounds.y2 = bounds.y2.max(rect.y2);
        rects_area += rect.area();
    }

    bounds.area() - rects_area
}

pub struct ScorePackingHeuristic;

impl PackingHeuristic for ScorePackingHeuristic {
    type Score = i32;

    fn score(&self, packing: &[Rect]) -> i32 {
        score(packing)
    }
}

impl PackingHeuristicScore<ScorePackingHeuristic> for i32 {
    fn is_better_than(&self, other: &Self) -> bool {
        self < other
    }

    fn best(scores: &[Self]) -> Option<usize> {
        scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.cmp(b.1))
            .map(|(i, _)| i)
    }
}
