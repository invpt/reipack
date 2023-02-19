use std::mem::swap;

use rand::Rng;

use crate::geometry::Rect;

use crate::{PackingAlgorithm, PackingHeuristic, PackingHeuristicScore};

pub struct IterativeRandomPackingAlgorithm<H: PackingHeuristic> {
    pub heuristic: H,
    pub trials_per_iteration: usize,
}

impl<H: PackingHeuristic> PackingAlgorithm for IterativeRandomPackingAlgorithm<H> {
    fn pack(&self, rects: &mut [Rect]) -> bool {
        let mut rng = rand::thread_rng();

        let outer_bounds = Rect {
            x1: 0,
            y1: 0,
            x2: rects.iter().map(|r| r.x2 - r.x1).sum::<i32>(),
            y2: rects.iter().map(|r| r.y2 - r.y1).sum::<i32>(),
        };

        for i in 0..rects.len() {
            let mut best_score = None;

            let rect_index = rects
                .iter()
                .enumerate()
                .skip(i)
                .max_by(|(_, a), (_, b)| a.area().cmp(&b.area()))
                .unwrap()
                .0;

            for _ in 0..self.trials_per_iteration {
                let rect = &rects[rect_index];

                let width = rect.x2 - rect.x1;
                let height = rect.y2 - rect.y1;
                let x1 = rng.gen_range(outer_bounds.x1..outer_bounds.x2 - width);
                let y1 = rng.gen_range(outer_bounds.y1..outer_bounds.y2 - height);

                let mut rect = Rect {
                    x1,
                    y1,
                    x2: x1 + width,
                    y2: y1 + height,
                };

                if rects[..i].iter().any(|p| rect.overlaps(p)) {
                    // if this rectangle overlaps any of the existing ones, we won't consider it
                    continue;
                }

                swap(&mut rect, &mut rects[i]);

                let score = self.heuristic.score(&rects[..=i]);

                if let Some(best_score) = &mut best_score {
                    if score.is_better_than(best_score) {
                        *best_score = score;
                        if rect_index != i {
                            rects[rect_index] = rect;
                        }
                    } else {
                        swap(&mut rect, &mut rects[i]);
                    }
                } else {
                    best_score = Some(score);
                    if rect_index != i {
                        rects[rect_index] = rect;
                    }
                }
            }

            if best_score.is_none() {
                return false;
            }
        }

        true
    }
}
