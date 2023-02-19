use rand::seq::SliceRandom;
use rand::Rng;

use crate::geometry::Rect;

use crate::PackingAlgorithm;

pub struct RandomPackingAlgorithm;

impl PackingAlgorithm for RandomPackingAlgorithm {
    fn pack(&self, rects: &mut [Rect]) -> bool {
        let mut rng = rand::thread_rng();

        rects.shuffle(&mut rng);

        let outer_bounds = Rect {
            x1: 0,
            y1: 0,
            x2: rects.iter().map(|r| r.x2 - r.x1).sum::<i32>(),
            y2: rects.iter().map(|r| r.y2 - r.y1).sum::<i32>(),
        };

        'outer: for i in 1..=rects.len() {
            let (rect, prev) = rects[..i].split_last_mut().unwrap();
            for _ in 0..100 {
                let width = rect.x2 - rect.x1;
                let height = rect.y2 - rect.y1;
                let x1 = rng.gen_range(outer_bounds.x1..outer_bounds.x2 - width);
                let y1 = rng.gen_range(outer_bounds.y1..outer_bounds.y2 - height);
                let new_rect = Rect {
                    x1,
                    y1,
                    x2: x1 + width,
                    y2: y1 + height,
                };
                if prev.iter().all(|p| !new_rect.overlaps(p)) {
                    *rect = new_rect;
                    continue 'outer;
                }
            }
            return false;
        }

        true
    }
}
