use rand::prelude::*;

use crate::RectChoiceAlgorithm;

pub struct RandomRectChoiceAlgorithm;

impl RectChoiceAlgorithm for RandomRectChoiceAlgorithm {
    fn choose(
        &self,
        _packing: &[crate::geometry::Rect],
        choices: &[crate::geometry::Rect],
    ) -> usize {
        rand::thread_rng().gen_range(0..choices.len())
    }

    fn nondeterministic_hint(&self) -> bool {
        true
    }
}
