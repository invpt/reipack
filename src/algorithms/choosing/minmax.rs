use crate::RectChoiceAlgorithm;

pub type MinRectChoiceAlgorithm = MinMaxRectChoiceAlgorithm<true>;
pub type MaxRectChoiceAlgorithm = MinMaxRectChoiceAlgorithm<false>;

#[doc(hidden)]
pub struct MinMaxRectChoiceAlgorithm<const CHOOSE_MIN: bool>;

impl<const CHOOSE_MIN: bool> RectChoiceAlgorithm for MinMaxRectChoiceAlgorithm<CHOOSE_MIN> {
    fn choose(
        &self,
        _packing: &[crate::geometry::Rect],
        choices: &[crate::geometry::Rect],
    ) -> usize {
        if CHOOSE_MIN {
            choices
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.area().cmp(&b.area()))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            choices
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.area().cmp(&b.area()))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }
}
