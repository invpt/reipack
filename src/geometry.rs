use serde::{de::Error, ser::SerializeSeq, Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone)]
pub struct Interval {
    pub start: i32,
    pub end: i32,
}

impl Interval {
    pub const ZERO: Interval = Interval { start: 0, end: 0 };

    /// Creates an interval from `start` to `end`.
    pub const fn new(start: i32, end: i32) -> Interval {
        Interval { start, end }
    }

    /// Calculates the length of this interval.
    pub const fn len(&self) -> i32 {
        self.end - self.start
    }

    /// Returns true if this interval overlaps with `other`.
    pub const fn overlaps(&self, other: &Interval) -> bool {
        self.start < other.end && self.end > other.start
    }

    /// Returns true if this interval fully contains `other`.
    pub const fn contains(&self, other: &Interval) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Calculates the intersection between `self` and `other`.
    pub fn intersection(&self, other: &Interval) -> Interval {
        if self.overlaps(other) {
            Interval {
                start: self.start.max(other.start),
                end: other.end.min(self.end),
            }
        } else {
            Interval::ZERO
        }
    }
}

#[derive(Debug, Clone)]
pub enum Side {
    Left,
    Top,
    Right,
    Bottom,
}

#[derive(Debug, Clone)]
pub struct Rect {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl Rect {
    /// A rectangle with zero size and zero values of all four coordinates.
    pub const ZERO: Rect = Rect {
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 0,
    };

    /// Calculates the minimal bounding box of a sequence of rectangles.
    ///
    /// Returns a zero rectangle if `rects` is empty.
    pub fn bbox<'a>(rects: impl Iterator<Item = &'a Rect>) -> Rect {
        rects
            .cloned()
            .reduce(|a, b| Rect {
                x1: a.x1.min(b.x1),
                y1: a.y1.min(b.y1),
                x2: a.x2.max(b.x2),
                y2: a.y2.max(b.y2),
            })
            .unwrap_or(Rect::ZERO)
    }

    /// Calculates the width of this rectangle.
    pub const fn width(&self) -> i32 {
        self.x2 - self.x1
    }

    /// Calculates the height of this rectangle.
    pub const fn height(&self) -> i32 {
        self.y2 - self.y1
    }

    /// Calculates the size of this rectangle.
    pub const fn size(&self) -> Size {
        Size {
            width: self.width(),
            height: self.height(),
        }
    }

    /// Returns the horizontal range of this rectangle as an [Interval].
    pub const fn horz(&self) -> Interval {
        Interval {
            start: self.x1,
            end: self.x2,
        }
    }

    /// Returns the vertical range of this rectangle as an [Interval].
    pub const fn vert(&self) -> Interval {
        Interval {
            start: self.y1,
            end: self.y2,
        }
    }

    /// Calculates this rectangle's area.
    pub const fn area(&self) -> i32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    /// Returns true if this rectangle has an area of zero.
    pub const fn is_empty(&self) -> bool {
        self.x1 == self.x2 || self.y1 == self.y2
    }

    /// Returns true if `self` overlaps with `other`.
    pub const fn overlaps(&self, other: &Rect) -> bool {
        self.horz().overlaps(&other.horz()) && self.vert().overlaps(&other.vert())
    }

    /// Returns true if `self` fully contains `other`.
    pub const fn contains(&self, other: &Rect) -> bool {
        self.horz().contains(&other.horz()) && self.vert().contains(&other.vert())
    }

    /// Merges this rectangle with `other` into a single resulting rectangle, if possible.
    pub fn merge(&self, other: &Rect) -> Option<Rect> {
        if self.can_merge(other) {
            Some(Rect {
                x1: self.x1.min(other.x1),
                x2: self.x2.max(other.x2),
                y1: self.y1.min(other.y1),
                y2: self.y2.max(other.y2),
            })
        } else {
            None
        }
    }

    /// Returns true if this rectangle could be merged with `other` into a single rectangle.
    pub const fn can_merge(&self, other: &Rect) -> bool {
        let top_bottom = (other.y2 == self.y1 || self.y2 == other.y1)
            && self.x1 == other.x1
            && self.x2 == other.x2;
        let left_right = (other.x2 == self.x1 || self.x2 == other.x1)
            && self.y1 == other.y1
            && self.y2 == other.y2;

        top_bottom || left_right
    }

    /// Calculates the side and length that `other` touches this rectangle.
    pub fn amount_touching(&self, other: &Rect) -> Option<(Side, i32)> {
        if other.x2 == self.x1 {
            Some((Side::Left, self.vert().intersection(&other.vert()).len()))
        } else if other.y2 == self.y1 {
            Some((Side::Top, self.horz().intersection(&other.horz()).len()))
        } else if self.x2 == other.x1 {
            Some((Side::Right, self.vert().intersection(&other.vert()).len()))
        } else if self.y2 == other.y1 {
            Some((Side::Bottom, self.horz().intersection(&other.horz()).len()))
        } else {
            None
        }
    }

    /// Cuts `other` out of `self`, returning an iterator of up to four resulting rectangles.
    ///
    /// Some or all of these rectangles may be empty.
    pub fn cut_out(&self, other: &Rect) -> impl Iterator<Item = Rect> {
        let left_rect_width = (other.x1 - self.x1).clamp(0, self.width());
        let top_rect_height = (other.y1 - self.y1).clamp(0, self.height());
        let right_rect_width = (self.x2 - other.x2).clamp(0, self.width());
        let bottom_rect_height = (self.y2 - other.y2).clamp(0, self.height());

        let left = Rect {
            x1: self.x1,
            x2: self.x1 + left_rect_width,
            y1: self.y1,
            y2: self.y2 - bottom_rect_height,
        };

        let top = Rect {
            x1: self.x1 + left_rect_width,
            x2: self.x2,
            y1: self.y1,
            y2: self.y1 + top_rect_height,
        };

        let right = Rect {
            x1: self.x2 - right_rect_width,
            x2: self.x2,
            y1: self.y1 + top_rect_height,
            y2: self.y2,
        };

        let bottom = Rect {
            x1: self.x1,
            x2: self.x2 - right_rect_width,
            y1: self.y2 - bottom_rect_height,
            y2: self.y2,
        };

        [left, top, right, bottom]
            .into_iter()
            .filter(|r| !r.is_empty())
    }

    pub fn inverse(packing: &[Rect]) -> Vec<Rect> {
        let mut inverted = vec![Rect::bbox(packing.iter())];
        for rect in packing {
            let mut i = 0;
            while i < inverted.len() {
                let mut iter = inverted[i].cut_out(rect);
                if let Some(first) = iter.next() {
                    inverted[i] = first;

                    for additional in iter {
                        inverted.push(additional);
                    }

                    i += 1;
                } else {
                    inverted.remove(i);
                }
            }
        }
        Self::simplify(&mut inverted);
        inverted
    }

    pub fn simplify(rects: &mut Vec<Rect>) {
        let mut i = 0;

        while i < rects.len() {
            let mut j = 0;

            while j < rects.len() {
                if j == i {
                    j += 1;
                    continue;
                }

                if let Some(merged) = rects[i].merge(&rects[j]) {
                    rects[i] = merged;
                    rects.remove(j);
                    j = 0;
                    if j < i {
                        i -= 1;
                    }
                } else {
                    j += 1;
                }
            }

            i += 1;
        }
    }
}

impl Serialize for Size {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(4))?;
        seq.serialize_element(&self.width)?;
        seq.serialize_element(&self.height)?;
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Size {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SizeVisitor;

        impl<'de> serde::de::Visitor<'de> for SizeVisitor {
            type Value = Size;

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let width = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                let height = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                Ok(Size { width, height })
            }

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "an array with a length of 2")
            }
        }

        deserializer.deserialize_seq(SizeVisitor)
    }
}

impl Serialize for Rect {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(4))?;
        seq.serialize_element(&self.x1)?;
        seq.serialize_element(&self.y1)?;
        seq.serialize_element(&self.x2)?;
        seq.serialize_element(&self.y2)?;
        seq.end()
    }
}

impl<'de> Deserialize<'de> for Rect {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RectVisitor;

        impl<'de> serde::de::Visitor<'de> for RectVisitor {
            type Value = Rect;

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let x1 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                let y1 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                let x2 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(2, &self))?;
                let y2 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(3, &self))?;
                Ok(Rect { x1, y1, x2, y2 })
            }

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "an array with a length of 4")
            }
        }

        deserializer.deserialize_seq(RectVisitor)
    }
}
