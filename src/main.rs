use std::io::prelude::*;

use base64::prelude::*;
use rand::prelude::*;

#[derive(Debug, Clone)]
struct Rect {
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
}

impl Rect {
    fn overlaps(&self, other: &Rect) -> bool {
        let x_overlap = self.x1 < other.x2 && self.x2 > other.x1;
        let y_overlap = self.y1 < other.y2 && self.y2 > other.y1;

        x_overlap && y_overlap
    }

    fn area(&self) -> i32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn serialize(&self) -> String {
        format!("[{},{},{},{}]", self.x1, self.y1, self.x2, self.y2)
    }

    fn serialize_size(&self) -> String {
        format!("[{},{}]", self.x2 - self.x1, self.y2 - self.y1)
    }

    fn can_merge(&self, other: &Rect) -> bool {
        let top_bottom = (other.y2 == self.y1 || self.y2 == other.y1)
            && self.x1 == other.x1
            && self.x2 == other.x2;
        let left_right = (other.x2 == self.x1 || self.x2 == other.x1)
            && self.y1 == other.y1
            && self.y2 == other.y2;

        top_bottom || left_right
    }

    fn merge(&self, other: &Rect) -> Option<Rect> {
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

    fn bottom_amount_touching(&self, other: &Rect) -> i32 {
        if self.y2 == other.y1 {
            // `other` is touching the top or bottom of `self`
            Self::amount_overlap(self.x1, self.x2, other.x1, other.x2)
        } else {
            0
        }
    }

    fn top_amount_touching(&self, other: &Rect) -> i32 {
        if other.y2 == self.y1 {
            // `other` is touching the top or bottom of `self`
            Self::amount_overlap(self.x1, self.x2, other.x1, other.x2)
        } else {
            0
        }
    }

    fn left_amount_touching(&self, other: &Rect) -> i32 {
        if other.x2 == self.x1 {
            // `other` is touching the left or right side of `self`
            Self::amount_overlap(self.y1, self.y2, other.y1, other.y2)
        } else {
            0
        }
    }

    fn right_amount_touching(&self, other: &Rect) -> i32 {
        if self.x2 == other.x1 {
            Self::amount_overlap(self.y1, self.y2, other.y1, other.y2)
        } else {
            0
        }
    }

    fn width(&self) -> i32 {
        self.x2 - self.x1
    }

    fn height(&self) -> i32 {
        self.y2 - self.y1
    }

    fn is_empty(&self) -> bool {
        self.x1 == self.x2 || self.y1 == self.y2
    }

    /// Cuts `other` out of `self`, returning the four resulting rectangles.
    ///
    /// Some or all of these rectangles may be empty.
    fn cut_out(&self, other: &Rect) -> impl Iterator<Item = Rect> {
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

    fn amount_overlap(a1: i32, a2: i32, b1: i32, b2: i32) -> i32 {
        if a1 < b2 && a2 > b1 {
            b2.min(a2) - a1.max(b1)
        } else {
            0
        }
    }
}

fn serialize_packing(packing: &[Rect]) -> String {
    let mut s = String::new();
    s += "[";
    for (i, rect) in packing.iter().enumerate() {
        s += &*rect.serialize();
        if i != packing.len() - 1 {
            s += ",";
        }
    }
    s += "]";
    BASE64_STANDARD.encode(s)
}

fn serialize_order(packing: &[Rect]) -> String {
    let mut s = String::new();
    s += "[";
    for (i, rect) in packing.iter().enumerate() {
        s += &*rect.serialize_size();
        if i != packing.len() - 1 {
            s += ",";
        }
    }
    s += "]";
    BASE64_STANDARD.encode(s)
}

fn bounds<'a>(packing: impl Iterator<Item = &'a Rect>) -> Rect {
    let mut bounds = Rect {
        x1: i32::MAX,
        y1: i32::MAX,
        x2: i32::MIN,
        y2: i32::MIN,
    };

    for rect in packing {
        bounds.x1 = bounds.x1.min(rect.x1);
        bounds.x2 = bounds.x2.max(rect.x2);
        bounds.y1 = bounds.y1.min(rect.y1);
        bounds.y2 = bounds.y2.max(rect.y2);
    }

    bounds
}

fn score<'a>(packing: impl Iterator<Item = &'a Rect>) -> i32 {
    let mut bounds = Rect {
        x1: i32::MAX,
        y1: i32::MAX,
        x2: i32::MIN,
        y2: i32::MIN,
    };

    let mut rects_area = 0;
    for rect in packing {
        bounds.x1 = bounds.x1.min(rect.x1);
        bounds.x2 = bounds.x2.max(rect.x2);
        bounds.y1 = bounds.y1.min(rect.y1);
        bounds.y2 = bounds.y2.max(rect.y2);
        rects_area += rect.area();
    }

    bounds.area() - rects_area
}

fn closeness_quotient(packing: &[Rect]) -> f32 {
    const POW: i32 = 4;

    let mut acc = 0.0;
    for a in packing.iter() {
        let (mut top, mut bottom, mut left, mut right) = (0, 0, 0, 0);
        for b in packing.iter() {
            top += a.top_amount_touching(b);
            bottom += a.bottom_amount_touching(b);
            left += a.left_amount_touching(b);
            right += a.right_amount_touching(b);
        }
        acc += (top as f32 / a.width() as f32).powi(POW);
        acc += (bottom as f32 / a.width() as f32).powi(POW);
        acc += (left as f32 / a.height() as f32).powi(POW);
        acc += (right as f32 / a.height() as f32).powi(POW);
    }
    acc
}

fn simplify(rects: &mut Vec<Rect>) {
    let mut i = 0;

    while i < rects.len() {
        let mut j = i + 1;

        while j < rects.len() {
            if let Some(merge) = rects[i].merge(&rects[j]) {
                rects[i] = merge;
                rects.remove(j);
            } else {
                j += 1;
            }
        }

        i += 1;
    }
}

fn invert(packing: &[Rect]) -> Vec<Rect> {
    let mut inverted = vec![bounds(packing.iter())];
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
    simplify(&mut inverted);
    inverted
}

fn main() {
    const RECTS_INIT: [Rect; 4] = [
        Rect {
            x1: 0,
            x2: 10,
            y1: 0,
            y2: 10,
        },
        Rect {
            x1: 0,
            x2: 20,
            y1: 0,
            y2: 10,
        },
        Rect {
            x1: 0,
            x2: 10,
            y1: 0,
            y2: 20,
        },
        Rect {
            x1: 0,
            x2: 20,
            y1: 0,
            y2: 20,
        },
    ];

    let trials = 100000;
    let mut output =
        std::io::BufWriter::with_capacity(65536, std::fs::File::create("out.csv").unwrap());

    let mut rects = RECTS_INIT;
    for _ in 0..trials {
        let mut score_val: i32 = i32::MAX;
        while score_val > 500 {
            while {
                rects = RECTS_INIT;
                !randomize_packing(&mut rects)
            } {}
            score_val = score(rects.iter());
        }
        output.write_all(score_val.to_string().as_bytes()).unwrap();
        output.write_all(b",").unwrap();
        output
            .write_all(serialize_packing(&invert(&rects)).as_bytes())
            .unwrap();
        output.write_all(b",").unwrap();
        output
            .write_all(serialize_packing(&rects).as_bytes())
            .unwrap();
        output.write_all(b",").unwrap();
        output
            .write_all(serialize_order(&rects).as_bytes())
            .unwrap();
        output.write_all(b",").unwrap();
        output
            .write_all(closeness_quotient(&rects).to_string().as_bytes())
            .unwrap();
        output.write_all(b"\n").unwrap();
    }
}

fn randomize_packing(rects: &mut [Rect]) -> bool {
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
