use std::io::prelude::*;

use base64::prelude::*;
use rand::prelude::*;

#[derive(Debug)]
struct Rect {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl Rect {
    fn overlaps(&self, other: &Rect) -> bool {
        let x_overlap = self.x1 <= other.x2 && self.x2 >= other.x1 || other.x1 <= self.x2 && other.x2 >= self.x1;
        let y_overlap = self.y1 <= other.y2 && self.y2 >= other.y1 || other.y1 <= self.y2 && other.y2 >= self.y1;

        x_overlap && y_overlap
    }

    fn area(&self) -> f64 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn serialize(&self) -> String {
        format!("[{},{},{},{}]", self.x1, self.y1, self.x2, self.y2)
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

fn score<'a>(packing: impl Iterator<Item = &'a Rect>) -> f64 {
    let mut bounds = Rect {
        x1: f64::INFINITY,
        y1: f64::INFINITY,
        x2: -f64::INFINITY,
        y2: -f64::INFINITY,
    };

    let mut rects_area = 0.0;
    for rect in packing {
        bounds.x1 = bounds.x1.min(rect.x1);
        bounds.x2 = bounds.x2.max(rect.x2);
        bounds.y1 = bounds.y1.min(rect.y1);
        bounds.y2 = bounds.y2.max(rect.y2);
        rects_area += rect.area();
    }

    bounds.area() - rects_area
}

fn main() {
    const RECTS_INIT: [Rect; 4] = [
        Rect {
            x1: 0.0,
            x2: 1.0,
            y1: 0.0,
            y2: 1.0,
        },
        Rect {
            x1: 0.0,
            x2: 2.0,
            y1: 0.0,
            y2: 1.0,
        },
        Rect {
            x1: 0.0,
            x2: 1.0,
            y1: 0.0,
            y2: 2.0,
        },
        Rect {
            x1: 0.0,
            x2: 2.0,
            y1: 0.0,
            y2: 2.0,
        },
    ];

    let trials = 100000000;
    let mut output = std::io::BufWriter::with_capacity(65536, std::fs::File::create("out.csv").unwrap());

    let mut rects;
    for _ in 0..trials {
        while {rects = RECTS_INIT; !randomize_packing(&mut rects)} {}
        let score = score(rects.iter());
        if score < 1.0 {
            output.write(score.to_string().as_bytes()).unwrap();
            output.write(b",").unwrap();
            output.write(serialize_packing(&rects).as_bytes()).unwrap();
            output.write(b"\n").unwrap();
        }
    }
}

fn randomize_packing(rects: &mut [Rect]) -> bool {
    let outer_bounds = Rect {
        x1: 0.0,
        y1: 0.0,
        x2: rects.iter().map(|r| r.x2 - r.x1).sum::<f64>() + 0.1,
        y2: rects.iter().map(|r| r.y2 - r.y1).sum::<f64>() + 0.1,
    };

    'outer: for i in 1..=rects.len() {
        let (rect, prev) = rects[..i].split_last_mut().unwrap();
        for _ in 0..100 {
            let width = rect.x2 - rect.x1;
            let height = rect.y2 - rect.y1;
            let x1 = rand::thread_rng().gen_range(outer_bounds.x1..outer_bounds.x2 - width);
            let y1 = rand::thread_rng().gen_range(outer_bounds.y1..outer_bounds.y2 - height);
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
        return false
    }

    true
}
