use std::io::prelude::*;

use base64::prelude::*;
use rand::prelude::*;

#[derive(Debug)]
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

fn main() {
    const RECTS_INIT: [Rect; 4] = [
        Rect {
            x1: 0,
            x2: 1,
            y1: 0,
            y2: 1,
        },
        Rect {
            x1: 0,
            x2: 2,
            y1: 0,
            y2: 1,
        },
        Rect {
            x1: 0,
            x2: 1,
            y1: 0,
            y2: 2,
        },
        Rect {
            x1: 0,
            x2: 2,
            y1: 0,
            y2: 2,
        },
    ];

    let trials = 10000000;
    let mut output = std::io::BufWriter::with_capacity(65536, std::fs::File::create("out.csv").unwrap());

    let mut rects;
    for _ in 0..trials {
        while {rects = RECTS_INIT; !randomize_packing(&mut rects)} {}
        let score = score(rects.iter());
        output.write_all(score.to_string().as_bytes()).unwrap();
        output.write_all(b",").unwrap();
        output.write_all(serialize_packing(&rects).as_bytes()).unwrap();
        output.write_all(b",").unwrap();
        output.write_all(serialize_order(&rects).as_bytes()).unwrap();
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
        return false
    }

    true
}
