use std::io::prelude::*;

use base64::prelude::*;
use rand::prelude::*;

mod geometry;

use geometry::Rect;

use crate::geometry::Side;

fn serialize_packing(packing: &[Rect]) -> String {
    BASE64_STANDARD.encode(serde_json::to_string(packing).unwrap())
}

fn serialize_order(packing: &[Rect]) -> String {
    BASE64_STANDARD.encode(serde_json::to_string(&packing.iter().map(|r| r.size()).collect::<Vec<_>>()).unwrap())
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
    simplify(&mut inverted);
    inverted
}

fn spread_score(packing: &[Rect]) -> f64 {
    const M: f64 = 0.5;
    const K: f64 = 0.75;
    let s = packing
        .iter()
        .map(|r| (r.width() + r.height()) as f64)
        .sum::<f64>()
        / (2 * packing.len()) as f64;
    let mapping = |x: f64| (2.0 * M * s.sqrt() * x.sqrt() + s * (K - 2.0 * M)).max(if x < s { x } else { 0.0 });
    let inverted = invert(packing);
    inverted
        .iter()
        .map(|r| mapping(r.area() as f64))
        .sum()
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
            .write_all(spread_score(&rects).to_string().as_bytes())
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
