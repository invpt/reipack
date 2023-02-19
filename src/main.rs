use std::io::prelude::*;

use base64::prelude::*;

mod algorithms;
mod geometry;
mod heuristics;
mod traits;
pub use traits::*;

use geometry::Rect;

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

    let trials = 10000;
    let mut output =
        std::io::BufWriter::with_capacity(65536, std::fs::File::create("out.csv").unwrap());

    let algorithm = algorithms::packing::iterative_random::IterativeRandomPackingAlgorithm {
        heuristic: heuristics::spread_score::SpreadScorePackingHeuristic,
        trials_per_iteration: 1000,
    };

    let mut rects;
    for _ in 0..trials {
        while {
            rects = RECTS_INIT;
            !algorithm.pack(&mut rects)
        } {}

        let score_val = heuristics::score::score(&rects);
        output.write_all(score_val.to_string().as_bytes()).unwrap();
        output.write_all(b",").unwrap();
        output
            .write_all(
                heuristics::spread_score::spread_score(&rects)
                    .to_string()
                    .as_bytes(),
            )
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
            .write_all(
                heuristics::closeness_score::closeness_score(&rects)
                    .to_string()
                    .as_bytes(),
            )
            .unwrap();
        output.write_all(b"\n").unwrap();
    }
}

fn serialize_packing(packing: &[Rect]) -> String {
    BASE64_STANDARD.encode(serde_json::to_string(packing).unwrap())
}

fn serialize_order(packing: &[Rect]) -> String {
    BASE64_STANDARD.encode(
        serde_json::to_string(&packing.iter().map(|r| r.size()).collect::<Vec<_>>()).unwrap(),
    )
}
