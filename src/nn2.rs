use neun::{AdamOptimizer, GradientDescentOptimizer, Model, Optimizer, OptimizerInstance};
use rand::Rng;
use std::fs::File;
use std::io::prelude::*;

use crate::{geometry::Rect, serialize_packing};

pub fn nn() {
    const LEARN_RATE: f32 = 0.00005;
    let b1 = 0.9;
    let b2 = 0.99;
    let start_exploit = 0.25;
    let end_exploit = 0.75;
    let reward_threshold_start = 0.6;
    let reward_threshold_end = 0.9;
    const FUTURE_DISCOUNT: f32 = 0.75;
    const RECTS_INIT: [Rect; 4] = [
        Rect {
            x1: 0,
            x2: 2,
            y1: 0,
            y2: 2,
        },
        Rect {
            x1: 0,
            x2: 2,
            y1: 0,
            y2: 4,
        },
        Rect {
            x1: 0,
            x2: 4,
            y1: 0,
            y2: 2,
        },
        Rect {
            x1: 0,
            x2: 4,
            y1: 0,
            y2: 4,
        },
    ];

    let mut rng = rand::thread_rng();

    let bounds = Rect {
        x1: 0,
        y1: 0,
        x2: 12,
        y2: 12,
    };

    let model_dimensions = [bounds.area() as usize + 2, 96, 96, bounds.area() as usize];

    let model_dimensions_id: String = model_dimensions
        .iter()
        .map(|d| d.to_string())
        .fold(String::new(), |a, b| a + "-" + &b);
    let weights_name = format!("weights{model_dimensions_id}.bin");

    println!("{weights_name}");

    /*let mut weights_file = File::create("weights.bin").unwrap();
    for variable in model.variables() {
        weights_file.write_all(&variable.to_be_bytes()).unwrap();
    } */

    let mut model = Model::new(&model_dimensions);

    if std::path::Path::new(&weights_name).exists() {
        let mut weights_in = File::open(&weights_name).unwrap();
        let mut buf = vec![];
        weights_in.read_to_end(&mut buf).unwrap();
        let vars = buf.chunks_exact(4).map(|chunk| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            f32::from_be_bytes(buf)
        });

        model
            .variables_mut()
            .zip(vars)
            .for_each(|(mv, fv)| *mv = fv);
    }

    let mut driver = model.driver_mut();

    for _ in 0..10 {
        let mut rects = RECTS_INIT.clone();
        let mut placed = [false; RECTS_INIT.len()];
        let mut unplaced_count = placed.len();
        for _ in 0..RECTS_INIT.len() {
            // choose a rectangle
            let idx = unplaced(&placed)
                .nth(rng.gen_range(0..unplaced_count))
                .unwrap();

            let input = vectorize_input(
                &bounds,
                &rects,
                &placed,
                rects[idx].width(),
                rects[idx].height(),
            );
            let (x1, y1) = interpret(&bounds, &rects, driver.run(&input).output());
            let rect = Rect {
                x1,
                y1,
                x2: x1 + rects[idx].width(),
                y2: y1 + rects[idx].height(),
            };
            println!("{:?}", rect);
            rects[idx] = rect;
            placed[idx] = true;
            unplaced_count -= 1;
        }
        println!("{:?}", rects);
        println!("{:?}", serialize_packing(&rects));
    }

    let variable_count = driver.model().variable_count();
    let mut optimizer = AdamOptimizer {
        a: LEARN_RATE,
        b1,
        b2,
    }
    .instance(variable_count);

    let mut dx = vec![0.0; variable_count];

    let mut grads = 0usize;

    let mut cnt_valid = 0;
    let mut cnt_invalid = 0;

    let trials = 5_000_000usize;
    let mut trial = 0usize;
    'trials: while trial < trials {
        let fraction_complete = (trials - trial) as f32 / trials as f32;
        let exploit_chance =
            (fraction_complete) * start_exploit + (1.0 - fraction_complete) * end_exploit;
        let learn_rate_mod = fraction_complete;
        let reward_threshold = fraction_complete * reward_threshold_start + (1.0 - fraction_complete) * reward_threshold_end;
        if trial % 100_000 == 0 {
            cnt_valid = 0;
            cnt_invalid = 0;
        }

        if trial % 1000 == 0 {
            println!("Learn rate: {LEARN_RATE}, b1: {b1}, b2: {b2}, rew thresh: {reward_threshold}, dimensions: {model_dimensions:?}");
            println!(
                "{}% ({} of {} trials)",
                trial / (trials / 100),
                trial,
                trials
            );
            println!(
                "valid%: {:?}",
                100.0 * (cnt_valid as f32) / (cnt_valid + cnt_invalid) as f32
            );
        }

        let mut rects = RECTS_INIT.clone();
        let mut placed = [false; RECTS_INIT.len()];
        let mut actions = vec![];

        let mut unplaced_count = placed.len();
        while unplaced_count > 0 {
            // choose a rectangle
            let chosen_idx = unplaced(&placed)
                .nth(rng.gen_range(0..unplaced_count))
                .unwrap();

            if rng.gen_range(0.0..1.0) < exploit_chance {
                // attempt to find a position for the rectangle using the model
                let input = vectorize_input(
                    &bounds,
                    &rects,
                    &placed,
                    rects[chosen_idx].width(),
                    rects[chosen_idx].height(),
                );
                let result = driver.run(&input);
                let (x1, y1) = interpret(&bounds, &rects, result.output());
                let positioned = Rect {
                    x1,
                    y1,
                    x2: x1 + rects[chosen_idx].width(),
                    y2: y1 + rects[chosen_idx].height(),
                };

                if bounds.contains(&positioned) && rects
                    .iter()
                    .zip(placed)
                    .all(|(r, p)| !p || !r.overlaps(&positioned))
                {
                    rects[chosen_idx] = positioned;
                } else {
                    continue 'trials;
                }
            } else {
                // attempt to find a position for the rectangle randomly
                let mut tries = 0;
                rects[chosen_idx] = loop {
                    if tries >= 100 {
                        continue 'trials;
                    }

                    tries += 1;

                    let x = rng.gen_range(0..bounds.width() - rects[chosen_idx].width());
                    let y = rng.gen_range(0..bounds.height() - rects[chosen_idx].height());

                    let positioned = Rect {
                        x1: x,
                        y1: y,
                        x2: x + rects[chosen_idx].width(),
                        y2: y + rects[chosen_idx].height(),
                    };

                    if rects
                        .iter()
                        .zip(placed)
                        .all(|(r, p)| !p || !r.overlaps(&positioned))
                    {
                        break positioned;
                    }
                };
            }

            actions.push(chosen_idx);
            placed[chosen_idx] = true;
            unplaced_count -= 1;
        }

        let reward = reward(&bounds, &rects);

        if reward > reward_threshold {
            let mut q = (reward - reward_threshold) / (1.0 - reward_threshold);

            if trial % 1000 == 0 {
                println!();
                println!();
            }

            for chosen_idx in actions.iter().copied().rev() {
                placed[chosen_idx] = false;

                let input = vectorize_input(
                    &bounds,
                    &rects,
                    &placed,
                    rects[chosen_idx].width(),
                    rects[chosen_idx].height(),
                );
                let target = vectorize_output(&bounds, rects[chosen_idx].x1, rects[chosen_idx].y1);

                let result = driver.run_and_record(&input);

                let state = rects
                    .iter()
                    .zip(placed)
                    .filter(|(_, p)| *p)
                    .map(|(r, _)| r)
                    .collect::<Vec<_>>();
                let (x1, y1) = interpret(&bounds, &rects, result.output());
                let interp = Rect {
                    x1,
                    y1,
                    x2: x1 + rects[chosen_idx].width(),
                    y2: y1 + rects[chosen_idx].height(),
                };

                let valid = state.iter().all(|r| !r.overlaps(&interp));

                if valid {
                    cnt_valid += 1;
                } else {
                    cnt_invalid += 1;
                }

                if trial % 1000 == 0 {
                    println!();
                    println!(" input: {:?}", input);
                    println!("target: {:?}", target);
                    println!("result: {:?}", result.output());
                    println!(" state: {:?}", state);
                    println!("interp: {:?}", interp);
                    println!(" valid: {:?}", valid);
                    println!("Learn rate: {LEARN_RATE}, modified learn rate: {}, b1: {b1}, b2: {b2}, rew thresh: {reward_threshold}, exploit chance: {exploit_chance}, dimensions: {model_dimensions:?}", learn_rate_mod * LEARN_RATE);
                    println!(
                        "{}% ({} of {} trials)",
                        trial / (trials / 100),
                        trial,
                        trials
                    );
                    println!(
                        "valid%: {:?}",
                        100.0 * (cnt_valid as f32) / (cnt_valid + cnt_invalid) as f32
                    );
                }

                result.compute_gradients(&target, |idx, val| dx[idx] += learn_rate_mod * q * val);

                if grads % 96 == 0 {
                    dx.iter_mut().for_each(|dx| *dx /= 96.0);
                    optimizer.apply(driver.model_mut().variables_mut().zip(dx.iter()));
                    dx.iter_mut().for_each(|dx| *dx = 0.0);
                }

                grads += 1;

                q *= FUTURE_DISCOUNT;
            }

            trial += 1;
        }
    }

    let mut rects = RECTS_INIT.clone();
    let mut placed = [false; RECTS_INIT.len()];
    let mut unplaced_count = placed.len();
    for _ in 0..RECTS_INIT.len() {
        // choose a rectangle
        let idx = unplaced(&placed)
            .nth(rng.gen_range(0..unplaced_count))
            .unwrap();

        let input = vectorize_input(
            &bounds,
            &rects,
            &placed,
            rects[idx].width(),
            rects[idx].height(),
        );
        let (x1, y1) = interpret(&bounds, &rects, driver.run(&input).output());
        let rect = Rect {
            x1,
            y1,
            x2: x1 + rects[idx].width(),
            y2: y1 + rects[idx].height(),
        };
        println!("{:?}", rect);
        rects[idx] = rect;
        placed[idx] = true;
        unplaced_count -= 1;
    }
    println!("{:?}", rects);
    println!("{:?}", serialize_packing(&rects));

    let mut weights_file = File::create(&weights_name).unwrap();
    for variable in model.variables() {
        weights_file.write_all(&variable.to_be_bytes()).unwrap();
    }
}

fn vectorize_input(
    bounds: &Rect,
    rects: &[Rect],
    placed: &[bool],
    chosen_width: i32,
    chosen_height: i32,
) -> Vec<f32> {
    let buf_len = bounds.area() as usize + 2;
    let mut buf = vec![1.0; bounds.area() as usize + 2];

    let placed_rects = rects.iter().zip(placed).filter(|(_, &p)| p).map(|(r, _)| r);

    for rect in placed_rects {
        for x in (rect.x1 - chosen_width + 1).max(0)..rect.x2 {
            for y in (rect.y1 - chosen_height + 1).max(0)..rect.y2 {
                buf[(x * bounds.height() + y) as usize] = 0.0;
            }
        }
    }

    buf[buf_len - 2] = chosen_width as f32;
    buf[buf_len - 1] = chosen_height as f32;

    buf
}

fn vectorize_output(bounds: &Rect, chosen_x: i32, chosen_y: i32) -> Vec<f32> {
    let mut buf = vec![0.0; bounds.area() as usize];

    buf[(bounds.height() * chosen_x) as usize + chosen_y as usize] = 1.0;

    buf
}

fn interpret(bounds: &Rect, rects: &[Rect], buf: &[f32]) -> (i32, i32) {
    let chosen_pos_index = buf
        .iter()
        .enumerate()
        .reduce(|a, b| if a.1 > b.1 { a } else { b })
        .unwrap()
        .0;
    let chosen_x1 = chosen_pos_index as i32 / bounds.height();
    let chosen_y1 = chosen_pos_index as i32 % bounds.height();

    (chosen_x1, chosen_y1)
}

fn reward(bounds: &Rect, rects: &[Rect]) -> f32 {
    let worst = bounds.area();
    let best = {
        let max_width = rects.iter().map(Rect::width).max().unwrap();
        let max_height = rects.iter().map(Rect::height).max().unwrap();
        max_width * max_height
    };
    let actual = crate::heuristics::score::score(rects);

    ((worst - actual) as f32 / (worst - best) as f32).powi(2)
}

fn unplaced(placed: &[bool]) -> impl Iterator<Item = usize> + '_ {
    placed
        .iter()
        .enumerate()
        .filter(|(_, &p)| !p)
        .map(|(i, _)| i)
}
