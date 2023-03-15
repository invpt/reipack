use neun::{GradientDescentOptimizer, Model, Optimizer, OptimizerInstance, AdamOptimizer};
use rand::Rng;

use crate::geometry::Rect;

pub fn nn() {
    const LEARN_RATE: f32 = 0.1;
    const FUTURE_DISCOUNT: f32 = 0.75;
    const RECTS_INIT: [Rect; 4] = [
        Rect {
            x1: 0,
            x2: 2,
            y1: 0,
            y2: 5,
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
        x2: RECTS_INIT.iter().map(Rect::width).max().unwrap() * 3,
        y2: RECTS_INIT.iter().map(Rect::height).max().unwrap() * 3,
    };

    let mut model = Model::new(&[
        bounds.area() as usize + RECTS_INIT.len(),
        128,
        128,
        bounds.area() as usize + RECTS_INIT.len(),
    ]);
    let mut driver = model.driver_mut();

    let variable_count = driver.model().variable_count();
    let mut optimizer = GradientDescentOptimizer {
        a: LEARN_RATE,
    }.instance(variable_count);

    let mut dx = vec![0.0; variable_count];

    let trials = 10_000_000;
    'trials: for trial in 0..trials {
        if trial % 1000 == 0 {
            println!("{}% ({} of {} trials)", (trial * 100) / trials, trial, trials);
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

            // attempt to find a position for the rectangle
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

            actions.push(chosen_idx);
            placed[chosen_idx] = true;
            unplaced_count -= 1;
        }

        let reward = reward(&bounds, &rects);

        if reward > 0.9 {
            let mut q = reward;

            if trial % 1000 == 0 {
                println!();
                println!();
            }

            for chosen_idx in actions.iter().copied().rev() {
                placed[chosen_idx] = false;

                let input = vectorize_input(&bounds, &rects, &placed);
                let target = vectorize_output(&bounds, rects.len(), chosen_idx, &rects[chosen_idx]);

                let result = driver.run_and_record(&input);

                if trial % 1000 == 0 {
                    let state = rects
                        .iter()
                        .zip(placed)
                        .filter(|(_, p)| *p)
                        .map(|(r, _)| r)
                        .collect::<Vec<_>>();
                    let (_, interp) = interpret(&bounds, &rects, result.output());

                    let valid = state.iter().all(|r| !r.overlaps(&interp));

                    println!();
                    println!(" input: {:?}", input);
                    println!("result: {:?}", result.output());
                    println!(" state: {:?}", state);
                    println!("interp: {:?}", interp);
                    println!(" valid: {:?}", valid);
                }

                result.compute_gradients(&target, |idx, val| dx[idx] = q * val);

                optimizer.apply(driver.model_mut().variables_mut().zip(dx.iter()));

                q *= FUTURE_DISCOUNT;
            }
        }
    }

    let mut rects = RECTS_INIT.clone();
    let mut placed = [false; RECTS_INIT.len()];
    for _ in 0..RECTS_INIT.len() {
        let input = vectorize_input(&bounds, &rects, &placed);
        let (idx, rect) = interpret(&bounds, &rects, driver.run(&input).output());
        println!("{:?}", rect);
        rects[idx] = rect;
        placed[idx] = true;
    }
    println!("{:?}", rects);
}

fn vectorize_input(bounds: &Rect, rects: &[Rect], placed: &[bool]) -> Vec<f32> {
    let mut buf = vec![0.0; bounds.area() as usize + rects.len()];

    let placed_rects = rects.iter().zip(placed).filter(|(_, &p)| p).map(|(r, _)| r);

    for rect in placed_rects {
        for x in rect.x1..rect.x2 {
            for y in rect.y1..rect.y2 {
                buf[(x * bounds.height() + y) as usize] = 1.0;
            }
        }
    }

    for (i, &placed) in placed.iter().enumerate() {
        if !placed {
            buf[bounds.area() as usize + i] = 1.0;
        }
    }

    buf
}

fn vectorize_output(
    bounds: &Rect,
    rect_count: usize,
    chosen_index: usize,
    chosen: &Rect,
) -> Vec<f32> {
    let mut buf = vec![0.0; bounds.area() as usize + rect_count];

    buf[(bounds.height() * chosen.x1) as usize + chosen.y1 as usize] = 1.0;
    buf[bounds.area() as usize + chosen_index] = 1.0;

    buf
}

fn interpret(bounds: &Rect, rects: &[Rect], buf: &[f32]) -> (usize, Rect) {
    let (pos, rect) = buf.split_at(bounds.area() as usize);

    let chosen_pos_index = pos
        .iter()
        .enumerate()
        .reduce(|a, b| if a.1 > b.1 { a } else { b })
        .unwrap()
        .0;
    let chosen_x1 = chosen_pos_index as i32 / bounds.height();
    let chosen_y1 = chosen_pos_index as i32 % bounds.height();

    let chosen_rect_index = rect
        .iter()
        .enumerate()
        .reduce(|a, b| if a.1 > b.1 { a } else { b })
        .unwrap()
        .0;

    (chosen_rect_index, Rect {
        x1: chosen_x1,
        y1: chosen_y1,
        x2: chosen_x1 + rects[chosen_rect_index].width(),
        y2: chosen_y1 + rects[chosen_rect_index].height(),
    })
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
