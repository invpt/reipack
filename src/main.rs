use std::{
    fs::File,
    io::{self, prelude::*},
    path::PathBuf,
};

use base64::prelude::*;
use clap::{Parser, Subcommand};
use neun::Model;
use rand::prelude::*;

mod algorithms;
mod geometry;
mod heuristics;
mod nn;
mod traits;

use nn::{evaluate_model, load_weights, store_weights, train_model, TrainingParameters};
pub use traits::*;

use geometry::Rect;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Trains a neural network.
    Train {
        /// Path to a training configuration file (TOML).
        #[arg(short, long)]
        config: PathBuf,

        /// Path to weights to load before training.
        #[arg(short, long)]
        in_weights: Option<PathBuf>,

        /// Path to store weights.
        #[arg(short, long)]
        out_weights: Option<PathBuf>,
    },
    /// Evaluates a neural network.
    Evaluate {
        /// Path to a training configuration file (TOML).
        #[arg(short, long)]
        config: PathBuf,

        /// Path to the weights.
        #[arg(short, long)]
        in_weights: PathBuf,

        /// Number of samples.
        #[arg(short, long)]
        num_samples: usize,
    },
}

fn main() -> Result<(), io::Error> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            config,
            in_weights,
            out_weights,
        } => {
            let mut config_file_content = vec![];
            let mut config_file = File::open(config)?;
            config_file.read_to_end(&mut config_file_content)?;
            let Ok(config_file_content) = std::str::from_utf8(&config_file_content) else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "config isn't valid UTF-8?"))
            };
            let config = match toml::from_str::<TrainingParameters>(config_file_content) {
                Ok(config) => config,
                Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err)),
            };

            let mut dimensions = vec![];
            dimensions.push(config.bounds.area() as usize + 2);
            dimensions.extend(config.hidden_layers.iter().copied());
            dimensions.push(config.bounds.area() as usize);

            let mut model = Model::new(&dimensions);

            if let Some(in_weights) = in_weights {
                load_weights(&mut model, &in_weights)?;
            }

            train_model(&mut model, &config);

            if let Some(out_weights) = out_weights {
                store_weights(&model, &out_weights)?;
            }
        }
        Command::Evaluate {
            config,
            in_weights,
            num_samples,
        } => {
            let mut config_file_content = vec![];
            let mut config_file = File::open(config)?;
            config_file.read_to_end(&mut config_file_content)?;
            let Ok(config_file_content) = std::str::from_utf8(&config_file_content) else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "config isn't valid UTF-8?"))
            };
            let Ok(config) = toml::from_str::<TrainingParameters>(config_file_content) else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "config isn't valid TOML?"))
            };

            let mut dimensions = vec![];
            dimensions.push(config.bounds.area() as usize + 2);
            dimensions.extend(config.hidden_layers.iter().copied());
            dimensions.push(config.bounds.area() as usize);

            let mut model = Model::new(&dimensions);

            load_weights(&mut model, &in_weights)?;

            let samples = std::iter::repeat_with(|| {
                let packing_size =
                    rand::thread_rng().gen_range(config.packing_size_min..=config.packing_size_max);

                std::iter::repeat_with(|| config.rects.choose(&mut rand::thread_rng()).unwrap())
                    .take(packing_size)
            })
            .take(num_samples);

            evaluate_model(&mut model.driver_mut(), &config.bounds, samples);
        }
    }

    Ok(())

    /*const RECTS_INIT: [Rect; 4] = [
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

    let trials = 10000000;
    let mut output =
        std::io::BufWriter::with_capacity(65536, std::fs::File::create("out.csv").unwrap());

    let algorithm = algorithms::packing::random::RandomPackingAlgorithm;

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
                heuristics::spread::spread_score(&rects)
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
                heuristics::closeness::closeness_score(&rects)
                    .to_string()
                    .as_bytes(),
            )
            .unwrap();
        output.write_all(b"\n").unwrap();
    }*/
}

fn serialize_packing(packing: &[Rect]) -> String {
    BASE64_STANDARD.encode(serde_json::to_string(packing).unwrap())
}

fn serialize_order(packing: &[Rect]) -> String {
    BASE64_STANDARD.encode(
        serde_json::to_string(&packing.iter().map(|r| r.size()).collect::<Vec<_>>()).unwrap(),
    )
}
