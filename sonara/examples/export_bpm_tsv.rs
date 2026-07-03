use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sonara::analyze;

struct InputRow {
    index: usize,
    path: String,
}

fn clean_tsv_cell(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

fn format_output_row(
    index: usize,
    path: &str,
    bpm_new: Option<f32>,
    duration_sec: Option<f32>,
    n_beats: Option<usize>,
    error: Option<&str>,
) -> String {
    let mut output = String::new();
    output.push_str(&index.to_string());
    output.push('\t');
    output.push_str(&clean_tsv_cell(path));
    output.push('\t');
    if let Some(value) = bpm_new {
        output.push_str(&format!("{value:.6}"));
    }
    output.push('\t');
    if let Some(value) = duration_sec {
        output.push_str(&format!("{value:.3}"));
    }
    output.push('\t');
    if let Some(value) = n_beats {
        output.push_str(&value.to_string());
    }
    output.push('\t');
    if let Some(error) = error {
        output.push_str(&clean_tsv_cell(error));
    }
    output.push('\n');
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: export_bpm_tsv <input.tsv> <output.tsv>");
        std::process::exit(2);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let input = std::fs::read_to_string(input_path)?;
    let rows: Vec<InputRow> = input
        .lines()
        .skip(1)
        .enumerate()
        .filter_map(|(index, line)| {
            let mut parts = line.split('\t');
            let path = parts.next()?.trim();
            if path.is_empty() {
                None
            } else {
                Some(InputRow {
                    index,
                    path: path.to_string(),
                })
            }
        })
        .collect();

    eprintln!("Analyzing {} files in compact mode...", rows.len());
    let started = Instant::now();
    let progress = AtomicUsize::new(0);
    let config = analyze::compact();
    let writer = Arc::new(Mutex::new(BufWriter::new(File::create(output_path)?)));

    {
        let mut writer = writer.lock().expect("output writer lock");
        writer.write_all(b"input_index\tpath\tbpm_sonara_new\tduration_sec\tn_beats\terror\n")?;
        writer.flush()?;
    }

    rows.par_iter()
        .for_each(|row| {
            let result = analyze::analyze_file(Path::new(&row.path), 22050, &config);
            let output = match result {
                Ok(analysis) => format_output_row(
                    row.index,
                    &row.path,
                    Some(analysis.bpm),
                    Some(analysis.duration_sec),
                    Some(analysis.beats.len()),
                    None,
                ),
                Err(err) => {
                    let message = err.to_string();
                    format_output_row(row.index, &row.path, None, None, None, Some(&message))
                }
            };

            {
                let mut writer = writer.lock().expect("output writer lock");
                writer.write_all(output.as_bytes()).expect("write output row");
                writer.flush().expect("flush output row");
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == rows.len() {
                let elapsed = started.elapsed().as_secs_f64();
                eprintln!(
                    "progress\t{}\t{}\t{:.1}s\t{:.2} files/s",
                    done,
                    rows.len(),
                    elapsed,
                    done as f64 / elapsed.max(0.001)
                );
            }
        });

    eprintln!(
        "wrote {}\telapsed {:.1}s",
        output_path,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
