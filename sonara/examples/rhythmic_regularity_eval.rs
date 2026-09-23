//! Evaluate `rhythmic_regularity` against a labeled corpus.
//!
//! Mirrors `accuracy_eval.rs`: reads a CSV of real files with an expected
//! class, analyzes them in parallel, and reports per-group accuracy plus the
//! component breakdown so a disagreement can be attributed to a specific
//! measure rather than to the score as a whole.
//!
//! ```text
//! cargo run --release --example rhythmic_regularity_eval -- manifest.csv [--sr 22050] [--limit N] [--tsv out.tsv]
//! ```
//!
//! CSV columns (header required): `path,genre,expected` where `expected` is
//! `regular`, `irregular` or `edge`. `edge` rows are reported but excluded
//! from the accuracy figures — they exist to show behaviour, not to be graded.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sonara::analyze;
use sonara::rhythmic_regularity as rr;

struct Args {
    csv: String,
    sr: u32,
    limit: usize,
    tsv: Option<String>,
    bpm_min: Option<f32>,
    bpm_max: Option<f32>,
}

fn parse_args() -> Args {
    let mut args = Args {
        csv: String::new(),
        sr: 22050,
        limit: usize::MAX,
        tsv: None,
        bpm_min: None,
        bpm_max: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--sr" => args.sr = it.next().and_then(|v| v.parse().ok()).unwrap_or(22050),
            "--limit" => args.limit = it.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
            "--tsv" => args.tsv = it.next(),
            "--bpm-min" => args.bpm_min = it.next().and_then(|v| v.parse().ok()),
            "--bpm-max" => args.bpm_max = it.next().and_then(|v| v.parse().ok()),
            "--help" | "-h" => {
                println!("usage: rhythmic_regularity_eval <manifest.csv> [--sr N] [--limit N] [--tsv PATH] [--bpm-min N --bpm-max N]");
                std::process::exit(0);
            }
            other => args.csv = other.to_owned(),
        }
    }
    if args.csv.is_empty() {
        eprintln!("error: a manifest CSV path is required (--help for usage)");
        std::process::exit(2);
    }
    args
}

struct Row {
    path: PathBuf,
    genre: String,
    expected: String,
}

/// Split one CSV line, honouring double-quoted fields (paths contain commas).
fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for ch in line.chars() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut current)),
            _ => current.push(ch),
        }
    }
    fields.push(current);
    fields.into_iter().map(|f| f.trim().to_owned()).collect()
}

fn read_manifest(path: &str, limit: usize) -> Vec<Row> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {path}: {e}");
        std::process::exit(2);
    });
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = split_csv(line);
        if index == 0 && fields.first().map(|f| f.eq_ignore_ascii_case("path")) == Some(true) {
            continue;
        }
        if fields.len() < 3 {
            continue;
        }
        rows.push(Row {
            path: PathBuf::from(&fields[0]),
            genre: fields[1].clone(),
            expected: fields[2].to_lowercase(),
        });
        if rows.len() >= limit {
            break;
        }
    }
    rows
}

#[derive(Default)]
struct Tally {
    total: usize,
    regular: usize,
    irregular: usize,
    abstained: usize,
    failed: usize,
    score_sum: f32,
    confidence_sum: f32,
    scored: usize,
}

impl Tally {
    fn add(&mut self, score: Option<f32>, label: Option<&str>, confidence: Option<f32>) {
        self.total += 1;
        match label {
            Some("regular") => self.regular += 1,
            Some("irregular") => self.irregular += 1,
            _ => self.abstained += 1,
        }
        if let Some(value) = score {
            self.score_sum += value;
            self.scored += 1;
        }
        if let Some(value) = confidence {
            self.confidence_sum += value;
        }
    }
    fn mean_score(&self) -> f32 {
        if self.scored == 0 {
            f32::NAN
        } else {
            self.score_sum / self.scored as f32
        }
    }
    fn mean_confidence(&self) -> f32 {
        if self.total == 0 {
            f32::NAN
        } else {
            self.confidence_sum / self.total as f32
        }
    }
}

fn main() {
    let args = parse_args();
    let rows = read_manifest(&args.csv, args.limit);
    if rows.is_empty() {
        eprintln!("error: manifest produced no rows");
        std::process::exit(2);
    }

    // Co-request the groups the measure reads so the component breakdown can
    // be recomputed from the emitted evidence through the public API alone.
    let config = analyze::AnalysisConfig {
        features: Some(
            ["rhythmic_regularity", "onset_bands", "beatgrid"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        ),
        bpm_min: args.bpm_min,
        bpm_max: args.bpm_max,
        ..Default::default()
    };
    if let (Some(lo), Some(hi)) = (args.bpm_min, args.bpm_max) {
        println!("BPM range alignment: {lo}-{hi}");
    }

    let paths: Vec<&Path> = rows.iter().map(|r| r.path.as_path()).collect();
    let started = std::time::Instant::now();
    let results = analyze::analyze_batch(&paths, args.sr, &config);
    let elapsed = started.elapsed();

    let mut tsv = String::from(
        "path\tgenre\texpected\tscore\tlabel\tconfidence\tconcentration\tcoverage\talignment\tperiodicity\tcompetition\tstability\twindows_valid\twindows_total\n",
    );
    let mut by_expected: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_genre: BTreeMap<String, Tally> = BTreeMap::new();
    let mut failures = 0usize;

    for (row, result) in rows.iter().zip(results.iter()) {
        let Ok(analysis) = result else {
            failures += 1;
            by_expected.entry(row.expected.clone()).or_default().failed += 1;
            by_genre.entry(row.genre.clone()).or_default().failed += 1;
            continue;
        };

        let score = analysis.rhythmic_regularity;
        let label = analysis.rhythmic_regularity_label.as_deref();
        let confidence = analysis.rhythmic_regularity_confidence;
        by_expected
            .entry(row.expected.clone())
            .or_default()
            .add(score, label, confidence);
        by_genre
            .entry(row.genre.clone())
            .or_default()
            .add(score, label, confidence);

        // Recompute the components from the emitted evidence for diagnostics.
        let detail = match (
            analysis.onset_strength_bands.as_ref(),
            analysis.downbeats.as_ref(),
        ) {
            (Some(bands), Some(downbeats)) if !bands.is_empty() => {
                let n_frames = bands[0].len();
                let flat: Vec<f32> = bands.iter().flatten().copied().collect();
                ndarray::Array2::from_shape_vec((bands.len(), n_frames), flat)
                    .ok()
                    .map(|array| {
                        rr::analyze(
                            array.view(),
                            &analysis.beats,
                            downbeats,
                            4,
                            analysis.bpm_confidence,
                            analysis.grid_stability.unwrap_or(0.0),
                            analysis.time_signature_confidence,
                        )
                    })
            }
            _ => None,
        };

        let fmt = |value: Option<f32>| match value {
            Some(v) => format!("{v:.4}"),
            None => "-".to_owned(),
        };
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.path.display(),
            row.genre,
            row.expected,
            fmt(score),
            label.unwrap_or("-"),
            fmt(confidence),
            fmt(detail.as_ref().and_then(|d| d.beat_concentration)),
            fmt(detail.as_ref().and_then(|d| d.beat_coverage)),
            fmt(detail.as_ref().and_then(|d| d.metrical_alignment)),
            fmt(detail.as_ref().and_then(|d| d.metrical_periodicity)),
            fmt(detail.as_ref().and_then(|d| d.pulse_competition)),
            fmt(detail.as_ref().map(|d| d.pattern_stability)),
            detail.as_ref().map(|d| d.windows_valid).unwrap_or(0),
            detail.as_ref().map(|d| d.windows_total).unwrap_or(0),
        ));
    }

    if let Some(path) = args.tsv.as_ref() {
        if let Err(e) = std::fs::write(path, &tsv) {
            eprintln!("warning: could not write {path}: {e}");
        } else {
            println!("per-track TSV written to {path}");
        }
    }

    println!(
        "\n{} files in {:.1}s ({:.2} tracks/s), {} decode failures\n",
        rows.len(),
        elapsed.as_secs_f64(),
        rows.len() as f64 / elapsed.as_secs_f64().max(1e-9),
        failures
    );

    println!("=== by expected class ===");
    println!(
        "{:<12} {:>6} {:>8} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "expected", "n", "regular", "irregular", "abstained", "failed", "mean", "mean conf"
    );
    for (name, tally) in &by_expected {
        println!(
            "{:<12} {:>6} {:>8} {:>10} {:>10} {:>8} {:>9.3} {:>9.3}",
            name,
            tally.total,
            tally.regular,
            tally.irregular,
            tally.abstained,
            tally.failed,
            tally.mean_score(),
            tally.mean_confidence()
        );
    }

    let graded: Vec<(&String, &Tally)> = by_expected
        .iter()
        .filter(|(name, _)| name.as_str() == "regular" || name.as_str() == "irregular")
        .collect();
    let mut correct = 0usize;
    let mut decided = 0usize;
    for (name, tally) in &graded {
        let hit = if name.as_str() == "regular" {
            tally.regular
        } else {
            tally.irregular
        };
        correct += hit;
        decided += tally.regular + tally.irregular;
    }
    if decided > 0 {
        println!(
            "\nagreement on decided tracks: {}/{} = {:.1}%",
            correct,
            decided,
            100.0 * correct as f64 / decided as f64
        );
    }

    println!("\n=== by genre folder ===");
    println!(
        "{:<26} {:>6} {:>8} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "genre", "n", "regular", "irregular", "abstained", "failed", "mean", "mean conf"
    );
    for (name, tally) in &by_genre {
        println!(
            "{:<26} {:>6} {:>8} {:>10} {:>10} {:>8} {:>9.3} {:>9.3}",
            name,
            tally.total,
            tally.regular,
            tally.irregular,
            tally.abstained,
            tally.failed,
            tally.mean_score(),
            tally.mean_confidence()
        );
    }
}
