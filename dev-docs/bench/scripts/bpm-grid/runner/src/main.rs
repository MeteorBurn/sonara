//! bpm-grid bench runner: Sonara's own Rust API on real bench tracks, no Python.
//!
//! ```text
//! cargo run --release -- <playlist.txt> <out.tsv> [--idx idx.txt] [--bpm-min 79 --bpm-max 192 | --no-range] [--sr 22050]
//! ```
//! The playlist line number is the 1-based `idx`; `--idx` limits the run to the listed idx
//! (one per line); `--no-range` passes no BPM range; `--cascade` writes the BPM-dependent
//! fields instead (build with `--features aggression`). Output: one TSV row per track with bpm, bpm_raw, confidence, the five
//! candidates, beats and downbeats (frames, hop 512).

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use sonara::analyze;

mod cascade;

fn main() {
    let mut pos = Vec::new();
    let (mut idx_file, mut lo, mut hi, mut sr) = (None, 79.0f32, 192.0f32, 22050u32);
    let (mut cascade_mode, mut no_range) = (false, false);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--idx" => idx_file = it.next(),
            "--bpm-min" => lo = it.next().unwrap().parse().unwrap(),
            "--bpm-max" => hi = it.next().unwrap().parse().unwrap(),
            "--sr" => sr = it.next().unwrap().parse().unwrap(),
            "--cascade" => cascade_mode = true,
            "--no-range" => no_range = true,
            _ => pos.push(a),
        }
    }
    let (playlist, out) = (&pos[0], &pos[1]);
    let wanted: Option<BTreeSet<usize>> = idx_file.map(|f| {
        std::fs::read_to_string(f).unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect()
    });
    let lines: Vec<(usize, String)> = std::fs::read_to_string(playlist)
        .unwrap()
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .filter(|(i, l)| !l.is_empty() && wanted.as_ref().map_or(true, |w| w.contains(i)))
        .collect();
    let paths: Vec<&Path> = lines.iter().map(|(_, l)| Path::new(l.as_str())).collect();
    let config = analyze::AnalysisConfig {
        features: Some(
            if cascade_mode { cascade::FEATURES } else { &["bpm", "beats", "beatgrid"][..] }
                .iter()
                .map(|s| s.to_string())
                .collect(),
        ),
        bpm_min: (!no_range).then_some(lo),
        bpm_max: (!no_range).then_some(hi),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let results = analyze::analyze_batch(&paths, sr, &config);
    let mut tsv = String::from(if cascade_mode {
        cascade::HEADER
    } else {
        "idx\tstatus\tbpm\tbpm_raw\tconf\tcands\tbeats\tdownbeats\n"
    });
    let join = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    for ((idx, _), r) in lines.iter().zip(results.iter()) {
        match r {
            Ok(t) if cascade_mode => cascade::row(&mut tsv, *idx, t),
            Ok(t) => {
                let cands: Vec<String> = t.bpm_candidates.iter().map(|(b, s)| format!("{b}:{s}")).collect();
                let down = t.downbeats.as_deref().map(join).unwrap_or_default();
                let _ = writeln!(tsv, "{idx}\tok\t{}\t{}\t{}\t{}\t{}\t{down}", t.bpm, t.bpm_raw,
                    t.bpm_confidence, cands.join(";"), join(&t.beats));
            }
            Err(e) => {
                let _ = writeln!(tsv, "{idx}\terror: {}\t\t\t\t\t\t", e.to_string().replace(['\t', '\n'], " "));
            }
        }
    }
    std::fs::write(out, tsv).unwrap();
    eprintln!("{} tracks in {:.1}s -> {out}", lines.len(), t0.elapsed().as_secs_f64());
}
