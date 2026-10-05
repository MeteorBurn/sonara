//! `--replay-envelopes <dir>`: rerun the tempo and beat stage of the analysis on
//! envelopes dumped by `--dump-envelopes`, without decoding or the STFT.
//!
//! The call is the analysis pass's own: `beat_track_detailed_with_dp_envelope` with
//! start 120 BPM, tightness 100, trimming, the run's range, the `onset` envelope for
//! the tempo and the `beat` envelope for the tracker; downbeats from the `low` and
//! `mid` envelopes as the beatgrid feature computes them (4/4). On the same build it
//! reproduces the analysis rows' bpm, bpm_raw, candidates, beats and downbeats; the
//! confidence column stays empty (it needs the whole pass).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use ndarray::Array1;
use sonara::{beat, beatgrid};

/// Envelopes of one track: (sample rate, hop length, name -> values).
fn read_envelopes(path: &Path) -> Option<(u32, usize, HashMap<String, Array1<f32>>)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 24 || &bytes[..8] != b"SNRENV01" {
        return None;
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (sr, hop, count, frames) = (word(8), word(12) as usize, word(16) as usize, word(20) as usize);
    let mut at = 24;
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        let len = bytes[at] as usize;
        names.push(String::from_utf8(bytes[at + 1..at + 1 + len].to_vec()).ok()?);
        at += 1 + len;
    }
    let mut envelopes = HashMap::with_capacity(count);
    for name in names {
        let values: Vec<f32> = bytes[at..at + 4 * frames]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        at += 4 * frames;
        envelopes.insert(name, Array1::from(values));
    }
    Some((sr, hop, envelopes))
}

/// One TSV row per idx, in the runner's default layout.
pub fn replay(idxs: &[usize], dir: &Path, bpm_min: Option<f32>, bpm_max: Option<f32>) -> String {
    let rows: Vec<String> = std::thread::scope(|scope| {
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunks: Vec<&[usize]> = idxs.chunks(idxs.len().div_ceil(workers).max(1)).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| scope.spawn(move || chunk.iter().map(|&idx| row(idx, dir, bpm_min, bpm_max)).collect::<Vec<_>>()))
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut tsv = String::from("idx\tstatus\tbpm\tbpm_raw\tconf\tcands\tbeats\tdownbeats\n");
    for r in rows {
        tsv.push_str(&r);
    }
    tsv
}

fn row(idx: usize, dir: &Path, bpm_min: Option<f32>, bpm_max: Option<f32>) -> String {
    let join = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let Some((sr, hop, env)) = read_envelopes(&dir.join(format!("{idx}.env"))) else {
        return format!("{idx}\terror: no envelopes\t\t\t\t\t\t\n");
    };
    let result = beat::beat_track_detailed_with_dp_envelope(
        None,
        Some(env["onset"].view()),
        sr,
        hop,
        120.0,
        100.0,
        true,
        bpm_min,
        bpm_max,
        Some(env["beat"].view()),
    );
    match result {
        Ok((estimate, beats)) => {
            let grid = beatgrid::analyze_grid_low_mid(
                &beats,
                env["low"].view(),
                env["mid"].view(),
                sr,
                hop,
                beatgrid::DEFAULT_BEATS_PER_BAR,
            );
            let cands: Vec<String> = estimate.candidates.iter().map(|(b, s)| format!("{b}:{s}")).collect();
            let mut out = String::new();
            let _ = writeln!(
                out,
                "{idx}\tok\t{}\t{}\t\t{}\t{}\t{}",
                estimate.tempo,
                estimate.tempo_raw,
                cands.join(";"),
                join(&beats),
                join(&grid.downbeats)
            );
            out
        }
        Err(e) => format!("{idx}\terror: {}\t\t\t\t\t\t\n", e.to_string().replace(['\t', '\n'], " ")),
    }
}
