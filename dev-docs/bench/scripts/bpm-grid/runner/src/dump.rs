//! `--dump-envelopes <dir>`: analyse each track on a worker thread and write the
//! rhythm envelopes of its fused pass next to the TSV row (build with
//! `--features bench-internals`).
//!
//! One file per track, `<dir>/<idx>.env`, little-endian:
//! `b"SNRENV01"`, `u32` sample rate, `u32` hop length, `u32` envelope count `k`,
//! `u32` frame count `n`, then `k` names (`u8` length + UTF-8 bytes), then `k × n`
//! `f32` values, one envelope after another.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use sonara::analyze::{self, AnalysisConfig, TrackAnalysis};

/// Hop length of the fused pass (frames of the envelopes).
const HOP_LENGTH: u32 = 512;

pub fn analyze_and_dump(
    tracks: &[(usize, &Path)],
    sr: u32,
    config: &AnalysisConfig,
    dir: &Path,
) -> Vec<sonara::Result<TrackAnalysis>> {
    std::fs::create_dir_all(dir).unwrap();
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<sonara::Result<TrackAnalysis>>>> =
        tracks.iter().map(|_| Mutex::new(None)).collect();
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(idx, path)) = tracks.get(i) else { break };
                let result = analyze::analyze_file_with_rhythm_envelopes(path, sr, config).map(
                    |(analysis, envelopes)| {
                        if let Some(envelopes) = envelopes {
                            let envelopes: Vec<(&str, Vec<f32>)> = envelopes
                                .into_iter()
                                .map(|(name, values)| (name, values.to_vec()))
                                .collect();
                            write_envelopes(&dir.join(format!("{idx}.env")), sr, &envelopes);
                        }
                        analysis
                    },
                );
                *slots[i].lock().unwrap() = Some(result);
            });
        }
    });
    slots.into_iter().map(|slot| slot.into_inner().unwrap().unwrap()).collect()
}

fn write_envelopes(path: &Path, sr: u32, envelopes: &[(&str, Vec<f32>)]) {
    let frames = envelopes.first().map_or(0, |(_, values)| values.len());
    assert!(envelopes.iter().all(|(_, values)| values.len() == frames));
    let mut bytes = Vec::with_capacity(32 + envelopes.len() * (16 + 4 * frames));
    bytes.extend_from_slice(b"SNRENV01");
    for value in [sr, HOP_LENGTH, envelopes.len() as u32, frames as u32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for (name, _) in envelopes {
        bytes.push(name.len() as u8);
        bytes.extend_from_slice(name.as_bytes());
    }
    for (_, values) in envelopes {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}
