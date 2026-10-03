//! `--cascade`: the fields that depend on BPM and beats, for a baseline-vs-revision comparison.

use std::fmt::Write as _;

use sonara::analyze::TrackAnalysis;

/// Features whose outputs follow the tempo and the beats.
pub const FEATURES: &[&str] = &[
    "bpm", "beats", "beatgrid", "tempo_curve", "chords", "danceability", "valence", "mood",
    "embedding", "aggression",
];

pub const HEADER: &str = "idx\tstatus\tbpm\tbpm_confidence\ttempo_variability\tgrid_offset_sec\tgrid_stability\t\
danceability\tvalence\tmood_happy\tmood_aggressive\tmood_relaxed\tmood_sad\taggression_score\t\
aggression_rhythm\tdownbeats\tchord_starts\ttempo_curve\tembedding\n";

fn opt(v: Option<f32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

fn list<T: ToString>(v: &[T]) -> String {
    v.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

/// Aggression fields exist only in a build with Sonara's `aggression` feature.
#[cfg(feature = "aggression")]
fn aggression(t: &TrackAnalysis) -> (Option<f32>, Option<f32>) {
    (t.aggression_score, t.aggression_rhythm)
}

#[cfg(not(feature = "aggression"))]
fn aggression(_: &TrackAnalysis) -> (Option<f32>, Option<f32>) {
    (None, None)
}

pub fn row(out: &mut String, idx: usize, t: &TrackAnalysis) {
    let chords: Vec<f32> = t.chord_events.as_deref().unwrap_or(&[]).iter().map(|c| c.start_sec).collect();
    let (aggression_score, aggression_rhythm) = aggression(t);
    let _ = writeln!(
        out,
        "{idx}\tok\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        t.bpm,
        t.bpm_confidence,
        opt(t.tempo_variability),
        opt(t.grid_offset_sec),
        opt(t.grid_stability),
        opt(t.danceability),
        opt(t.valence),
        opt(t.mood_happy),
        opt(t.mood_aggressive),
        opt(t.mood_relaxed),
        opt(t.mood_sad),
        opt(aggression_score),
        opt(aggression_rhythm),
        list(t.downbeats.as_deref().unwrap_or(&[])),
        list(&chords),
        list(t.tempo_curve.as_deref().unwrap_or(&[])),
        list(t.embedding.as_deref().unwrap_or(&[])),
    );
}
