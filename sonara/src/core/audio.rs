//! Audio loading, resampling, and time-domain processing.
//!
//! Audio I/O, resampling, streaming, and time-domain processing.
//! Includes load, stream, to_mono, resample, get_duration, get_samplerate,
//! autocorrelate, lpc, zero_crossings, clicks, tone, chirp, mu_compress, mu_expand.

use std::f32::consts::PI;
use std::path::Path;

#[cfg(test)]
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::error::{Result, SonaraError};
use crate::types::{AudioBuffer, Float};

// ============================================================
// Audio I/O
// ============================================================

/// Container/stream metadata tags (ID3v2, Vorbis comments, etc.) read straight
/// from an audio file.
///
/// Populated only by `analyze_file`/`analyze_batch` when the `"tags"` feature is
/// requested (`features=["tags"]`); always `None` otherwise and for
/// `analyze_signal` (a bare signal has no container to read). Each field is
/// `Some` only when the file actually carries that tag.
///
/// Tags are read from **symphonia-decoded** containers (FLAC/Vorbis comments,
/// MP3/AAC ID3v2, MP4, etc.). WAV tries the hound fast path first and falls
/// back to Symphonia 0.6 only when hound rejects the file or its samples.
/// Neither WAV path exposes tags, so a `.wav` input always yields `None` here.
///
/// Note: `genre` here is the *file's* metadata genre string (e.g. "Electronic"),
/// which is distinct from `TrackAnalysis::genre` — the latter is a reserved
/// placeholder for a future computed/ML genre and is unrelated to this tag.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TrackTags {
    /// Track title (`TrackTitle`).
    pub title: Option<String>,
    /// Track artist (`Artist`).
    pub artist: Option<String>,
    /// Album name (`Album`).
    pub album: Option<String>,
    /// Genre string as stored in the file (`Genre`).
    pub genre: Option<String>,
    /// Release year of *this* file/edition, derived from the leading 4 digits of
    /// the first parseable `Date`/`ReleaseDate` tag (there is no dedicated year
    /// tag). On a reissue or compilation this is the reissue date — see
    /// [`original_year`](Self::original_year) for the original release year.
    pub year: Option<u32>,
    /// Original release year, from the original-release-date tags: ID3v2.4
    /// `TDOR`, ID3v2.3 `TORY`, `TXXX:originalyear`-style frames, or Vorbis
    /// `ORIGINALDATE`/`ORIGINALYEAR` (parsed to its leading 4 digits). `None`
    /// when the file carries no such tag — there is no fallback to `year`.
    ///
    /// Consumers doing era reasoning should prefer `original_year` over `year`
    /// when it is present: on reissues/compilations `year` is the reissue date
    /// while `original_year` is the true original release year.
    pub original_year: Option<u32>,
    /// Track number (`TrackNumber`); the leading integer of values like
    /// `"3"` or `"3/12"`.
    pub track_no: Option<u32>,
}

/// Load an audio file as a floating-point time series.
///
/// Audio is automatically resampled to the given rate (default 22050 Hz).
/// Use `sr=None` (pass 0) to preserve the native sample rate.
///
/// - `path`: Path to audio file (WAV, FLAC, OGG, MP3, etc.)
/// - `sr`: Target sample rate. 0 means use native.
/// - `mono`: Convert to mono if true.
/// - `offset`: Start reading after this many seconds.
/// - `duration`: Only read this many seconds (0 = all).
pub fn load(
    path: &Path,
    sr: u32,
    mono: bool,
    offset: Float,
    duration: Float,
) -> Result<(Array1<Float>, u32)> {
    let (y, sr, _tags) = load_with_tags(path, sr, mono, offset, duration, false)?;
    Ok((y, sr))
}

/// Like [`load`], but optionally also extracts container/stream metadata tags.
///
/// When `want_tags` is `false` this does exactly the same work as [`load`] and
/// returns `None` for the tags — the default fast path pays nothing. When
/// `true`, tags are read from symphonia-decoded containers (see [`TrackTags`]);
/// WAV files try hound first, retry through Symphonia 0.6 on failure, and
/// always yield `None`.
pub fn load_with_tags(
    path: &Path,
    sr: u32,
    mono: bool,
    offset: Float,
    duration: Float,
    want_tags: bool,
) -> Result<(Array1<Float>, u32, Option<TrackTags>)> {
    // Try hound first for WAV files (fastest path). WAV carries no tags here.
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let (samples, native_sr, n_channels, tags) = if ext.eq_ignore_ascii_case("wav") {
        match load_wav(path) {
            Ok((s, r, c)) => (s, r, c, None),
            Err(hound_err) => match load_symphonia(path, false) {
                Ok((s, r, c, _)) => (s, r, c, None),
                Err(symphonia_err) => {
                    return Err(combine_wav_errors(path, hound_err, symphonia_err));
                }
            },
        }
    } else {
        load_symphonia(path, want_tags)?
    };

    let native_sr = native_sr;

    // Apply offset and duration
    let offset_samples = (offset * native_sr as Float) as usize;
    let total_samples = samples.len() / n_channels;

    let end_sample = if duration > 0.0 {
        let dur_samples = (duration * native_sr as Float) as usize;
        (offset_samples + dur_samples).min(total_samples)
    } else {
        total_samples
    };

    if offset_samples >= total_samples {
        return Err(SonaraError::InvalidParameter {
            param: "offset",
            reason: format!(
                "offset {offset}s exceeds file duration {}s",
                total_samples as Float / native_sr as Float
            ),
        });
    }

    // Deinterleave if multi-channel
    let mut audio = if n_channels == 1 {
        Array1::from_vec(samples[offset_samples..end_sample].to_vec())
    } else {
        let n_samples = end_sample - offset_samples;
        let mut channel_data = Array2::<Float>::zeros((n_channels, n_samples));
        for i in 0..n_samples {
            for ch in 0..n_channels {
                channel_data[(ch, i)] = samples[(offset_samples + i) * n_channels + ch];
            }
        }
        if mono {
            to_mono(channel_data.view())
        } else {
            // Return first channel for now (multi-channel support in later phases)
            channel_data.row(0).to_owned()
        }
    };

    // Resample if needed
    let target_sr = if sr == 0 { native_sr } else { sr };
    if target_sr != native_sr {
        audio = resample(audio.view(), native_sr, target_sr)?;
    }

    Ok((audio, target_sr, tags))
}

/// Load WAV file using hound (fast path).
fn load_wav(path: &Path) -> Result<(Vec<Float>, u32, usize)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| SonaraError::AudioFile(format!("{}: {}", path.display(), e)))?;

    let spec = reader.spec();
    let sr = spec.sample_rate;
    let n_channels = spec.channels as usize;

    let samples: Vec<Float> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map(|v| v as Float))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| SonaraError::Decode(e.to_string()))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let scale = 1.0 / (1u64 << (bits - 1)) as Float;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as Float * scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| SonaraError::Decode(e.to_string()))?
        }
    };

    Ok((samples, sr, n_channels))
}

/// Preserve the public error category when both WAV backends fail for the same
/// reason while retaining both diagnostics. In particular, a missing or
/// unreadable file remains an I/O error instead of being reported as corrupt
/// audio merely because the fallback was attempted.
fn combine_wav_errors(
    path: &Path,
    hound_err: SonaraError,
    symphonia_err: SonaraError,
) -> SonaraError {
    let message = format!(
        "{}: WAV decode failed with Hound ({hound_err}); \
         Symphonia 0.6 fallback failed ({symphonia_err})",
        path.display(),
    );
    match (&hound_err, &symphonia_err) {
        (SonaraError::AudioFile(_), SonaraError::AudioFile(_)) => SonaraError::AudioFile(message),
        (SonaraError::UnsupportedFormat(_), SonaraError::UnsupportedFormat(_)) => {
            SonaraError::UnsupportedFormat(message)
        }
        _ => SonaraError::Decode(message),
    }
}

/// Map a symphonia error into the most appropriate `SonaraError` variant,
/// annotating it with the file path, a stage label, and (when known) the
/// container/codec that was in play so batch callers get an actionable message.
fn map_symphonia_err(
    path: &Path,
    stage: &str,
    codec: Option<&str>,
    err: symphonia::core::errors::Error,
) -> SonaraError {
    use symphonia::core::errors::Error as SymErr;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("?");
    let codec = codec.unwrap_or("unknown");
    let ctx = format!(
        "{} (container='{}', codec='{}', stage={}): {}",
        path.display(),
        ext,
        codec,
        stage,
        err
    );
    match err {
        // The probe/codec registry does not recognize this container or codec.
        SymErr::Unsupported(_) => SonaraError::UnsupportedFormat(ctx),
        // Underlying filesystem/stream I/O problem (not a clean EOF).
        SymErr::IoError(_) => SonaraError::AudioFile(ctx),
        // Malformed bitstream, reset required, limit exceeded, seek error → decode.
        _ => SonaraError::Decode(ctx),
    }
}

/// Human-readable short name for a symphonia codec type, if registered.
fn codec_short_name(codec: symphonia::core::codecs::audio::AudioCodecId) -> Option<&'static str> {
    symphonia::default::get_codecs()
        .get_audio_decoder(codec)
        .map(|d| d.codec.info.short_name)
}

/// Probe enabled container formats without registering Symphonia metadata readers.
///
/// This prevents leading and trailing tag/visual probing when callers did not
/// request tags, while retaining every format reader enabled for SONARA.
fn probe_symphonia_without_metadata<'s>(
    hint: &symphonia::core::formats::probe::Hint,
    mss: symphonia::core::io::MediaSourceStream<'s>,
) -> symphonia::core::errors::Result<Box<dyn symphonia::core::formats::FormatReader + 's>> {
    let mut probe = symphonia::core::formats::probe::Probe::new();
    // `register_enabled_formats` also registers the optional APE and ID3
    // metadata readers. Register SONARA's enabled container readers directly
    // so this branch performs no separate metadata probing.
    probe.register_format::<symphonia::default::formats::AdtsReader<'_>>();
    probe.register_format::<symphonia::default::formats::CafReader<'_>>();
    probe.register_format::<symphonia::default::formats::FlacReader<'_>>();
    probe.register_format::<symphonia::default::formats::IsoMp4Reader<'_>>();
    probe.register_format::<symphonia::default::formats::MpaReader<'_>>();
    probe.register_format::<symphonia::default::formats::AiffReader<'_>>();
    probe.register_format::<symphonia::default::formats::WavReader<'_>>();
    probe.register_format::<symphonia::default::formats::OggReader<'_>>();
    probe.register_format::<symphonia::default::formats::MkvReader<'_>>();
    probe.probe(
        hint,
        mss,
        symphonia::core::formats::FormatOptions::default(),
        symphonia::core::meta::MetadataOptions::default(),
    )
}

/// Load audio file using symphonia (supports mp3, flac, ogg, etc.).
///
/// When `want_tags` is `true`, container/stream metadata tags are also collected
/// (see [`TrackTags`]); when `false`, Symphonia metadata readers are not probed.
fn load_symphonia(
    path: &Path,
    want_tags: bool,
) -> Result<(Vec<Float>, u32, usize, Option<TrackTags>)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path)
        .map_err(|e| SonaraError::AudioFile(format!("{}: {}", path.display(), e)))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probe_result = if want_tags {
        symphonia::default::get_probe().probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
    } else {
        probe_symphonia_without_metadata(&hint, mss)
    };
    let mut format: Box<dyn FormatReader> = match probe_result {
        Ok(format) => format,
        Err(probe_err)
            if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3")) =>
        {
            // Generic probing stops at the first recognized marker. That can
            // make valid MP3 audio unavailable when an ID3 frame is malformed,
            // or when junk before the first MPEG frame resembles ADTS. Retry
            // the extension-confirmed stream with the MP3 reader, which
            // resynchronizes to consecutive MPEG frames and bypasses metadata.
            let file = std::fs::File::open(path)
                .map_err(|e| SonaraError::AudioFile(format!("{}: {}", path.display(), e)))?;
            let mss = MediaSourceStream::new(Box::new(file), Default::default());
            match symphonia::default::formats::MpaReader::try_new(mss, FormatOptions::default()) {
                Ok(reader) => Box::new(reader),
                // Keep the original probe diagnostic when explicit MP3
                // recovery also rejects the stream.
                Err(_) => return Err(map_symphonia_err(path, "probe", None, probe_err)),
            }
        }
        Err(e) => return Err(map_symphonia_err(path, "probe", None, e)),
    };

    let (track_id, codec_params, estimated_frames) = {
        let track = format.default_track(TrackType::Audio).ok_or_else(|| {
            SonaraError::Decode(format!("{}: no audio track found", path.display()))
        })?;
        let codec_params = track
            .codec_params
            .as_ref()
            .and_then(|params| params.audio())
            .cloned()
            .ok_or_else(|| {
                SonaraError::Decode(format!(
                    "{}: missing audio codec parameters",
                    path.display()
                ))
            })?;
        (track.id, codec_params, track.num_frames)
    };

    let codec_name = codec_short_name(codec_params.codec);
    let sr = codec_params.sample_rate.ok_or_else(|| {
        SonaraError::Decode(format!(
            "{} (codec='{}'): missing sample rate in stream header",
            path.display(),
            codec_name.unwrap_or("unknown"),
        ))
    })?;
    let n_channels = codec_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(1);

    // In Symphonia 0.6, metadata discovered while probing is queued directly
    // on the format reader. Merge all currently available revisions in order,
    // including track-specific metadata, so the first value per field wins.
    let tags = if want_tags {
        let mut t = TrackTags::default();
        merge_format_metadata(&mut *format, track_id, &mut t);
        Some(t)
    } else {
        None
    };

    let decoder_options = AudioDecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&codec_params, &decoder_options)
        .map_err(|e| map_symphonia_err(path, "codec-init", codec_name, e))?;

    // Pre-allocate output with estimated capacity (avoid repeated growth)
    let estimated_samples = estimated_frames
        .map(|f| f as usize * n_channels)
        .unwrap_or(sr as usize * 300 * n_channels); // fallback: 5 min
    let mut samples = Vec::with_capacity(estimated_samples);

    // Symphonia 0.6 converts generic decoded buffers directly into an
    // interleaved vector. Reuse the packet vector across the decode loop.
    let mut packet_samples = Vec::<f32>::new();

    // Packet-level decode tolerance: symphonia classifies bitstream-local
    // defects (e.g. mpa "invalid main_data offset" or huffman overruns in
    // MP3s with a damaged bit reservoir) as recoverable `DecodeError` — the
    // decoder remains usable and subsequent packets can decode normally.
    // Such packets are skipped rather than failing the whole file; every
    // other error stays fatal. Guardrails after the loop ensure a stream
    // where nothing decoded still surfaces its decode error.
    let mut skipped_packets: usize = 0;
    let mut first_decode_err: Option<SonaraError> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(map_symphonia_err(path, "demux", codec_name, e)),
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e @ symphonia::core::errors::Error::DecodeError(_)) => {
                skipped_packets += 1;
                if first_decode_err.is_none() {
                    first_decode_err = Some(map_symphonia_err(path, "decode", codec_name, e));
                }
                continue;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => {
                // The decoder must be rebuilt (e.g. stream parameters changed
                // mid-file); the triggering packet is lost.
                skipped_packets += 1;
                decoder = symphonia::default::get_codecs()
                    .make_audio_decoder(&codec_params, &decoder_options)
                    .map_err(|e| map_symphonia_err(path, "codec-init", codec_name, e))?;
                continue;
            }
            Err(e) => return Err(map_symphonia_err(path, "decode", codec_name, e)),
        };

        decoded.copy_to_vec_interleaved(&mut packet_samples);
        samples.extend_from_slice(&packet_samples);
    }

    // Guardrails for the tolerant path (zero cost when nothing was skipped):
    // the recovered stream must still be genuine audio — non-empty, finite PCM.
    if skipped_packets > 0 {
        let ctx = |msg: String| {
            SonaraError::Decode(format!(
                "{} (container='{}', codec='{}', stage=decode): {}",
                path.display(),
                path.extension().and_then(|e| e.to_str()).unwrap_or("?"),
                codec_name.unwrap_or("unknown"),
                msg,
            ))
        };
        if samples.is_empty() {
            return Err(first_decode_err.unwrap_or_else(|| {
                ctx(format!("all {skipped_packets} packets failed to decode"))
            }));
        }
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(ctx(format!(
                "non-finite samples after skipping {skipped_packets} damaged packets"
            )));
        }
    }

    Ok((samples, sr, n_channels, tags))
}

/// Merge all metadata revisions currently queued on a format reader.
fn merge_format_metadata(
    format: &mut dyn symphonia::core::formats::FormatReader,
    track_id: u32,
    tags: &mut TrackTags,
) {
    let mut metadata = format.metadata();
    let mut revisions = Vec::new();
    loop {
        let mut revision_tags = TrackTags::default();
        {
            let Some(revision) = metadata.current() else {
                break;
            };
            merge_tags(&mut revision_tags, &revision.media.tags);
            for per_track in &revision.per_track {
                if per_track.track_id == u64::from(track_id) {
                    merge_tags(&mut revision_tags, &per_track.metadata.tags);
                }
            }
        }
        revisions.push(revision_tags);
        if metadata.pop().is_none() {
            break;
        }
    }

    // Symphonia queues revisions oldest-first. Prefer the newest revision
    // (for example, leading ID3v2 over trailing ID3v1), then let older
    // revisions fill only fields the newer metadata did not provide.
    for revision_tags in revisions.into_iter().rev() {
        merge_track_tags(tags, revision_tags);
    }
}

/// Fill still-empty fields from one already-normalized metadata revision.
fn merge_track_tags(target: &mut TrackTags, source: TrackTags) {
    if target.title.is_none() {
        target.title = source.title;
    }
    if target.artist.is_none() {
        target.artist = source.artist;
    }
    if target.album.is_none() {
        target.album = source.album;
    }
    if target.genre.is_none() {
        target.genre = source.genre;
    }
    if target.year.is_none() {
        target.year = source.year;
    }
    if target.original_year.is_none() {
        target.original_year = source.original_year;
    }
    if target.track_no.is_none() {
        target.track_no = source.track_no;
    }
}

/// Fill any still-empty [`TrackTags`] fields from a slice of symphonia `Tag`s.
///
/// First value per field wins (so calling this on the probe metadata first,
/// then container metadata, prefers the probe source). `year` is derived from
/// the leading 4 digits of the first parseable `Date`/`ReleaseDate` (the
/// file/edition date); `original_year` from `OriginalDate` or a raw
/// original-release-date key (see [`is_original_date_key`]); `track_no` from
/// the leading integer of `TrackNumber`. The `year`/`original_year` split is
/// deliberate: reissues carry the reissue date in `Date` and the true original
/// date in the original-release-date tags.
fn merge_tags(t: &mut TrackTags, tags: &[symphonia::core::meta::Tag]) {
    use symphonia::core::meta::StandardTag;

    for tag in tags {
        // Typed standard-tag mapping (preferred when Symphonia recognized it).
        if let Some(std_tag) = tag.std.as_ref() {
            match std_tag {
                StandardTag::TrackTitle(value) => set_str(&mut t.title, value),
                StandardTag::Artist(value) => set_str(&mut t.artist, value),
                StandardTag::Album(value) => set_str(&mut t.album, value),
                StandardTag::Genre(value) => set_str(&mut t.genre, value),
                StandardTag::RecordingDate(value) | StandardTag::ReleaseDate(value) => {
                    set_year(&mut t.year, value)
                }
                StandardTag::RecordingYear(value) | StandardTag::ReleaseYear(value) => {
                    if t.year.is_none() {
                        t.year = Some(u32::from(*value));
                    }
                }
                StandardTag::OriginalRecordingDate(value)
                | StandardTag::OriginalReleaseDate(value) => set_year(&mut t.original_year, value),
                StandardTag::OriginalRecordingYear(value)
                | StandardTag::OriginalReleaseYear(value) => {
                    if t.original_year.is_none() {
                        t.original_year = Some(u32::from(*value));
                    }
                }
                StandardTag::TrackNumber(value) => {
                    if t.track_no.is_none() {
                        t.track_no = u32::try_from(*value).ok();
                    }
                }
                _ => {}
            }
        }

        // Raw-key fallback for original-release-date tags symphonia may not
        // standardize (e.g. `TXXX:originalyear`, `ORIGINALYEAR`, a legacy `TORY`).
        if t.original_year.is_none() && is_original_date_key(&tag.raw.key) {
            set_year(&mut t.original_year, &tag.raw.value.to_string());
        }
    }
}

/// True if a raw tag key denotes an original-release-date field that symphonia
/// may not map to `StandardTagKey::OriginalDate`. Tolerant of an optional
/// `TXXX:` frame prefix and of case: matches `originalyear`, `originaldate`,
/// `tory`, and `tdor`.
fn is_original_date_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    let core = lower.strip_prefix("txxx:").unwrap_or(&lower);
    matches!(core, "originalyear" | "originaldate" | "tory" | "tdor")
}

/// Set `slot` to the parsed leading-4-digit year of the tag value if not
/// already set.
fn set_year(slot: &mut Option<u32>, value: &str) {
    if slot.is_none() {
        if let Some(y) = parse_year(value) {
            *slot = Some(y);
        }
    }
}

/// Set `slot` to the tag's string value if not already set and non-empty.
fn set_str(slot: &mut Option<String>, value: &str) {
    if slot.is_none() {
        let s = value.trim();
        if !s.is_empty() {
            *slot = Some(s.to_string());
        }
    }
}

/// Parse the leading 4-digit year out of a date string (e.g. "2024",
/// "2024-05-01", "2024/05").
fn parse_year(s: &str) -> Option<u32> {
    let digits: String = s
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.len() == 4 {
        digits.parse().ok()
    } else {
        None
    }
}

/// Parse the leading integer of a track-number string (e.g. "3" or "3/12").
#[cfg(test)]
fn parse_leading_u32(s: &str) -> Option<u32> {
    let digits: String = s
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

/// Convert a multi-channel signal to mono by averaging channels.
pub fn to_mono(y: ndarray::ArrayView2<Float>) -> Array1<Float> {
    y.mean_axis(Axis(0)).unwrap()
}

/// Resample a signal from `orig_sr` to `target_sr`.
///
/// Fast path for exact 2:1 decimation (e.g., 44100→22050) using a half-band
/// FIR filter — ~20x faster than full sinc resampling for this common case.
/// Falls back to rubato sinc interpolation for all other ratios.
pub fn resample(y: ArrayView1<Float>, orig_sr: u32, target_sr: u32) -> Result<Array1<Float>> {
    if orig_sr == target_sr {
        return Ok(y.to_owned());
    }

    // Fast path: exact 2:1 decimation (e.g., 44100 → 22050)
    if orig_sr == 2 * target_sr {
        return Ok(decimate_half(y));
    }

    // General path: rubato sinc interpolation
    resample_rubato(y, orig_sr, target_sr)
}

/// 2:1 decimation using a 31-tap half-band FIR low-pass filter.
///
/// Half-band filters have the property that every other coefficient (at even
/// offsets from center) is zero, so only ~8 multiply-accumulate operations
/// per output sample. This is dramatically faster than general sinc resampling.
///
/// Filter: Hamming-windowed sinc, normalized to unity DC gain.
/// Cutoff at Fs/4 (the new Nyquist after 2:1 decimation).
fn decimate_half(y: ArrayView1<Float>) -> Array1<Float> {
    // 31-tap half-band FIR: h[center] = 0.5, non-zero at odd offsets ±1,±3,...,±15
    // Coefficients: (1/2)*sinc(k/2) * hamming_window, then all normalized to sum=1.0
    //
    // Ideal sinc(k/2) at odd offsets from center:
    //   k=1:  2/π    k=3: -2/(3π)   k=5:  2/(5π)   k=7: -2/(7π)
    //   k=9:  2/(9π) k=11:-2/(11π)  k=13: 2/(13π)  k=15:-2/(15π)
    // Multiplied by 0.5 and Hamming window w[15±k] = 0.54 - 0.46*cos(2π(15±k)/30)
    // Then all 17 non-zero coefficients normalized so their sum = 1.0.

    // Pre-computed normalized coefficients for offsets ±k from center (k=1,3,5,...,15)
    // These sum (with center=0.5 before normalization) to 1.0 after scaling.
    const N_TAPS: usize = 8; // number of (symmetric) non-zero tap pairs
    const CENTER_COEFF: Float = 0.5;

    // Ideal half-sinc values * Hamming window, before normalization
    // h_raw[j] for offset k = 2*j+1 from center
    let raw_coeffs: [Float; N_TAPS] = {
        let pi = std::f32::consts::PI;
        [
            0.5 * (pi * 0.5).recip() * (0.54 - 0.46 * (2.0 * pi * 14.0 / 30.0).cos()), // k=1
            0.5 * -(pi * 1.5).recip() * (0.54 - 0.46 * (2.0 * pi * 12.0 / 30.0).cos()), // k=3
            0.5 * (pi * 2.5).recip() * (0.54 - 0.46 * (2.0 * pi * 10.0 / 30.0).cos()), // k=5
            0.5 * -(pi * 3.5).recip() * (0.54 - 0.46 * (2.0 * pi * 8.0 / 30.0).cos()), // k=7
            0.5 * (pi * 4.5).recip() * (0.54 - 0.46 * (2.0 * pi * 6.0 / 30.0).cos()),  // k=9
            0.5 * -(pi * 5.5).recip() * (0.54 - 0.46 * (2.0 * pi * 4.0 / 30.0).cos()), // k=11
            0.5 * (pi * 6.5).recip() * (0.54 - 0.46 * (2.0 * pi * 2.0 / 30.0).cos()),  // k=13
            0.5 * -(pi * 7.5).recip() * (0.54 - 0.46 * (2.0 * pi * 0.0 / 30.0).cos()), // k=15
        ]
    };

    // Normalize: sum of all coefficients should be 1.0 for unity DC gain
    // Pre-compute scaled coefficients (avoid per-sample division)
    let raw_sum = CENTER_COEFF + 2.0 * raw_coeffs.iter().sum::<Float>();
    let scale = 1.0 / raw_sum;
    let center = CENTER_COEFF * scale;
    let coeffs: [Float; N_TAPS] = std::array::from_fn(|j| raw_coeffs[j] * scale);

    let n = y.len();
    let n_out = n / 2;
    let mut out = Array1::<Float>::zeros(n_out);
    let raw = y.as_slice().unwrap();

    // Interior samples (no boundary checks needed)
    let safe_start = (N_TAPS * 2) / 2 + 1; // first output where all taps are in-bounds
    let safe_end = n_out.saturating_sub(N_TAPS); // last safe output

    // Boundary samples (with bounds checking)
    for i in 0..safe_start.min(n_out) {
        let c = i * 2;
        let mut sum = center * raw[c];
        for (j, &coeff) in coeffs.iter().enumerate() {
            let k = (2 * j + 1) as isize;
            let il = c as isize - k;
            let ir = c as isize + k;
            let left = if il >= 0 { raw[il as usize] } else { 0.0 };
            let right = if (ir as usize) < n {
                raw[ir as usize]
            } else {
                0.0
            };
            sum += coeff * (left + right);
        }
        out[i] = sum;
    }

    // Interior samples (hot loop, no bounds checks)
    for i in safe_start..safe_end {
        let c = i * 2;
        let mut sum = center * raw[c];
        for (j, &coeff) in coeffs.iter().enumerate() {
            let k = 2 * j + 1;
            sum += coeff * (raw[c - k] + raw[c + k]);
        }
        out[i] = sum;
    }

    // Trailing boundary samples
    for i in safe_end.max(safe_start)..n_out {
        let c = i * 2;
        let mut sum = center * raw[c];
        for (j, &coeff) in coeffs.iter().enumerate() {
            let k = (2 * j + 1) as isize;
            let il = c as isize - k;
            let ir = c as isize + k;
            let left = if il >= 0 { raw[il as usize] } else { 0.0 };
            let right = if (ir as usize) < n {
                raw[ir as usize]
            } else {
                0.0
            };
            sum += coeff * (left + right);
        }
        out[i] = sum;
    }

    out
}

/// General resampling using rubato sinc interpolation.
fn resample_rubato(y: ArrayView1<Float>, orig_sr: u32, target_sr: u32) -> Result<Array1<Float>> {
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{Fft, FixedSync, Resampler};

    let chunk_size = 1024;
    let mut resampler = Fft::<f32>::new(
        orig_sr as usize,
        target_sr as usize,
        chunk_size,
        1, // sub_chunks
        1, // mono
        FixedSync::Input,
    )
    .map_err(|e| SonaraError::Fft(format!("resampler init: {e}")))?;

    let input_len = y.len();
    let input_vec: Vec<Float> = y.to_vec();
    let input_data = vec![input_vec];
    let input = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .map_err(|e| SonaraError::Fft(format!("resampler input buffer: {e}")))?;

    let output_len = resampler.process_all_needed_output_len(input_len);
    let mut output_data = vec![vec![0.0f32; output_len]];
    let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .map_err(|e| SonaraError::Fft(format!("resampler output buffer: {e}")))?;

    let (_nbr_in, nbr_out) = resampler
        .process_all_into_buffer(&input, &mut output, input_len, None)
        .map_err(|e| SonaraError::Fft(format!("resample: {e}")))?;

    output_data[0].truncate(nbr_out);
    Ok(Array1::from_vec(output_data.into_iter().next().unwrap()))
}

/// Get the duration of an audio file in seconds.
pub fn get_duration(path: &Path) -> Result<Float> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("wav") {
        let reader = hound::WavReader::open(path)
            .map_err(|e| SonaraError::AudioFile(format!("{}: {}", path.display(), e)))?;
        let spec = reader.spec();
        let n_samples = reader.len() as Float / spec.channels as Float;
        Ok(n_samples / spec.sample_rate as Float)
    } else {
        // Fallback: load and measure (not ideal for large files)
        let (samples, sr, _, _) = load_symphonia(path, false)?;
        Ok(samples.len() as Float / sr as Float)
    }
}

/// Get the native sample rate of an audio file.
pub fn get_samplerate(path: &Path) -> Result<u32> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("wav") {
        let reader = hound::WavReader::open(path)
            .map_err(|e| SonaraError::AudioFile(format!("{}: {}", path.display(), e)))?;
        Ok(reader.spec().sample_rate)
    } else {
        let (_, sr, _, _) = load_symphonia(path, false)?;
        Ok(sr)
    }
}

// ============================================================
// Time-domain processing
// ============================================================

/// Stream audio from a file in fixed-length blocks.
///
/// Returns a Vec of audio blocks. Each block is an AudioBuffer of `block_length` samples.
/// For true streaming, the caller should process blocks as they arrive.
pub fn stream(
    path: &Path,
    block_length: usize,
    frame_length: usize,
    hop_length: usize,
    sr: u32,
    mono: bool,
) -> Result<Vec<AudioBuffer>> {
    let (y, _) = load(path, sr, mono, 0.0, 0.0)?;
    let step = block_length * hop_length;
    let mut blocks = Vec::new();
    let mut pos = 0;
    while pos < y.len() {
        let end = (pos + step + frame_length).min(y.len());
        blocks.push(y.slice(ndarray::s![pos..end]).to_owned());
        pos += step;
    }
    Ok(blocks)
}

/// Stateful stream resampler for chunk-by-chunk processing.
///
/// Wraps rubato's FFT resampler with internal buffering so that input
/// chunks of arbitrary size can be processed incrementally.
pub struct StreamResampler {
    resampler: rubato::Fft<f32>,
    buffer: Vec<f32>,
    #[allow(dead_code)]
    orig_sr: u32,
    #[allow(dead_code)]
    target_sr: u32,
}

impl StreamResampler {
    /// Create a new stream resampler.
    ///
    /// - `orig_sr` / `target_sr`: source and target sample rates
    /// - `chunk_size`: internal processing chunk size (default 1024)
    pub fn new(orig_sr: u32, target_sr: u32, chunk_size: usize) -> Result<Self> {
        use rubato::{Fft, FixedSync};

        if orig_sr == target_sr {
            return Err(SonaraError::InvalidParameter {
                param: "target_sr",
                reason: "target_sr must differ from orig_sr".into(),
            });
        }

        let resampler = Fft::<f32>::new(
            orig_sr as usize,
            target_sr as usize,
            chunk_size,
            1,
            1,
            FixedSync::Input,
        )
        .map_err(|e| SonaraError::Fft(format!("stream resampler init: {e}")))?;

        Ok(Self {
            resampler,
            buffer: Vec::new(),
            orig_sr,
            target_sr,
        })
    }

    /// Process a chunk of input samples, returning resampled output.
    ///
    /// Internally buffers samples until enough are available for the
    /// resampler's required input size. May return an empty Vec if
    /// not enough samples have been accumulated yet.
    pub fn process_chunk(&mut self, chunk: &[Float]) -> Result<Vec<Float>> {
        use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
        use rubato::Resampler;

        self.buffer.extend_from_slice(chunk);

        let needed = self.resampler.input_frames_next();
        if self.buffer.len() < needed {
            return Ok(Vec::new());
        }

        let mut output = Vec::new();

        while self.buffer.len() >= needed {
            let input_data = vec![self.buffer[..needed].to_vec()];
            let input = SequentialSliceOfVecs::new(&input_data, 1, needed)
                .map_err(|e| SonaraError::Fft(format!("stream resample input: {e}")))?;

            let out_len = self.resampler.output_frames_next();
            let mut output_data = vec![vec![0.0f32; out_len]];
            let mut out_buf = SequentialSliceOfVecs::new_mut(&mut output_data, 1, out_len)
                .map_err(|e| SonaraError::Fft(format!("stream resample output: {e}")))?;

            let (_n_in, n_out) = self
                .resampler
                .process_into_buffer(&input, &mut out_buf, None)
                .map_err(|e| SonaraError::Fft(format!("stream resample: {e}")))?;

            output.extend_from_slice(&output_data[0][..n_out]);
            self.buffer.drain(..needed);
        }

        Ok(output)
    }

    /// Flush remaining samples through the resampler.
    ///
    /// Call this after the last chunk to drain the internal buffers.
    pub fn flush(&mut self) -> Result<Vec<Float>> {
        use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
        use rubato::Resampler;

        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Pad to the required input size
        let needed = self.resampler.input_frames_next();
        self.buffer.resize(needed, 0.0);

        let input_data = vec![self.buffer.clone()];
        let input = SequentialSliceOfVecs::new(&input_data, 1, needed)
            .map_err(|e| SonaraError::Fft(format!("stream resample flush input: {e}")))?;

        let out_len = self.resampler.output_frames_next();
        let mut output_data = vec![vec![0.0f32; out_len]];
        let mut out_buf = SequentialSliceOfVecs::new_mut(&mut output_data, 1, out_len)
            .map_err(|e| SonaraError::Fft(format!("stream resample flush output: {e}")))?;

        let (_n_in, n_out) = self
            .resampler
            .process_into_buffer(&input, &mut out_buf, None)
            .map_err(|e| SonaraError::Fft(format!("stream resample flush: {e}")))?;

        self.buffer.clear();
        output_data[0].truncate(n_out);
        Ok(output_data.into_iter().next().unwrap())
    }
}

/// Stream audio with per-block resampling.
///
/// Loads the file at its native sample rate, then chunks and resamples
/// each block independently. This reduces peak memory vs resampling the
/// entire file at once.
pub fn stream_with_resample(
    path: &Path,
    block_length: usize,
    frame_length: usize,
    hop_length: usize,
    target_sr: u32,
    mono: bool,
) -> Result<Vec<AudioBuffer>> {
    // Load at native rate (sr=0 means preserve native)
    let (y_native, native_sr) = load(path, 0, mono, 0.0, 0.0)?;

    if native_sr == target_sr {
        // No resampling needed, just chunk
        return stream_from_signal(y_native.view(), block_length, frame_length, hop_length);
    }

    // Create stateful resampler
    let mut resampler = StreamResampler::new(native_sr, target_sr, 1024)?;
    let step = block_length * hop_length;

    // Resample in chunks of `step` samples (native rate)
    let native_step = (step as f64 * native_sr as f64 / target_sr as f64).ceil() as usize;
    let mut resampled = Vec::new();
    let mut pos = 0;

    while pos < y_native.len() {
        let end = (pos + native_step).min(y_native.len());
        let chunk = &y_native.as_slice().unwrap()[pos..end];
        let out = resampler.process_chunk(chunk)?;
        resampled.extend_from_slice(&out);
        pos = end;
    }

    // Flush remaining
    let flush = resampler.flush()?;
    resampled.extend_from_slice(&flush);

    let y_resampled = Array1::from_vec(resampled);
    stream_from_signal(y_resampled.view(), block_length, frame_length, hop_length)
}

/// Helper: chunk a signal into stream blocks.
fn stream_from_signal(
    y: ArrayView1<Float>,
    block_length: usize,
    frame_length: usize,
    hop_length: usize,
) -> Result<Vec<AudioBuffer>> {
    let step = block_length * hop_length;
    let mut blocks = Vec::new();
    let mut pos = 0;
    while pos < y.len() {
        let end = (pos + step + frame_length).min(y.len());
        blocks.push(y.slice(ndarray::s![pos..end]).to_owned());
        pos += step;
    }
    Ok(blocks)
}

/// Compute the autocorrelation of a 1-D signal.
///
/// Uses the FFT method: autocorrelation = ifft(|fft(x)|²).
pub fn autocorrelate(y: ArrayView1<Float>, max_size: Option<usize>) -> Result<Array1<Float>> {
    let n = y.len();
    let fft_size = (2 * n).next_power_of_two();

    // Zero-pad to fft_size
    let mut padded = vec![0.0; fft_size];
    for i in 0..n {
        padded[i] = y[i];
    }

    // FFT
    let mut spectrum = crate::core::fft::rfft_alloc(&mut padded)?;

    // |X|² (power spectrum)
    for c in spectrum.iter_mut() {
        *c = num_complex::Complex::new(c.norm_sqr(), 0.0);
    }

    // IFFT
    let autocorr = crate::core::fft::irfft_alloc(&mut spectrum, fft_size)?;

    // Trim and normalize
    let max_lag = max_size.unwrap_or(n).min(n);
    let mut result = Array1::<Float>::zeros(max_lag);
    let scale = 1.0 / fft_size as Float;
    for i in 0..max_lag {
        result[i] = autocorr[i] * scale;
    }

    Ok(result)
}

/// Linear Prediction Coefficients via the Burg method.
///
/// Returns `order + 1` coefficients where the first is always 1.0.
pub fn lpc(y: ArrayView1<Float>, order: usize) -> Result<Array1<Float>> {
    let n = y.len();
    if n <= order {
        return Err(SonaraError::InsufficientData {
            needed: order + 1,
            got: n,
        });
    }

    // Levinson-Durbin recursion on autocorrelation
    let acf = autocorrelate(y, Some(order + 1))?;

    if acf[0].abs() < 1e-30 {
        // Signal is essentially zero
        let mut coeffs = Array1::<Float>::zeros(order + 1);
        coeffs[0] = 1.0;
        return Ok(coeffs);
    }

    let mut a = vec![0.0; order + 1];
    let mut a_prev = vec![0.0; order + 1];
    a[0] = 1.0;
    a_prev[0] = 1.0;

    let mut error = acf[0];

    for i in 1..=order {
        // Compute reflection coefficient
        let mut lambda = 0.0;
        for j in 0..i {
            lambda -= a_prev[j] * acf[i - j];
        }
        lambda /= error;

        // Update coefficients
        for j in 0..=i {
            a[j] = a_prev[j] + lambda * a_prev[i - j];
        }

        // Update error
        error *= 1.0 - lambda * lambda;

        if error.abs() < 1e-30 {
            break;
        }

        // Copy for next iteration
        a_prev[..=i].copy_from_slice(&a[..=i]);
    }

    Ok(Array1::from_vec(a))
}

/// Count zero crossings in a signal.
///
/// Returns a boolean array where `true` indicates a zero crossing.
pub fn zero_crossings(y: ArrayView1<Float>, threshold: Float) -> Array1<bool> {
    let n = y.len();
    if n < 2 {
        return Array1::from_elem(n, false);
    }

    Array1::from_shape_fn(n, |i| {
        if i == 0 {
            false
        } else {
            let a = if y[i - 1].abs() <= threshold {
                0.0
            } else {
                y[i - 1]
            };
            let b = if y[i].abs() <= threshold { 0.0 } else { y[i] };
            (a > 0.0 && b <= 0.0) || (a <= 0.0 && b > 0.0)
        }
    })
}

// ============================================================
// Signal generation
// ============================================================

/// Generate click (impulse) signals at specified times.
pub fn clicks(
    times: &[Float],
    sr: u32,
    length: usize,
    click_freq: Float,
    click_duration: Float,
) -> Array1<Float> {
    let mut y = Array1::<Float>::zeros(length);
    let click_samples = (click_duration * sr as Float).ceil() as usize;

    for &t in times {
        let start = (t * sr as Float).round() as usize;
        for i in 0..click_samples {
            let idx = start + i;
            if idx < length {
                y[idx] += (2.0 * PI * click_freq * i as Float / sr as Float).sin();
            }
        }
    }

    y
}

/// Generate a pure tone (sine wave).
pub fn tone(frequency: Float, sr: u32, length: usize) -> Array1<Float> {
    Array1::from_shape_fn(length, |i| {
        (2.0 * PI * frequency * i as Float / sr as Float).sin()
    })
}

/// Generate a chirp (swept-frequency sine wave).
///
/// Linear sweep from `fmin` to `fmax` over `length` samples.
pub fn chirp(fmin: Float, fmax: Float, sr: u32, length: usize) -> Array1<Float> {
    let duration = length as Float / sr as Float;
    Array1::from_shape_fn(length, |i| {
        let t = i as Float / sr as Float;
        let phase = 2.0 * PI * (fmin * t + (fmax - fmin) * t * t / (2.0 * duration));
        phase.sin()
    })
}

// ============================================================
// Companding
// ============================================================

/// Mu-law compression.
///
/// `y = sign(x) * ln(1 + mu*|x|) / ln(1 + mu)`
pub fn mu_compress(x: ArrayView1<Float>, mu: Float) -> Array1<Float> {
    let log_mu = (1.0 + mu).ln();
    x.mapv(|v| v.signum() * (1.0 + mu * v.abs()).ln() / log_mu)
}

/// Mu-law expansion (inverse of mu_compress).
///
/// `x = sign(y) * (1/mu) * ((1 + mu)^|y| - 1)`
pub fn mu_expand(y: ArrayView1<Float>, mu: Float) -> Array1<Float> {
    y.mapv(|v| v.signum() * ((1.0 + mu).powf(v.abs()) - 1.0) / mu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_to_mono_stereo() {
        let stereo =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let mono = to_mono(stereo.view());
        assert_eq!(mono.len(), 4);
        assert_abs_diff_eq!(mono[0], 3.0, epsilon = 1e-5); // (1+5)/2
        assert_abs_diff_eq!(mono[1], 4.0, epsilon = 1e-5); // (2+6)/2
    }

    #[test]
    fn test_tone_440hz() {
        let y = tone(440.0, 22050, 22050);
        assert_eq!(y.len(), 22050);
        // Check it starts at zero
        assert_abs_diff_eq!(y[0], 0.0, epsilon = 1e-5);
        // Check energy is reasonable (RMS of sine = 1/sqrt(2))
        let rms = (y.mapv(|v| v * v).sum() / y.len() as Float).sqrt();
        assert_abs_diff_eq!(rms, 1.0 / 2.0_f32.sqrt(), epsilon = 0.01);
    }

    #[test]
    fn test_chirp_basic() {
        let y = chirp(100.0, 1000.0, 22050, 22050);
        assert_eq!(y.len(), 22050);
        // Values should be in [-1, 1]
        for &v in y.iter() {
            assert!(v >= -1.01 && v <= 1.01);
        }
    }

    #[test]
    fn test_mu_compress_expand_roundtrip() {
        let x = array![-0.8, -0.3, 0.0, 0.5, 1.0];
        let mu = 255.0;
        let compressed = mu_compress(x.view(), mu);
        let expanded = mu_expand(compressed.view(), mu);
        for i in 0..x.len() {
            assert_abs_diff_eq!(x[i], expanded[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_mu_compress_zero() {
        let x = array![0.0];
        let compressed = mu_compress(x.view(), 255.0);
        assert_abs_diff_eq!(compressed[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_zero_crossings_sine() {
        let n = 1000;
        let y = Array1::from_shape_fn(n, |i| (2.0 * PI * 10.0 * i as Float / n as Float).sin());
        let zc = zero_crossings(y.view(), 0.0);
        let count: usize = zc.iter().filter(|&&v| v).count();
        // 10 cycles → ~20 zero crossings
        assert!(count >= 18 && count <= 22, "got {count} crossings");
    }

    #[test]
    fn test_zero_crossings_threshold() {
        let y = array![0.0, 0.01, -0.01, 0.0];
        // With high threshold, small values are treated as zero → fewer crossings
        let zc = zero_crossings(y.view(), 0.1);
        let count: usize = zc.iter().filter(|&&v| v).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_autocorrelate_sine() {
        let n = 2048;
        let freq = 100.0;
        let sr = 22050.0;
        let y = Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr).sin());
        let acf = autocorrelate(y.view(), Some(n)).unwrap();
        // Autocorrelation of sine peaks at 0 and at period
        assert!(acf[0] > 0.0); // Peak at lag 0
        let period_samples = (sr / freq).round() as usize;
        // Should have a peak near the period
        let peak_region = &acf.as_slice().unwrap()[period_samples - 5..period_samples + 5];
        let local_max = peak_region
            .iter()
            .copied()
            .fold(Float::NEG_INFINITY, Float::max);
        assert!(
            local_max > acf[0] * 0.5,
            "autocorrelation should peak near period"
        );
    }

    #[test]
    fn test_lpc_basic() {
        let y = Array1::from_shape_fn(256, |i| (i as Float * 0.1).sin());
        let coeffs = lpc(y.view(), 4).unwrap();
        assert_eq!(coeffs.len(), 5); // order + 1
        assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-5); // first coeff is always 1
    }

    #[test]
    fn test_lpc_insufficient_data() {
        let y = array![1.0, 2.0];
        assert!(lpc(y.view(), 5).is_err());
    }

    #[test]
    fn test_clicks_basic() {
        let y = clicks(&[0.0, 0.5], 22050, 22050, 1000.0, 0.01);
        assert_eq!(y.len(), 22050);
        // Should have energy near the click times
        assert!(y.slice(s![0..220]).iter().any(|&v| v.abs() > 0.01));
        let mid = 22050 / 2;
        assert!(y.slice(s![mid..mid + 220]).iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn test_resample_identity() {
        let y = Array1::from_shape_fn(1024, |i| (i as Float * 0.1).sin());
        let resampled = resample(y.view(), 22050, 22050).unwrap();
        assert_eq!(resampled.len(), y.len());
        for i in 0..y.len() {
            assert_abs_diff_eq!(y[i], resampled[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_decimate_half_length() {
        // 2:1 decimation should halve the length
        let y = Array1::from_shape_fn(44100, |i| (i as Float * 0.1).sin());
        let decimated = resample(y.view(), 44100, 22050).unwrap();
        assert_eq!(decimated.len(), 22050);
    }

    #[test]
    fn test_patched_hound_reads_data_after_odd_length_unknown_chunk() {
        let wav = &b"RIFF\x32\x00\x00\x00WAVE\
JUNK\x03\x00\x00\x00\x01\x02\x03\x00\
fmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00\
data\x02\x00\x00\x00\x34\x12"[..];

        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 8_000);
        let samples: Vec<i16> = reader.samples().map(|sample| sample.unwrap()).collect();
        assert_eq!(samples, vec![0x1234]);
    }

    #[test]
    fn test_wav_falls_back_to_symphonia_when_hound_rejects_codec() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/fallback-adpcm-ima.wav"
        ));

        let hound_error = match hound::WavReader::open(path) {
            Ok(_) => panic!("the fixture must exercise the fallback rather than Hound"),
            Err(error) => error,
        };
        assert!(
            matches!(
                hound_error,
                hound::Error::Unsupported | hound::Error::FormatError(_)
            ),
            "expected a Hound format rejection, got {hound_error:?}"
        );

        let (y, sr, tags) = load_with_tags(path, 8_000, true, 0.0, 0.0, true).unwrap();
        assert_eq!(sr, 8_000);
        assert_eq!(tags, None, "WAV fallback must preserve WAV tag semantics");
        assert!(!y.is_empty());
        assert!(y.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn test_wav_fallback_reports_both_decoder_errors() {
        let dir = std::env::temp_dir().join("sonara_wav_fallback_error_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not-a-wave.wav");
        std::fs::write(&path, b"not a WAVE stream").unwrap();

        let error = load(&path, 8_000, true, 0.0, 0.0).unwrap_err();
        match error {
            SonaraError::Decode(message) => {
                assert!(
                    message.contains("WAV decode failed with Hound"),
                    "{message}"
                );
                assert!(
                    message.contains("Symphonia 0.6 fallback failed"),
                    "{message}"
                );
            }
            other => panic!("expected combined WAV decode error, got {other:?}"),
        }
    }

    #[test]
    fn test_wav_fallback_preserves_io_error_for_missing_file() {
        let path = std::env::temp_dir()
            .join("sonara_wav_fallback_missing_test")
            .join("definitely-missing.wav");

        let error = load(&path, 8_000, true, 0.0, 0.0).unwrap_err();
        match error {
            SonaraError::AudioFile(message) => {
                assert!(
                    message.contains("WAV decode failed with Hound"),
                    "{message}"
                );
                assert!(
                    message.contains("Symphonia 0.6 fallback failed"),
                    "{message}"
                );
            }
            other => panic!("expected combined WAV I/O error, got {other:?}"),
        }
    }

    #[test]
    fn test_load_with_tags_flac() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.flac"
        ));
        let (y, _sr, tags) = load_with_tags(path, 22050, true, 0.0, 0.0, true).unwrap();
        assert!(!y.is_empty());
        let t = tags.expect("tags requested → Some");
        assert_eq!(t.title.as_deref(), Some("Test Title"));
        assert_eq!(t.artist.as_deref(), Some("Test Artist"));
        assert_eq!(t.album.as_deref(), Some("Test Album"));
        assert_eq!(t.genre.as_deref(), Some("Electronic"));
        assert_eq!(t.year, Some(2024));
        // Vorbis ORIGINALDATE=1969 → original_year; year stays the file date.
        assert_eq!(t.original_year, Some(1969));
        assert_eq!(t.track_no, Some(3));
    }

    #[test]
    fn test_load_with_tags_mp3_id3() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.mp3"
        ));
        let (y, _sr, tags) = load_with_tags(path, 22050, true, 0.0, 0.0, true).unwrap();
        assert!(!y.is_empty());
        let t = tags.expect("tags requested → Some");
        assert_eq!(t.title.as_deref(), Some("Test Title"));
        assert_eq!(t.artist.as_deref(), Some("Test Artist"));
        assert_eq!(t.album.as_deref(), Some("Test Album"));
        assert_eq!(t.genre.as_deref(), Some("Electronic"));
        assert_eq!(t.year, Some(2024));
        // ID3v2.3 TORY=1969 (StandardTagKey::OriginalDate) → original_year;
        // year comes from TYER=2024 (the semantic split).
        assert_eq!(t.original_year, Some(1969));
        assert_eq!(t.track_no, Some(3));
    }

    #[test]
    fn test_load_with_tags_prefers_newer_id3v2_over_trailing_id3v1() {
        let fixture = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.mp3"
        ));
        let mut bytes = std::fs::read(fixture).unwrap();

        fn set_id3v1_field(tag: &mut [u8; 128], start: usize, len: usize, value: &[u8]) {
            assert!(value.len() <= len);
            tag[start..start + value.len()].copy_from_slice(value);
        }

        let mut id3v1 = [0_u8; 128];
        id3v1[..3].copy_from_slice(b"TAG");
        set_id3v1_field(&mut id3v1, 3, 30, b"Legacy Title");
        set_id3v1_field(&mut id3v1, 33, 30, b"Legacy Artist");
        set_id3v1_field(&mut id3v1, 63, 30, b"Legacy Album");
        set_id3v1_field(&mut id3v1, 93, 4, b"1999");
        bytes.extend_from_slice(&id3v1);

        let dir = std::env::temp_dir().join("sonara_metadata_revision_precedence_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("conflicting-id3.mp3");
        std::fs::write(&path, bytes).unwrap();

        let (_y, _sr, tags) = load_with_tags(&path, 22_050, true, 0.0, 0.0, true).unwrap();
        let tags = tags.expect("tags requested");
        assert_eq!(tags.title.as_deref(), Some("Test Title"));
        assert_eq!(tags.artist.as_deref(), Some("Test Artist"));
        assert_eq!(tags.album.as_deref(), Some("Test Album"));
        assert_eq!(tags.year, Some(2024));
    }

    // Fixtures for decode-recovery tests (both fully synthetic):
    // - corrupt.mp3: 3 s 440 Hz tone (64 kbps mono CBR, no Xing header) with
    //   600 bytes zeroed at the 40% offset — several mid-stream frames
    //   damaged (recoverable; sync resumes after the hole).
    // - allbad.mp3: 40 hand-built MPEG1 Layer III frames whose side info
    //   claims part2_3_length=4095 bits with main_data_begin=0 — every
    //   packet raises a DecodeError ("huffman decode overrun"), so nothing
    //   decodes and the guardrail must surface the first decode error.

    #[test]
    fn test_load_recovers_from_damaged_packets() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/corrupt.mp3"
        ));
        let (y, sr) = load(path, 22050, true, 0.0, 0.0).unwrap();
        assert_eq!(sr, 22050);
        // 3 s tone minus the corrupted hole (~0.1 s) and decoder priming;
        // recovery must yield the overwhelming majority of the audio.
        let secs = y.len() as f64 / sr as f64;
        assert!(
            (2.5..=3.2).contains(&secs),
            "expected ~3s recovered, got {secs:.2}s"
        );
        assert!(y.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_load_all_packets_damaged_still_errors() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/allbad.mp3"
        ));
        let res = load(path, 22050, true, 0.0, 0.0);
        match res {
            Err(SonaraError::Decode(msg)) => {
                assert!(msg.contains("allbad.mp3"), "path missing in: {msg}");
                assert!(msg.contains("stage=decode"), "stage missing in: {msg}");
            }
            other => panic!("all-packets-damaged file must fail with Decode: {other:?}"),
        }
    }

    #[test]
    fn test_load_fake_mp3_still_errors() {
        let dir = std::env::temp_dir().join("sonara_fake_mp3_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not_audio.mp3");
        std::fs::write(&path, "this is not an mp3 file\n".repeat(200)).unwrap();
        let res = load(&path, 22050, true, 0.0, 0.0);
        assert!(
            matches!(
                res,
                Err(SonaraError::UnsupportedFormat(_)) | Err(SonaraError::Decode(_))
            ),
            "fake mp3 must still fail: {res:?}"
        );
    }

    #[test]
    fn test_load_mp3_skips_malformed_id3_text_frame() {
        let fixture = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.mp3"
        ));
        let mut bytes = std::fs::read(fixture).unwrap();

        // The synthetic fixture starts with an ID3v2.3 TIT2 frame. Encoding
        // values 0..=3 are defined. Symphonia 0.6 skips the malformed TIT2
        // value while preserving the remaining valid ID3 fields.
        assert_eq!(&bytes[..14], b"ID3\x03\0\0\0\0\x01,TIT2");
        bytes[20] = 4;

        let dir = std::env::temp_dir().join("sonara_malformed_id3_mp3_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("malformed-text-frame.mp3");
        std::fs::write(&path, bytes).unwrap();

        let (y, sr, tags) = load_with_tags(&path, 22050, true, 0.0, 0.0, true).unwrap();
        assert_eq!(sr, 22050);
        assert!(!y.is_empty());
        assert!(y.iter().all(|s| s.is_finite()));
        let tags = tags.expect("tags requested");
        assert_eq!(tags.title, None);
        assert_eq!(tags.artist.as_deref(), Some("Test Artist"));
        assert_eq!(tags.album.as_deref(), Some("Test Album"));
        assert_eq!(tags.genre.as_deref(), Some("Electronic"));
        assert_eq!(tags.year, Some(2024));
        assert_eq!(tags.original_year, Some(1969));
        assert_eq!(tags.track_no, Some(3));
    }

    #[test]
    fn test_load_mp3_retries_after_false_adts_probe() {
        let fixture = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.mp3"
        ));
        let source = std::fs::read(fixture).unwrap();

        // Strip the fixture's 310-byte ID3 tag and prepend the minimum ADTS
        // header that selects reserved sample-rate index 13. Generic probing
        // commits to ADTS and errors; the following bytes are valid synthetic
        // MP3 frames.
        let mut bytes = vec![0xff, 0xf1, 0x7b, 0x7e, 0x2e, 0x58, 0xe2];
        bytes.extend_from_slice(&source[310..]);

        let dir = std::env::temp_dir().join("sonara_false_adts_mp3_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("false-adts-marker.mp3");
        std::fs::write(&path, bytes).unwrap();

        let (y, sr) = load(&path, 22050, true, 0.0, 0.0).unwrap();
        assert_eq!(sr, 22050);
        assert!(!y.is_empty());
        assert!(y.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_probe_without_metadata_skips_trailing_id3v1() {
        use symphonia::core::formats::probe::Hint;
        use symphonia::core::io::MediaSourceStream;

        let fixture = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.mp3"
        ));
        let mut bytes = std::fs::read(fixture).unwrap();
        let mut id3v1 = [0_u8; 128];
        let trailing_title = b"Trailing Title";
        id3v1[..3].copy_from_slice(b"TAG");
        id3v1[3..3 + trailing_title.len()].copy_from_slice(trailing_title);
        bytes.extend_from_slice(&id3v1);
        let path = std::env::temp_dir().join("sonara_no_metadata_trailing_id3v1.mp3");
        std::fs::write(&path, bytes).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        hint.with_extension("mp3");

        let mut format = probe_symphonia_without_metadata(&hint, mss).unwrap();
        let mut metadata = format.metadata();
        let mut tag_values = Vec::new();
        while let Some(revision) = metadata.current() {
            tag_values.extend(
                revision
                    .media
                    .tags
                    .iter()
                    .map(|tag| tag.raw.value.to_string()),
            );
            if metadata.pop().is_none() {
                break;
            }
        }
        assert!(
            !tag_values.iter().any(|value| value == "Trailing Title"),
            "the no-metadata probe must not read a trailing ID3v1 revision"
        );
        drop(format);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_load_without_tags_is_none() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/fixtures/tagged.flac"
        ));
        let (_y, _sr, tags) = load_with_tags(path, 22050, true, 0.0, 0.0, false).unwrap();
        assert!(tags.is_none(), "want_tags=false must not read tags");
    }

    #[test]
    fn test_parse_year_and_track() {
        assert_eq!(parse_year("2024"), Some(2024));
        assert_eq!(parse_year("2024-05-01"), Some(2024));
        assert_eq!(parse_year("24"), None);
        assert_eq!(parse_year(""), None);
        assert_eq!(parse_leading_u32("3"), Some(3));
        assert_eq!(parse_leading_u32("3/12"), Some(3));
        assert_eq!(parse_leading_u32(""), None);
    }

    #[test]
    fn test_is_original_date_key() {
        // Bare frame/key names, any case.
        assert!(is_original_date_key("TORY"));
        assert!(is_original_date_key("TDOR"));
        assert!(is_original_date_key("ORIGINALDATE"));
        assert!(is_original_date_key("originalyear"));
        assert!(is_original_date_key("OriginalYear"));
        // TXXX-prefixed user frames, prefix and desc both case-insensitive.
        assert!(is_original_date_key("TXXX:originalyear"));
        assert!(is_original_date_key("txxx:ORIGINALDATE"));
        // Non-matches.
        assert!(!is_original_date_key("date"));
        assert!(!is_original_date_key("year"));
        assert!(!is_original_date_key("TYER"));
        assert!(!is_original_date_key("TXXX:comment"));
    }

    /// Build a `Tag` with a raw key and no standard tag.
    fn raw_tag(key: &str, val: &str) -> symphonia::core::meta::Tag {
        symphonia::core::meta::Tag::new_from_parts(key, val, None)
    }

    #[test]
    fn test_merge_tags_year_original_year_split() {
        use std::sync::Arc;
        use symphonia::core::meta::{StandardTag, Tag};
        // ReleaseDate is the (reissue) file year; OriginalReleaseDate is the original year.
        let tags = vec![
            Tag::new_from_parts(
                "",
                "2024",
                Some(StandardTag::ReleaseDate(Arc::new("2024".to_string()))),
            ),
            Tag::new_from_parts(
                "",
                "1969-08-15",
                Some(StandardTag::OriginalReleaseDate(Arc::new(
                    "1969-08-15".to_string(),
                ))),
            ),
        ];
        let mut t = TrackTags::default();
        merge_tags(&mut t, &tags);
        assert_eq!(t.year, Some(2024));
        assert_eq!(t.original_year, Some(1969));
    }

    #[test]
    fn test_merge_tags_original_year_raw_key() {
        // No std_key at all: only a raw TXXX:originalyear frame.
        let tags = vec![raw_tag("TXXX:originalyear", "1969")];
        let mut t = TrackTags::default();
        merge_tags(&mut t, &tags);
        assert_eq!(t.original_year, Some(1969));
        assert_eq!(t.year, None);
    }

    #[test]
    fn test_merge_tags_no_original_year_is_none() {
        use std::sync::Arc;
        use symphonia::core::meta::{StandardTag, Tag};
        // A file with only a plain Date tag → original_year stays None.
        let tags = vec![Tag::new_from_parts(
            "",
            "2024",
            Some(StandardTag::ReleaseDate(Arc::new("2024".to_string()))),
        )];
        let mut t = TrackTags::default();
        merge_tags(&mut t, &tags);
        assert_eq!(t.year, Some(2024));
        assert_eq!(t.original_year, None);
    }

    #[test]
    fn test_merge_tags_original_year_first_wins() {
        use std::sync::Arc;
        use symphonia::core::meta::{StandardTag, Tag};
        // First parseable original-date value wins over later ones.
        let tags = vec![
            Tag::new_from_parts(
                "",
                "1969",
                Some(StandardTag::OriginalReleaseDate(Arc::new(
                    "1969".to_string(),
                ))),
            ),
            raw_tag("ORIGINALYEAR", "1987"),
        ];
        let mut t = TrackTags::default();
        merge_tags(&mut t, &tags);
        assert_eq!(t.original_year, Some(1969));
    }

    #[test]
    fn test_decimate_half_low_freq_preserved() {
        // A low-frequency sine (well below Nyquist) should be preserved
        let sr = 44100u32;
        let freq = 440.0;
        let n = sr as usize; // 1 second
        let y = Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin());
        let decimated = resample(y.view(), sr, sr / 2).unwrap();
        // Check RMS is preserved (energy should be similar)
        let rms_orig = (y.mapv(|v| v * v).sum() / y.len() as Float).sqrt();
        let rms_dec = (decimated.mapv(|v| v * v).sum() / decimated.len() as Float).sqrt();
        assert_abs_diff_eq!(rms_orig, rms_dec, epsilon = 0.05);
    }
}
