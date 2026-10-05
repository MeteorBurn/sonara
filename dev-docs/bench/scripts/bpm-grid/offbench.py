"""Off-bench test set for bpm-precision: text search over the DJTS library, then a
filtered, stratified selection.

    python offbench.py search [--labels L ...] [--models clap mulan] [--limit 60] [--no-negatives]
    python offbench.py pool
    python offbench.py select

Everything lives in the dataset folder `dev-docs/bench/datasets/offbench-2026-10/`
(override: SONARA_OFFBENCH_DATA); machine paths come from the "## Machine paths" JSON
block of its README. `search` needs the DJTS server running on the library named
there; it posts each label's prompt bank through the `text-music-search` helper
(`project_text_search.py --expected-db ... --json`) and keeps every full
`{results, execution}` envelope under `search/<model>/<label>.json`. `pool` joins all
returned tracks with the library (read-only): path, duration, tag BPM, stored Sonara
tempo and vocal probability, the `voice_presence` classifier and the MAEST top genres.
`select` keeps existing instrumental tracks of at least 90 s that are not in the
bpm-grid bench. The library is almost all 115-150 BPM, so the scarce tempos are taken
whole: every such track the searches returned goes to "slow" (tag BPM, else the stored
Sonara raw BPM, below 100) or "fast" (150 and up, or a MAEST drum and bass, jungle,
footwork or juke style, whose tags often sit at half tempo). The rest fill "broken",
"straight" and "live" round-robin over their labels in rank order, a label counting
only when the track's MAEST genres agree with it (labels without a MAEST style are not
checked). Outputs: `pool.csv`, `selection.csv`, `playlist.txt` (line = idx),
`playlist.m3u8`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATASET = Path(os.environ.get("SONARA_OFFBENCH_DATA") or REPO / "dev-docs" / "bench" / "datasets" / "offbench-2026-10")
_BLOCK = re.compile(r"^## Machine paths\b.*?^```json[ \t]*\r?\n(.*?)^```", re.S | re.M)

# Stratum of each label; rhythm presets get theirs from the tempo (see stratum_of).
STRATA = {
    "slow": ["hiphop_instrumental", "trip_hop_downtempo", "dub_reggae", "dubstep_halftime", "trap_instrumental"],
    "fast": ["drum_and_bass", "jungle", "footwork_juke", "hardcore_gabber", "hardstyle"],
    "broken": ["breaks", "uk_garage_2step", "electro", "broken_beat", "amapiano", "dembow_reggaeton",
               "club_jersey_baltimore", "afro_house"],
    "straight": ["house", "techno", "trance"],
    "live": ["funk_soul_live", "jazz_fusion", "disco_live", "afrobeat", "post_rock"],
}
RHYTHM_LABELS = {"rhythm_four_on_the_floor": "straight", "rhythm_breakbeat": "broken", "rhythm_syncopated": "broken",
                 "rhythm_polyrhythm": "broken", "rhythm_shuffle": "broken", "groove_swing": "broken",
                 "groove_laid_back": "broken"}
# Strata filled by label quotas; "slow" and "fast" take every track of their tempo.
QUOTA = {"broken": 50, "straight": 30, "live": 40}
SLOW_BELOW_BPM = 100.0
FAST_FROM_BPM = 150.0
FAST_STYLES = ("Electronic---Drum n Bass", "Electronic---Jungle", "Electronic---Footwork", "Electronic---Juke")
# MAEST (Discogs) styles that confirm a label; a style ending in "---" matches its whole genre.
MAEST = {
    "hiphop_instrumental": ["Hip Hop---"],
    "trip_hop_downtempo": ["Electronic---Trip Hop", "Hip Hop---Trip Hop", "Electronic---Downtempo", "Electronic---Abstract"],
    "dub_reggae": ["Reggae---", "Electronic---Dub"],
    "dubstep_halftime": ["Electronic---Dubstep", "Hip Hop---Bass Music", "Electronic---Grime"],
    "trap_instrumental": ["Hip Hop---"],
    "drum_and_bass": ["Electronic---Drum n Bass", "Electronic---Jungle"],
    "jungle": ["Electronic---Jungle", "Electronic---Drum n Bass"],
    "footwork_juke": ["Electronic---Footwork", "Electronic---Juke", "Electronic---Ghetto", "Electronic---Ghettotech",
                      "Electronic---Ghetto House"],
    "hardcore_gabber": ["Electronic---Hardcore", "Electronic---Gabber", "Electronic---Hard Techno"],
    "hardstyle": ["Electronic---Hardstyle", "Electronic---Hard Trance", "Electronic---Hardcore", "Electronic---Hard Techno"],
    "breaks": ["Electronic---Breakbeat", "Electronic---Breaks", "Electronic---Progressive Breaks", "Electronic---Big Beat"],
    "uk_garage_2step": ["Electronic---UK Garage", "Electronic---Speed Garage", "Electronic---Bassline",
                        "Electronic---Grime", "Electronic---UK Funky"],
    "electro": ["Electronic---Electro", "Hip Hop---Electro", "Electronic---Ghettotech"],
    "broken_beat": ["Electronic---Broken Beat", "Electronic---Future Jazz", "Electronic---Jazzdance"],
    "club_jersey_baltimore": ["Electronic---Baltimore Club", "Electronic---Ghetto", "Electronic---Ghettotech",
                              "Electronic---Juke"],
    "afro_house": ["Electronic---Tribal House", "Electronic---Tribal", "Funk / Soul---Afrobeat"],
    "house": ["Electronic---House", "Electronic---Deep House", "Electronic---Tech House"],
    "techno": ["Electronic---Techno", "Electronic---Minimal Techno", "Electronic---Deep Techno", "Electronic---Hard Techno"],
    "trance": ["Electronic---Trance", "Electronic---Progressive Trance", "Electronic---Psy-Trance", "Electronic---Goa Trance",
               "Electronic---Neo Trance", "Electronic---Hard Trance"],
    "funk_soul_live": ["Funk / Soul---"],
    "jazz_fusion": ["Jazz---", "Electronic---Future Jazz", "Electronic---Acid Jazz", "Hip Hop---Jazzy Hip-Hop"],
    "disco_live": ["Funk / Soul---Disco", "Funk / Soul---Boogie", "Electronic---Disco"],
    "afrobeat": ["Funk / Soul---", "Jazz---"],
    "post_rock": ["Rock---"],
}
MAEST_TOP = 5
MIN_DURATION_SEC = 90.0
MAX_VOICE_PRESENCE = 0.3
MAX_VOCAL_PROBABILITY = 0.5


def machine() -> dict:
    match = _BLOCK.search((DATASET / "README.md").read_text(encoding="utf-8"))
    return {k: os.path.expandvars(v) for k, v in json.loads(match.group(1)).items()} if match else {}


def norm_path(path: str) -> str:
    return path.replace("\\", "/").casefold()


def run_search(args: argparse.Namespace) -> int:
    m = machine()
    for model in args.models:
        bank = json.loads((DATASET / "prompts" / f"{model}.json").read_text(encoding="utf-8"))
        negatives = [] if args.no_negatives else bank.get("global_hard_negatives", [])
        out_dir = DATASET / "search" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        for label, spec in bank["labels"].items():
            if args.labels and label not in args.labels:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                pos, neg = Path(tmp) / "positive.txt", Path(tmp) / "negative.txt"
                pos.write_text("\n".join(spec["prompts"]) + "\n", encoding="utf-8")
                cmd = [m["djts_python"], m["djts_search"], "--base-url", m["djts_api"], "--model", model,
                       "--expected-db", m["djts_db"], "--positive-file", str(pos), "--limit", str(args.limit), "--json"]
                if negatives:
                    neg.write_text("\n".join(negatives) + "\n", encoding="utf-8")
                    cmd += ["--negative-file", str(neg)]
                done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if done.returncode != 0:
                print(f"{model} {label}: FAILED ({done.returncode}) {done.stderr.strip()[:400]}")
                return 1
            envelope = json.loads(done.stdout)
            (out_dir / f"{label}.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=1), encoding="utf-8")
            execution = envelope["execution"]
            print(f"{model:5s} {label:26s} results {len(envelope['results']):3d} run {execution['run_id'][:8]} "
                  f"eligible {execution.get('eligible_count')} feedback {execution['feedback']['reason']}")
    return 0


def hits() -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for path in sorted((DATASET / "search").glob("*/*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        for rank, item in enumerate(envelope["results"], start=1):
            track = item["track"]
            out[int(track["track_id"])].append({"label": path.stem, "model": path.parent.name, "rank": rank,
                                                "score": float(item["score"]), "path": track.get("file_path")})
    return out


def run_pool(_: argparse.Namespace) -> int:
    m = machine()
    found = hits()
    con = sqlite3.connect(f"file:{m['djts_db']}?mode=ro", uri=True)
    rows = []
    for track_id, track_hits in sorted(found.items()):
        q = con.execute(
            "SELECT t.file_path, t.audio_duration_seconds, g.tag_bpm, s.detected_bpm, s.raw_bpm, s.bpm_confidence, "
            "s.tempo_variability, s.vocal_probability, mg.genres_json, mg.syncopated_rhythm, c.score "
            "FROM tracks t LEFT JOIN tags g ON g.track_id = t.track_id "
            "LEFT JOIN sonara_features s ON s.track_id = t.track_id "
            "LEFT JOIN maest_genres mg ON mg.track_id = t.track_id "
            "LEFT JOIN classifier_scores c ON c.track_id = t.track_id AND c.classifier_key = 'voice_presence' "
            "WHERE t.track_id = ?", (track_id,)).fetchone()
        if q is None:
            continue
        (path, duration, tag_bpm, det, raw, conf, tvar, vocal, genres, sync, voice) = q
        maest = [g["label"] for g in json.loads(genres)[:MAEST_TOP]] if genres else []
        best = min(track_hits, key=lambda h: (h["rank"], -h["score"]))
        rows.append({
            "track_id": track_id, "path": path, "duration_sec": duration, "tag_bpm": tag_bpm,
            "sonara_bpm": det, "sonara_raw_bpm": raw, "sonara_conf": conf, "tempo_variability": tvar,
            "vocal_probability": vocal, "voice_presence": voice, "syncopated_rhythm": sync,
            "maest": "|".join(maest), "best_label": best["label"], "best_model": best["model"],
            "best_rank": best["rank"],
            "hits": ";".join(f"{h['label']}:{h['model']}:{h['rank']}" for h in sorted(
                track_hits, key=lambda h: (h["rank"], h["label"]))),
        })
    con.close()
    with (DATASET / "pool.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"pool: {len(rows)} tracks from {sum(len(v) for v in found.values())} hits -> {DATASET / 'pool.csv'}")
    return 0


def maest_ok(label: str, maest: list[str]) -> bool:
    styles = MAEST.get(label)
    if not styles:
        return True
    return any(g == s or (s.endswith("---") and g.startswith(s)) for g in maest for s in styles)


def tempo_stratum(row: dict, maest: list[str]) -> str | None:
    """"slow" or "fast" by the stored tempo (tag BPM, else Sonara raw BPM) and genre."""
    tempo = float(row["tag_bpm"] or 0) or float(row["sonara_raw_bpm"] or 0)
    if any(style in maest for style in FAST_STYLES) or tempo >= FAST_FROM_BPM:
        return "fast"
    if 0 < tempo < SLOW_BELOW_BPM:
        return "slow"
    return None


def label_stratum(label: str) -> str | None:
    """The quota stratum of a label; slow and fast genre labels have none at mid tempo."""
    for stratum in QUOTA:
        if label in STRATA.get(stratum, []):
            return stratum
    return RHYTHM_LABELS.get(label)


def run_select(_: argparse.Namespace) -> int:
    m = machine()
    bench = {norm_path(line.strip()) for p in Path(m["bench_sources"]).glob("*.txt")
             for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
    with (DATASET / "pool.csv").open(encoding="utf-8", newline="") as handle:
        pool = list(csv.DictReader(handle))
    reasons: dict[str, int] = defaultdict(int)
    chosen: list[tuple[str, str, dict]] = []
    taken: set[str] = set()
    # stratum -> label -> eligible rows by their rank under that label
    queues: dict[str, dict[str, list[tuple[int, dict]]]] = defaultdict(lambda: defaultdict(list))
    for row in pool:
        if norm_path(row["path"]) in bench:
            reasons["in bench"] += 1
            continue
        if not row["duration_sec"] or float(row["duration_sec"]) < MIN_DURATION_SEC:
            reasons["short"] += 1
            continue
        voice = float(row["voice_presence"]) if row["voice_presence"] else None
        vocal = float(row["vocal_probability"]) if row["vocal_probability"] else 0.0
        if (voice is not None and voice >= MAX_VOICE_PRESENCE) or vocal >= MAX_VOCAL_PROBABILITY:
            reasons["vocals"] += 1
            continue
        if not Path(row["path"]).exists():
            reasons["missing file"] += 1
            continue
        maest = row["maest"].split("|") if row["maest"] else []
        scarce = tempo_stratum(row, maest)
        if scarce:
            chosen.append((scarce, row["best_label"], row))
            taken.add(row["track_id"])
            continue
        placed = False
        for hit in row["hits"].split(";"):
            label, _model, rank = hit.split(":")
            stratum = label_stratum(label)
            if stratum and maest_ok(label, maest):
                queues[stratum][label].append((int(rank), row))
                placed = True
        if not placed:
            reasons["no label for its tempo"] += 1
    for stratum in ("slow", "fast"):
        labels_in: dict[str, int] = defaultdict(int)
        for c in chosen:
            if c[0] == stratum:
                labels_in[c[1]] += 1
        print(f"{stratum:8s} {sum(labels_in.values()):3d} (every track of its tempo) by best label "
              f"{dict(sorted(labels_in.items()))}")
    for stratum, quota in QUOTA.items():
        labels = {k: sorted(v, key=lambda x: x[0]) for k, v in queues[stratum].items()}
        n = 0
        while n < quota and any(labels.values()):
            for label in sorted(labels):
                while labels[label] and labels[label][0][1]["track_id"] in taken:
                    labels[label].pop(0)
                if labels[label] and n < quota:
                    _rank, row = labels[label].pop(0)
                    taken.add(row["track_id"])
                    chosen.append((stratum, label, row))
                    n += 1
        print(f"{stratum:8s} {n:3d}/{quota} from {', '.join(f'{k} {len(v)}' for k, v in sorted(queues[stratum].items()))}")
    fields = ["idx", "stratum", "label"] + [k for k in pool[0] if k not in ("best_label", "best_model", "best_rank")]
    with (DATASET / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for idx, (stratum, label, row) in enumerate(chosen, start=1):
            writer.writerow({"idx": idx, "stratum": stratum, "label": label, **row})
    (DATASET / "playlist.txt").write_text("\n".join(r["path"] for _, _, r in chosen) + "\n", encoding="utf-8")
    (DATASET / "playlist.m3u8").write_text(
        "#EXTM3U\n" + "\n".join(r["path"].replace("/", "\\") for _, _, r in chosen) + "\n", encoding="utf-8")
    print(f"selected {len(chosen)}; excluded {dict(reasons)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("--labels", nargs="*")
    s.add_argument("--models", nargs="*", default=["clap", "mulan"])
    s.add_argument("--limit", type=int, default=60)
    s.add_argument("--no-negatives", action="store_true")
    sub.add_parser("pool")
    sub.add_parser("select")
    args = ap.parse_args()
    return {"search": run_search, "pool": run_pool, "select": run_select}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
