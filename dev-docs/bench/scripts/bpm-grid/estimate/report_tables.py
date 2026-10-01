"""Render results/*.json into markdown tables (results/tables.md) for REPORT.md."""

from __future__ import annotations

import json
import sys

import est_common as ec

ORDER = ["baseline", "fix1", "fix2", "fix3", "fix4", "1+2", "1+2+3", "1+2+3+4", "1+2+3+4+snap"]
LABEL = {"baseline": "Baseline 0.3.7", "fix1": "Fix 1 alone (BPM from beats)",
         "fix2": "Fix 2 alone (latency -24.35 ms)", "fix3": "Fix 3 alone (low-band downbeat)",
         "fix4": "Fix 4 alone (candidate re-rank + re-track)", "1+2": "1+2", "1+2+3": "1+2+3",
         "1+2+3+4": "**1+2+3+4 (combined)**", "1+2+3+4+snap": "extra: combined + grid snap"}


def pct(d, key="mean"):
    if not d or d.get(key) is None:
        return "-"
    s = f"{100 * d[key]:.1f}%"
    if d.get("lo") is not None:
        s += f" [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    return s


def num(d, nd=3):
    if not d or d.get("mean") is None:
        return "-"
    s = f"{d['mean']:.{nd}f}"
    if d.get("lo") is not None:
        s += f" [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"
    return s


def main() -> int:
    R = json.loads((ec.RESULTS / "combined.json").read_text(encoding="utf-8"))
    T, P = R["table"], R["paired_diff_vs_baseline"]
    out = []
    for split in ("holdout", "all", "dev"):
        a0 = T["baseline"][split]
        out.append(f"\n### BPM vs MIK ({split}: {a0['agree_tracks']} agree tracks)\n")
        out.append("| Variant | exact <=0.05 [95% CI] | <=0.5 | critical [95% CI] | counts exact/fine/minor/critical | abs err p50 / p90 (non-crit) |")
        out.append("|---|---|---|---|---|---|")
        for v in ORDER:
            a = T[v][split]
            c = a["counts"]
            out.append(f"| {LABEL[v]} | {pct(a['exact_rate'])} | {pct(a['le_0.5_rate'], 'mean')} | "
                       f"{pct(a['critical_rate'])} | {c['exact']}/{c['fine']}/{c['minor']}/{c['critical']} | "
                       f"{a['abs_err_noncrit_p50']:.3f} / {a['abs_err_noncrit_p90']:.3f} |")
        out.append(f"\n### Beat grid ({split})\n")
        out.append(f"| Variant | F +-70 ms vs BT [95% CI] | F +-20 ms vs BT | F +-10 ms vs BT | phase vs BT ms (p10..p90) | "
                   f"F70/F20/F10 vs MIK grid (n={a0['mik_grid_tracks']}) | phase vs MIK ms |")
        out.append("|---|---|---|---|---|---|---|")
        for v in ORDER:
            a = T[v][split]
            out.append(f"| {LABEL[v]} | {num(a['F70_mean'])} | {num(a['F20_mean'])} | {num(a['F10_mean'])} | "
                       f"{a['phase_ms_median']:+.1f} ({a['phase_ms_p10_p90'][0]:+.1f}..{a['phase_ms_p10_p90'][1]:+.1f}) | "
                       f"{a['MF70_mean']:.3f} / {a['MF20_mean']:.3f} / {a['MF10_mean']:.3f} | {a['mik_phase_ms_median']:+.1f} |")
        out.append(f"\n### Downbeats ({split})\n")
        out.append(f"| Variant | downbeat F +-70 ms vs BT [95% CI] | MIK cues on downbeats (cues >= 5 s, {a0['cue_tracks']} tracks) | tracks with majority of cues hit |")
        out.append("|---|---|---|---|")
        for v in ORDER:
            a = T[v][split]
            out.append(f"| {LABEL[v]} | {num(a['DF70_mean'])} | {100 * a['cue_hit_rate']:.1f}% | "
                       f"{100 * a['cue_track_majority']:.1f}% |")
    for split in ("holdout", "all"):
        out.append(f"\n### Paired difference vs baseline ({split}; bootstrap 95% CI, same tracks)\n")
        out.append("| Variant | exact rate | critical rate | F +-20 ms | F +-10 ms | downbeat F | F vs MIK +-20 ms |")
        out.append("|---|---|---|---|---|---|---|")
        for v in ORDER[1:]:
            d = P[v][split]
            f = lambda k, scale=100, unit="pp": (f"{scale * d[k]['diff']:+.1f} [{scale * d[k]['lo']:+.1f}, {scale * d[k]['hi']:+.1f}] {unit}"
                                                if k in d else "-")
            g = lambda k: (f"{d[k]['diff']:+.3f} [{d[k]['lo']:+.3f}, {d[k]['hi']:+.3f}]" if k in d else "-")
            out.append(f"| {LABEL[v]} | {f('exact')} | {f('critical')} | {g('F20')} | {g('F10')} | {g('DF70')} | {g('MF20')} |")
    out.append("\n### Fix 4 transitions (critical -> not critical)\n")
    out.append("| Pipeline / split | fixed | regressed | switched |")
    out.append("|---|---|---|---|")
    for k, v in R["fix4_transitions"].items():
        out.append(f"| {k} | {v['fixed']} | {v['regressed']} | {v['switched']} |")
    text = "\n".join(out) + "\n"
    (ec.RESULTS / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
