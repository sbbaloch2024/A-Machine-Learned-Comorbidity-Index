"""
TREND TABLE CODE
================

Run this script from the `RiskCurves/` directory.

Expected diagnostic CSVs
------------------------
This script expects the following files in the sibling `DiagnosticCSV/` directory:

    ../DiagnosticCSV/CORE_seed11_test_scores_and_outcomes.csv
    ../DiagnosticCSV/CORE_seed101_test_scores_and_outcomes.csv
    ../DiagnosticCSV/CORE_seed1001_test_scores_and_outcomes.csv

Generate 4 binned and 4 isotonic curves per seed
-------------------------------------------------

Run from the `RiskCurves/` directory:

    for SEED in 11 101 1001; do
      CSV="../DiagnosticCSV/CORE_seed${SEED}_test_scores_and_outcomes.csv"
      OUTDIR="seed${SEED}"
      mkdir -p "${OUTDIR}/plots_unconstrained" "${OUTDIR}/plots_isotonic"

      echo "======================================"
      echo "Running MIMIC-IV seed ${SEED}"
      echo "CSV: ${CSV}"
      echo "OUTDIR: ${OUTDIR}"
      echo "======================================"

      python3 fit_risk_curves.py \
        --csv "${CSV}" \
        --score_col learned_score \
        --targets mortality mortality_30d long_stay icu_transfer \
        --method bins \
        --n_bins 60 \
        --out "${OUTDIR}/curves_bins.csv"

      python3 fit_risk_curves.py \
        --csv "${CSV}" \
        --score_col learned_score \
        --targets mortality mortality_30d long_stay icu_transfer \
        --method isotonic \
        --grid_n 200 \
        --out "${OUTDIR}/curves_isotonic.csv"

      SEED="${SEED}" python3 - <<'PY'
import os
import pandas as pd
import matplotlib.pyplot as plt

seed = os.environ["SEED"]
outdir = f"seed{seed}"

DISPLAY = {
    "mortality": "mortality",
    "mortality_30d": "mortality_30d",
    "long_stay": "length of stay",
    "icu_transfer": "icu_transfer",
}

# Plot: one PNG per task (binned)
df = pd.read_csv(f"{outdir}/curves_bins.csv")
for t in df["task"].unique():
    d = df[df["task"] == t].sort_values("score")
    label = DISPLAY.get(t, t)

    plt.figure(figsize=(6, 4))
    plt.plot(d["score"], d["p"])
    plt.xlabel("learned_score")
    plt.ylabel(f"P({label}=1 | score bin)")
    plt.title(f"{label} (binned, unconstrained)")
    plt.tight_layout()
    plt.savefig(
        f"{outdir}/plots_unconstrained/curve_{t}_bins.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# Plot: all tasks in one PNG (binned)
plt.figure(figsize=(8, 5))
for t in df["task"].unique():
    d = df[df["task"] == t].sort_values("score")
    label = DISPLAY.get(t, t)
    plt.plot(d["score"], d["p"], label=label)

plt.xlabel("learned_score")
plt.ylabel("P(outcome=1 | score bin)")
plt.title("Risk curves by outcome (binned, unconstrained)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/risk_curves_all_bins.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot: one PNG per task (isotonic)
df = pd.read_csv(f"{outdir}/curves_isotonic.csv")
for t in df["task"].unique():
    d = df[df["task"] == t].sort_values("score")
    label = DISPLAY.get(t, t)

    plt.figure(figsize=(6, 4))
    plt.plot(d["score"], d["p"])
    plt.xlabel("learned_score")
    plt.ylabel(f"P({label}=1 | score)")
    plt.title(f"{label} (isotonic, monotone)")
    plt.tight_layout()
    plt.savefig(
        f"{outdir}/plots_isotonic/curve_{t}_isotonic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# Plot: all tasks in one PNG (isotonic)
plt.figure(figsize=(8, 5))
for t in df["task"].unique():
    d = df[df["task"] == t].sort_values("score")
    label = DISPLAY.get(t, t)
    plt.plot(d["score"], d["p"], label=label)

plt.xlabel("learned_score")
plt.ylabel("P(outcome=1 | score)")
plt.title("Risk curves by outcome (isotonic, monotone)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/risk_curves_all_isotonic.png", dpi=300, bbox_inches="tight")
plt.close()
PY

    done

Compute one LaTeX row per seed
------------------------------

Run from the `RiskCurves/` directory:

    for SEED in 11 101 1001; do
      SEED="${SEED}" python3 - <<'PY'
import os
import pandas as pd

seed = os.environ["SEED"]
CSV = f"seed{seed}/curves_isotonic.csv"
TASKS = ["mortality", "mortality_30d", "long_stay", "icu_transfer"]

df = pd.read_csv(CSV)

def label_from_curve(d):
    d = d.sort_values("score").reset_index(drop=True)
    p5 = float(d["p"].iloc[int(0.05 * (len(d) - 1))])
    p95 = float(d["p"].iloc[int(0.95 * (len(d) - 1))])
    delta = p95 - p5

    if delta >= 0.05:
        lab = "inc."
    elif delta >= 0.01:
        lab = "weakly inc."
    else:
        lab = "flat"

    return lab, delta

entries = []
for t in TASKS:
    d = df[df["task"] == t]
    lab, delta = label_from_curve(d)
    entries.append(f"{lab} (\\(\\Delta={delta:.3f}\\))")

print(f"Seed {seed}:")
print("\\textbf{MIMIC-IV} & " + " & ".join(entries) + " \\\\")
PY
    done

Compute mean ± SD LaTeX row across seeds
-----------------------------------------

Run from the `RiskCurves/` directory:

    python3 - <<'PY'
import numpy as np
import pandas as pd

SEEDS = [11, 101, 1001]
TASKS = ["mortality", "mortality_30d", "long_stay", "icu_transfer"]

def label_from_delta(delta):
    if delta >= 0.05:
        return "inc."
    elif delta >= 0.01:
        return "weakly inc."
    return "flat"

def delta_from_curve(d):
    d = d.sort_values("score").reset_index(drop=True)
    p5 = float(d["p"].iloc[int(0.05 * (len(d) - 1))])
    p95 = float(d["p"].iloc[int(0.95 * (len(d) - 1))])
    return p95 - p5

all_deltas = {t: [] for t in TASKS}

for seed in SEEDS:
    df = pd.read_csv(f"seed{seed}/curves_isotonic.csv")
    for t in TASKS:
        d = df[df["task"] == t]
        all_deltas[t].append(delta_from_curve(d))

entries = []
for t in TASKS:
    vals = np.asarray(all_deltas[t], dtype=float)
    mean = vals.mean()
    sd = vals.std(ddof=1)
    lab = label_from_delta(mean)
    entries.append(f"{lab} (\\(\\Delta={mean:.3f}\\pm{sd:.3f}\\))")

print("\\textbf{MIMIC-IV} & " + " & ".join(entries) + " \\\\")
print()
print("Raw deltas:")
for t in TASKS:
    print(t, [round(x, 3) for x in all_deltas[t]])
PY

Check outputs
-------------

    find seed11 seed101 seed1001 -type f \\( -name "*.png" -o -name "*.csv" \\) | sort

Notes
-----
- This script uses relative paths only and does not expose local user or cluster paths.
- The diagnostic CSVs are not included in the public repository.
- Output files are written into `seed11/`, `seed101/`, and `seed1001/`.
"""

#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def get_valid_mask(df, t):
    if t == "long_stay" and "long_stay_defined" in df.columns:
        return df["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
    return ~pd.to_numeric(df[t], errors="coerce").isna().to_numpy(dtype=bool)

def rolling_mean_masked(y, valid, win):
    y = np.asarray(y, dtype=float)
    v = np.asarray(valid, dtype=float)
    yv = pd.Series(y * v).rolling(window=win, center=True, min_periods=1).sum()
    vv = pd.Series(v).rolling(window=win, center=True, min_periods=1).sum()
    return (yv / (vv + 1e-12)).to_numpy()

def curve_bins(s, y, valid, n_bins):
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    v = np.asarray(valid, bool)
    s_v = s[v]
    y_v = y[v]
    if s_v.size < 10:
        return pd.DataFrame(columns=["score", "p", "n"])

    # quantile bins
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(s_v, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        # fallback: uniform bins
        edges = np.linspace(s_v.min(), s_v.max(), n_bins + 1)

    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s_v >= a) & (s_v < b) if b < edges[-1] else (s_v >= a) & (s_v <= b)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "score": float(np.mean(s_v[m])),
            "p": float(np.mean(y_v[m])),
            "n": n
        })
    return pd.DataFrame(rows)

def curve_rolling(s, y, valid, win_frac):
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    v = np.asarray(valid, bool)

    # sort by score
    order = np.argsort(s)
    s = s[order]
    y = y[order]
    v = v[order]

    n = len(s)
    win = max(50, int(round(win_frac * n)))
    p = rolling_mean_masked(y, v, win)
    # keep only positions with some validity in window (rough)
    return pd.DataFrame({"score": s, "p": p, "n": np.nan})

def curve_isotonic(s, y, valid, grid_n):
    from sklearn.isotonic import IsotonicRegression
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    v = np.asarray(valid, bool)

    s_v = s[v]
    y_v = y[v]
    if s_v.size < 10:
        return pd.DataFrame(columns=["score", "p", "n"])

    # fit isotonic (monotone) on valid rows
    order = np.argsort(s_v)
    s_v = s_v[order]
    y_v = y_v[order]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s_v, y_v)

    # grid = score quantiles for stable spacing
    qs = np.linspace(0, 1, grid_n)
    grid = np.quantile(s_v, qs)
    p = iso.predict(grid)
    return pd.DataFrame({"score": grid.astype(float), "p": p.astype(float), "n": np.nan})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--score_col", default="learned_score")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--method", choices=["bins", "rolling", "isotonic"], default="bins")
    ap.add_argument("--n_bins", type=int, default=60)
    ap.add_argument("--win_frac", type=float, default=0.01)
    ap.add_argument("--grid_n", type=int, default=200)
    ap.add_argument("--intersection", action="store_true",
                    help="If set: restrict to rows where ALL targets are valid (same patient set).")
    ap.add_argument("--out", default="risk_curves.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.score_col not in df.columns:
        raise ValueError(f"Missing score_col={args.score_col}")

    # build masks
    masks = {t: get_valid_mask(df, t) for t in args.targets}
    if args.intersection:
        keep = np.ones(len(df), dtype=bool)
        for t in args.targets:
            keep &= masks[t]
        df = df.loc[keep].reset_index(drop=True)
        masks = {t: get_valid_mask(df, t) for t in args.targets}

    s = pd.to_numeric(df[args.score_col], errors="coerce").to_numpy(dtype=float)

    out_rows = []
    for t in args.targets:
        y = pd.to_numeric(df[t], errors="coerce").fillna(0).to_numpy(dtype=float)
        valid = masks[t]

        if args.method == "bins":
            cur = curve_bins(s, y, valid, args.n_bins)
        elif args.method == "rolling":
            cur = curve_rolling(s, y, valid, args.win_frac)
        else:
            cur = curve_isotonic(s, y, valid, args.grid_n)

        cur.insert(0, "task", t)
        cur.insert(1, "method", args.method)
        out_rows.append(cur)

    out_df = pd.concat(out_rows, axis=0, ignore_index=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    print(out_df.groupby(['task','method']).size())

if __name__ == "__main__":
    main()
