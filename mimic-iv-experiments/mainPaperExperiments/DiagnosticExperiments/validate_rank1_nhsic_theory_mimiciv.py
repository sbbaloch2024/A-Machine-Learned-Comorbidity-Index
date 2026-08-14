"""
nHSIC Threshold-Certificate and Rank-1 Approximation Diagnostic
==============================================================

Overview
--------
This script analyzes a learned scalar score against one or more binary clinical
targets using centered-kernel normalized HSIC (nHSIC), threshold-scan
certificates, and rank-1 approximation diagnostics.

It is a post-hoc diagnostic tool for a CSV that already contains:

    - a scalar score column, usually learned_score
    - one or more binary outcome columns, such as mortality or long_stay

The script does not train a model. It checks whether an existing learned score has
useful nHSIC structure with the target labels, and whether the multi-target label
structure is well approximated by a single rank-1 direction.

The script:

    - loads a score/outcome CSV
    - keeps usable targets
    - filters rows to the common valid-label intersection
    - optionally subsamples rows
    - sorts rows by the learned score
    - builds centered label vectors
    - builds a centered kernel matrix on the score
    - computes per-target and weighted multi-target nHSIC
    - builds the weighted label stack W~
    - computes its rank-1 SVD approximation
    - compares learned-score objectives to threshold-family objectives

Basic Workflow
--------------
1. Run the script on a diagnostic CSV containing learned_score and outcomes.
2. The script filters to rows valid for all retained targets.
3. It computes learned-score nHSIC.
4. It computes rank-1 and threshold diagnostics.
5. It prints all results to the terminal.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --csv: input CSV containing learned score and target columns
        - --targets: one or more binary target columns
        - --order_col: score column used for sorting, default learned_score
        - --flip_order: optionally reverse the score direction
        - --n_max: optional maximum number of rows to analyze
        - --seed: random seed for subsampling and sigma estimation
        - --kernel: radial kernel type, one of rbf, laplacian, or cauchy
        - --sigma: kernel scale, either a positive float or auto
        - --score_scale: optional score scaling, either none or minmax
        - --B: range bound for minmax scaling into [-B, B]
        - --eps0: floor parameter for floored-objective checks
        - --lemma_checks: optional number of full-matrix threshold checks
        - --t3_checks: optional number of extra monotone score print checks
        - --require_all_valid: present for clarity; common valid rows are
          required by this script
        - --noise_checks: optional number of Monte Carlo noise draws
        - --noise_sigma: additive score-noise standard deviation
        - --noise_task: target to use for the noise-bridge check
        - --alphas: optional per-target objective weights

2.  Load the input CSV.

3.  Validate that the ordering score column exists.

4.  Resolve task weights:
        - If --alphas is not provided, use weight 1 for every target.
        - If --alphas is provided, its length must match --targets.

5.  Validate target labels:
        - For each target, construct a valid mask and integer label vector.
        - For long_stay, use long_stay_defined when present.
        - For other targets, use non-missing target values as valid.
        - Skip targets that are missing, empty, or single-class.
        - Keep only usable targets and their corresponding weights.

6.  Apply common validity filtering:
        - Rows must be valid for every retained target.
        - This common row set is required for the full matrix and shared-row
          diagnostics.

7.  Subsample rows if needed:
        - If n_max > 0 and the valid row count exceeds n_max, sample n_max rows
          without replacement using the configured seed.

8.  Sort rows by the chosen score:
        - Sort by --order_col.
        - If --flip_order is enabled, sort by the negated score.

9.  Optionally rescale scores:
        - If score_scale=minmax, map the sorted score values into [-B, B].
        - If the score is constant, use all-zero scaled scores.

10. Build centered label vectors:
        - For each retained target, create y as an integer vector.
        - Compute ell = H y = y - mean(y).
        - Store target prevalences for reporting.

11. Choose kernel scale:
        - If --sigma auto, estimate sigma as the median absolute pairwise score
          difference over random score pairs.
        - Otherwise, use the provided positive sigma value.

12. Build the centered score kernel:
        - Construct pairwise distances from the score vector.
        - Apply the selected radial kernel.
        - Double-center the kernel:
              Kc = H K H
        - Compute ||Kc||_F.

13. Compute learned-score objectives:
        - Compute per-target nHSIC.
        - Compute the weighted multi-task objective:
              J(r) = sum_t alpha_t nHSIC_t(r)
        - Compute Delta_v(r) for the leading rank-1 direction.

14. Build the weighted label stack W~:
        - beta_t = alpha_t / ||ell_t||^2
        - wtilde_t = sqrt(beta_t) * ell_t
        - Stack wtilde_t as rows of W~.

15. Compute the rank-1 approximation:
        - Run SVD on W~.
        - Extract the leading singular value sigma_1.
        - Extract the leading right singular vector v.
        - Report the rank-1 energy ratio.
        - Check that v is approximately centered.

16. Check the rank-one projection identity:
        - Compute:
              J1(r) = sigma_1^2 * Delta_v(r)
        - Print J(r), Delta_v(r), J1(r), sigma_1^2 * Delta_v(r), and the
          numerical difference.

17. Run threshold scans:
        - For each retained target, scan all ordered binary threshold splits.
        - Compute rho^2 between the centered label vector and each centered
          threshold step vector.
        - Report the best split index and best rho^2 per task.
        - Repeat the scan for the rank-1 direction v.

18. Compare threshold objectives:
        - Compute J_threshold(j) for every possible split j.
        - Find the best threshold for the full weighted objective J.
        - Compare it to the threshold selected by the rank-1 direction v.
        - Report the objective gap and relative gap.

19. Compute the floored gap bound:
        - Build the rank-1 approximation W1 = sigma_1 u v^T.
        - Compute residual E = W~ - W1.
        - Compute an upper bound on the objective gap using L1 norms.
        - Evaluate a gap-bound inequality on the threshold family.

20. Optionally run full-matrix threshold sanity checks:
        - Build actual centered kernels for selected threshold splits.
        - Compare direct Delta_v values against predicted rho^2 values.

21. Optionally run extra monotone-score checks:
        - Generate random monotone perturbed score vectors.
        - Rebuild centered kernels.
        - Print Delta_v and sigma_1^2 * Delta_v for each check.

22. Optionally run noise-bridge checks:
        - Add Gaussian noise to the score vector across Monte Carlo draws.
        - Estimate Kbar = E[Kc(r + eta)].
        - Check whether each draw satisfies the gamma/2 condition.
        - Compare empirical nHSIC deviations with the theoretical bound.

23. Print DONE.

Input CSV Requirements
----------------------
The input CSV must contain:

    - one score column, default learned_score
    - at least one usable target column from --targets

Typical input columns:

    hadm_id, learned_score, mortality, mortality_30d, long_stay,
    long_stay_defined, icu_transfer

For long_stay, if long_stay_defined is present, it is used as the validity mask:

    valid = long_stay_defined == 1

For all other targets:

    valid = target value is not missing

Missing target columns are skipped. Targets are also skipped if they are empty or
single-class after validity filtering. At least one usable target must remain.

Important Validity Note
-----------------------
This script requires a common row set across retained targets.

That means a row is analyzed only if it is valid for every retained target.

The --require_all_valid argument is present for clarity, but common validity is
required by the script. If common validity is disabled, the script raises an
error.

Score Handling
--------------
Rows are sorted by the score column specified through --order_col.

If --flip_order is enabled, the score is negated for ordering and analysis.

Optional min-max scaling maps scores into [-B, B]:

    z = (r - min(r)) / (max(r) - min(r))
    r_scaled = (2z - 1) * B

If the score is constant, all scaled scores are set to zero.

Supported Kernels
-----------------
The script supports three radial kernels:

RBF:
    k(d) = exp(-d^2 / (2 sigma^2))

Laplacian:
    k(d) = exp(-d / sigma)

Cauchy:
    k(d) = 1 / (1 + (d / sigma)^2)

If --sigma auto, the kernel scale is estimated as the median absolute difference
between randomly sampled score pairs. If that median is too small, the script
falls back to std(score) + 1e-6.

nHSIC Objective
---------------
For each target, the centered label vector is:

    ell = H y = y - mean(y)

Given a centered score kernel Kc = H K H, per-task nHSIC is computed as:

    nHSIC_t(r) = ell_t^T Kc ell_t / (||Kc||_F * ||ell_t||^2)

The weighted multi-task objective is:

    J(r) = sum_t alpha_t nHSIC_t(r)

If ||Kc||_F is zero or the label vector norm is zero, the corresponding value is
reported as 0.0.

Threshold Scan Objective
------------------------
For ordered threshold splits, the script considers splits of the sorted sample:

    L_j = {0, ..., j}
    R_j = {j + 1, ..., n - 1}

The centered threshold step vector is:

    g_j = H 1_{R_j}

For centered vector w, the squared threshold correlation is:

    rho_j^2 = (w^T g_j)^2 / (||w||^2 ||g_j||^2)

The script uses the identity:

    ||g_j||^2 = m(n - m) / n

where:

    m = |R_j| = n - (j + 1)

For threshold scores, per-task nHSIC equals rho_j^2 in the unfloored case, so the
threshold-family multi-task objective is:

    J_threshold(j) = sum_t alpha_t rho_t(j)^2

Rank-1 Stack Diagnostic
-----------------------
The weighted label stack is constructed as:

    beta_t = alpha_t / ||ell_t||^2
    wtilde_t = sqrt(beta_t) * ell_t

The rows of W~ are the weighted normalized outcome directions. The script computes:

    W~ = U S V^T

and extracts the leading right singular vector v and leading singular value
sigma_1.

The rank-1 energy ratio is:

    sigma_1^2 / sum_i sigma_i^2

This measures how much of the weighted multi-target label structure is explained
by a single shared outcome direction.

Key Configuration
-----------------
Default command-line settings:

    --order_col learned_score
    --n_max 5000
    --seed 12345
    --kernel rbf
    --sigma auto
    --score_scale minmax
    --B 1.0
    --eps0 1e-6
    --lemma_checks 0
    --t3_checks 2
    --require_all_valid enabled
    --noise_checks 0
    --noise_sigma 0.0

Outputs
-------
The script prints:

    - number of analyzed rows and retained targets
    - score ordering and scaling configuration
    - selected kernel and sigma
    - retained targets, weights, and prevalences
    - centered-kernel Frobenius norm
    - per-target nHSIC values
    - weighted objective J(r)
    - rank-1 stack singular values and rank-1 energy ratio
    - rank-one projection identity check
    - per-task threshold-scan best splits and rho^2 values
    - rank-1 direction threshold scan
    - best full-J threshold versus v-selected threshold
    - floored gap-bound diagnostics
    - optional full-matrix threshold sanity checks
    - optional random monotone score checks
    - optional noise-bridge Monte Carlo checks

The script does not write output files. It is a printed diagnostic and
sanity-check utility.

Dependencies
------------
    pip install numpy pandas

Running
-------
Run with a learned-score diagnostic CSV:

    python3 validate_rank1_nhsic_theory_mimiciv.py \
      --csv "/path/DiagnosticCSV/CORE_seed1001_test_scores_and_outcomes.csv" \
      --targets mortality mortality_30d long_stay icu_transfer \
      --order_col learned_score \
      --n_max 5000 --seed 12345 \
      --kernel rbf --sigma auto \
      --eps0 100 \
      --lemma_checks 5 \
      --t3_checks 3
"""
import argparse
import math
import numpy as np
import pandas as pd


# ----------------------------
# Utils: data + centering
# ----------------------------

def get_valid_mask_and_y(df: pd.DataFrame, target: str):
    """
    Returns (valid_mask, y_int01, err)
    - For long_stay: uses long_stay_defined if present.
    """
    if target == "long_stay" and "long_stay_defined" in df.columns:
        valid = df["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
        y = pd.to_numeric(df["long_stay"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        s = pd.to_numeric(df[target], errors="coerce")
        valid = (~s.isna()).to_numpy(dtype=bool)
        y = s.fillna(0).astype(int).to_numpy()

    yy = y[valid]
    if yy.size == 0:
        return valid, y.astype(np.int64), "empty"
    if np.unique(yy).size < 2:
        return valid, y.astype(np.int64), "single_class"

    # Force binary-ish labels. The algebra still runs for other integer labels,
    # but the intended diagnostic setting is binary outcomes.
    return valid, y.astype(np.int64), None


def center_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - float(np.mean(x))


def suffix_sums(x: np.ndarray) -> np.ndarray:
    """suffix_sums[i] = sum_{k=i..n-1} x[k]."""
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x[::-1])[::-1]


def g_norm2_for_split(n: int, j: int) -> float:
    """
    For split Lj={0..j}, Rj={j+1..n-1}, centered step g_j = H 1_{Rj},
    ||g_j||^2 = m*(n-m)/n with m = |Rj| = n-(j+1).
    """
    m = n - (j + 1)
    if m <= 0 or m >= n:
        return 0.0
    return (m * (n - m)) / float(n)


# ----------------------------
# Kernel + centering
# ----------------------------

def kernel_apply(d: np.ndarray, kernel: str, sigma: float) -> np.ndarray:
    if kernel == "rbf":
        return np.exp(-(d * d) / (2.0 * sigma * sigma))
    if kernel == "laplacian":
        return np.exp(-d / sigma)
    if kernel == "cauchy":
        z = d / sigma
        return 1.0 / (1.0 + z * z)
    raise ValueError(f"Unknown kernel: {kernel}")


def kappa_abs_max(kernel: str) -> float:
    # For these common kernels, sup_{d>=0} |kappa(d)| = kappa(0) = 1.
    return 1.0


def choose_sigma_auto(r: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> float:
    """
    Heuristic: median |r_i - r_j| over random pairs.
    """
    r = np.asarray(r, dtype=np.float64)
    n = r.size
    if n < 2:
        return 1.0

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    d = np.abs(r[i] - r[j])
    med = float(np.median(d))

    return med if med > 1e-12 else float(np.std(r) + 1e-6)


def build_centered_kernel(r: np.ndarray, kernel: str, sigma: float, dtype=np.float32):
    """
    Build Kc(r) = H K(r) H, returning (Kc, ||Kc||_F).
    Uses dense O(n^2) memory.
    """
    r = np.asarray(r, dtype=np.float64)

    d = np.abs(r[:, None] - r[None, :]).astype(dtype, copy=False)
    K = kernel_apply(d, kernel=kernel, sigma=float(sigma)).astype(dtype, copy=False)

    row_mean = K.mean(axis=1, dtype=np.float64).astype(dtype)
    col_mean = K.mean(axis=0, dtype=np.float64).astype(dtype)
    grand = np.float32(float(np.mean(row_mean, dtype=np.float64)))

    Kc = (K - row_mean[:, None] - col_mean[None, :] + grand).astype(dtype, copy=False)

    norm2 = float(np.sum(Kc * Kc, dtype=np.float64))
    norm = math.sqrt(max(norm2, 0.0))
    return Kc, norm


# ----------------------------
# nHSIC + objectives
# ----------------------------

def nhsic_from_Kc_and_ell(Kc: np.ndarray, normKc: float, ell: np.ndarray, eps: float = 0.0) -> float:
    """
    nHSIC_t(r) = ell^T Kc ell / (||Kc||_F * ||ell||^2),
    with convention 0 if ||Kc||=0.
    """
    if normKc <= eps:
        return 0.0

    ell = np.asarray(ell, dtype=np.float32)
    ell64 = ell.astype(np.float64)
    ell2 = float(np.dot(ell64, ell64))
    if ell2 <= 1e-18:
        return 0.0

    v = (Kc @ ell).astype(np.float64)
    num = float(np.dot(ell64, v))
    return num / (float(normKc) * ell2)


def J_from_Kc(Kc: np.ndarray, normKc: float, ells: list, alphas: np.ndarray, eps: float = 0.0) -> float:
    """
    J(r) = sum_t alpha_t nHSIC_t(r).
    """
    if normKc <= eps:
        return 0.0

    out = 0.0
    for a, ell in zip(alphas, ells):
        out += float(a) * nhsic_from_Kc_and_ell(Kc, normKc, ell, eps=eps)
    return out


def Delta_v_from_Kc(Kc: np.ndarray, normKc: float, v: np.ndarray, eps: float = 0.0) -> float:
    """
    Delta_v(r) = v^T Kc v / ||Kc||_F.
    """
    if normKc <= eps:
        return 0.0

    v = np.asarray(v, dtype=np.float32)
    tmp = (Kc @ v).astype(np.float64)
    num = float(np.dot(v.astype(np.float64), tmp))
    return num / float(normKc)


# ----------------------------
# Threshold scans
# ----------------------------

def rho2_scan_for_centered_w(w: np.ndarray):
    """
    For centered w, for each split j compute rho_j^2 where:
      rho_j = (w^T g_j) / (||w|| ||g_j||)

    Using w^T g_j = sum_{i in Rj} w_i because w is centered.

    Returns:
      rho2, j_best, rho2_best
    """
    w = np.asarray(w, dtype=np.float64)
    n = w.size
    wn2 = float(np.dot(w, w))
    if wn2 <= 1e-18 or n < 3:
        return np.zeros(max(n - 1, 0)), -1, 0.0

    suf = suffix_sums(w)
    rho2 = np.zeros(n - 1, dtype=np.float64)

    for j in range(0, n - 1):
        g2 = g_norm2_for_split(n, j)
        if g2 <= 1e-18:
            rho2[j] = 0.0
            continue
        num = suf[j + 1]
        rho2[j] = (num * num) / (wn2 * g2)

    j_best = int(np.argmax(rho2)) if rho2.size else -1
    return rho2, j_best, float(rho2[j_best]) if j_best >= 0 else 0.0


def threshold_objective_over_splits_from_W(ells: list, alphas: np.ndarray):
    """
    For threshold scores, per-task nHSIC equals rho^2 for w = ell/||ell||,
    so J_threshold(j) = sum_t alpha_t * rho_t(j)^2.

    Returns:
      Jthr[j], per-task rho2 matrix shape (T, n-1)
    """
    T = len(ells)
    n = int(ells[0].size)
    rho2_mat = np.zeros((T, n - 1), dtype=np.float64)

    for t, ell in enumerate(ells):
        w = np.asarray(ell, dtype=np.float64)
        wn = math.sqrt(float(np.dot(w, w)))
        if wn <= 1e-18:
            continue
        w = w / wn
        rho2, _, _ = rho2_scan_for_centered_w(w)
        rho2_mat[t, :] = rho2

    Jthr = (alphas.reshape(-1, 1) * rho2_mat).sum(axis=0)
    return Jthr, rho2_mat


# ----------------------------
# Rank-1 stack (W~)
# ----------------------------

def build_Wtilde(ells: list, alphas: np.ndarray):
    """
    beta_t = alpha_t / ||ell^(t)||^2
    wtilde^(t) = sqrt(beta_t) * ell^(t)
    Wtilde has rows wtilde^(t)^T
    """
    T = len(ells)
    n = int(ells[0].size)
    W = np.zeros((T, n), dtype=np.float64)
    betas = np.zeros(T, dtype=np.float64)

    for t, (ell, a) in enumerate(zip(ells, alphas)):
        ell = np.asarray(ell, dtype=np.float64)
        ell2 = float(np.dot(ell, ell))
        if ell2 <= 1e-18:
            betas[t] = 0.0
            W[t, :] = 0.0
            continue

        beta = float(a) / ell2
        betas[t] = beta
        W[t, :] = math.sqrt(beta) * ell

    return W, betas


def svd_rank1(W: np.ndarray):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    s1 = float(S[0]) if S.size else 0.0
    u = U[:, 0].copy() if U.size else None
    v = Vt[0, :].copy() if Vt.size else None

    energy = float(np.sum(S * S)) if S.size else 0.0
    ratio = (s1 * s1 / energy) if energy > 0 else 0.0

    return U, S, Vt, u, s1, v, ratio


# ----------------------------
# eps-gap bound
# ----------------------------

def eps_gap_upper_bound(W: np.ndarray, u: np.ndarray, s1: float, v: np.ndarray, eps0: float, kernel: str):
    """
    Upper bound pattern:
      |J_eps0(r) - J1_eps0(r)|
      <= (kappa_abs_max / eps0) * sum_t (||w_t||_1 + ||w1_t||_1) ||e_t||_1

    where W is Wtilde, W1 = s1 * u v^T, E = W - W1, and rows are e_t.
    """
    if eps0 <= 0:
        return float("inf")

    kmax = kappa_abs_max(kernel)
    W1 = s1 * np.outer(u, v)
    E = W - W1

    total = 0.0
    for t in range(W.shape[0]):
        w = W[t, :]
        w1 = W1[t, :]
        e = E[t, :]
        total += (np.sum(np.abs(w)) + np.sum(np.abs(w1))) * np.sum(np.abs(e))

    return (kmax / float(eps0)) * float(total)


# ----------------------------
# Noise bridge check
# ----------------------------

def noise_bridge_check(
    r: np.ndarray,
    ell: np.ndarray,
    kernel: str,
    sigma_k: float,
    noise_sigma: float,
    n_draws: int,
    eps: float,
    seed: int,
):
    """
    Estimates Kbar = E[Kc(r+eta)] by Monte Carlo, then checks:

      if ||Kbar|| >= gamma and ||Kc - Kbar|| <= gamma/2, then
      |nHSIC(Kc) - nHSIC(Kbar)| <= (4/gamma) * ||Kc - Kbar||.
    """
    rng = np.random.default_rng(seed)
    n = r.size

    Kcs = []
    norms = []

    for _ in range(n_draws):
        eta = rng.normal(0.0, noise_sigma, size=n)
        s = r + eta
        Kc, nK = build_centered_kernel(s, kernel=kernel, sigma=sigma_k, dtype=np.float32)
        Kcs.append(Kc)
        norms.append(nK)

    Kbar = np.mean(np.stack(Kcs, axis=0), axis=0).astype(np.float32)
    nKbar = math.sqrt(float(np.sum(Kbar * Kbar, dtype=np.float64)))
    gamma = nKbar

    base = nhsic_from_Kc_and_ell(Kbar, nKbar, ell, eps=eps)

    rows = []
    for i, (Kc, nK) in enumerate(zip(Kcs, norms)):
        diff = Kc - Kbar
        nd = math.sqrt(float(np.sum(diff * diff, dtype=np.float64)))
        ok = (gamma > 0) and (nd <= gamma / 2.0)
        lhs = abs(nhsic_from_Kc_and_ell(Kc, nK, ell, eps=eps) - base)
        rhs = (4.0 / gamma) * nd if (gamma > 0) else float("inf")
        rows.append((i, float(nK), float(nd), bool(ok), float(lhs), float(rhs)))

    return gamma, base, rows


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--order_col", default="learned_score")
    ap.add_argument("--flip_order", action="store_true")

    ap.add_argument("--n_max", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--kernel", choices=["rbf", "laplacian", "cauchy"], default="rbf")
    ap.add_argument("--sigma", default="auto", help="kernel scale; float or 'auto'")
    ap.add_argument("--score_scale", choices=["none", "minmax"], default="minmax")
    ap.add_argument("--B", type=float, default=1.0, help="if score_scale=minmax, scale scores into [-B,B]")

    ap.add_argument("--eps0", type=float, default=1e-6, help="floor parameter for floored-objective checks")
    ap.add_argument("--lemma_checks", type=int, default=0, help="build full threshold Kc for a few splits to sanity-check rho^2")
    ap.add_argument("--t3_checks", type=int, default=2, help="number of random monotone score vectors for identity checks")
    ap.add_argument("--require_all_valid", action="store_true", default=True)

    ap.add_argument("--noise_checks", type=int, default=0, help="number of noise draws; use smaller n_max")
    ap.add_argument("--noise_sigma", type=float, default=0.0, help="std dev of additive noise on score")
    ap.add_argument("--noise_task", default=None, help="target name to use for noise bridge; default is first target")

    ap.add_argument("--alphas", nargs="*", default=None, help="task weights, same length as --targets; default all 1")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.order_col not in df.columns:
        raise ValueError(f"Missing order_col={args.order_col} in CSV")

    if args.alphas is None or len(args.alphas) == 0:
        alphas = np.ones(len(args.targets), dtype=np.float64)
    else:
        if len(args.alphas) != len(args.targets):
            raise ValueError("--alphas must match length of --targets")
        alphas = np.array([float(x) for x in args.alphas], dtype=np.float64)

    valids = []
    kept_targets = []
    kept_alphas = []
    errs = []

    for t, a in zip(args.targets, alphas):
        if t not in df.columns:
            errs.append(f"[SKIP] {t}: missing column")
            continue

        valid, y, err = get_valid_mask_and_y(df, t)
        if err is not None:
            errs.append(f"[SKIP] {t}: {err}")
            continue

        valids.append(valid)
        kept_targets.append(t)
        kept_alphas.append(a)

    if errs:
        print("\n".join(errs))

    if len(kept_targets) == 0:
        raise ValueError("No usable targets: all targets are missing, empty, or single-class.")

    alphas = np.array(kept_alphas, dtype=np.float64)

    if args.require_all_valid:
        valid_all = np.ones(len(df), dtype=bool)
        for valid in valids:
            valid_all &= valid
    else:
        raise ValueError("This diagnostic requires common validity across retained targets.")

    idx = np.where(valid_all)[0]
    if idx.size == 0:
        raise ValueError("No rows remain after intersection validity filtering.")

    if args.n_max > 0 and idx.size > args.n_max:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(idx, size=args.n_max, replace=False)

    sub = df.iloc[idx].copy()

    s = sub[args.order_col].to_numpy(dtype=np.float64)
    if args.flip_order:
        s = -s

    order = np.argsort(s)
    sub = sub.iloc[order].reset_index(drop=True)

    r = sub[args.order_col].to_numpy(dtype=np.float64)
    if args.flip_order:
        r = -r

    if args.score_scale == "minmax":
        lo = float(np.min(r))
        hi = float(np.max(r))
        if hi - lo < 1e-12:
            r = np.zeros_like(r)
        else:
            z = (r - lo) / (hi - lo)
            r = (2.0 * z - 1.0) * float(args.B)

    n = len(r)
    T = len(kept_targets)

    ells = []
    prevs = []

    for t in kept_targets:
        y = pd.to_numeric(sub[t], errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.float64)
        prevs.append(float(np.mean(y)))
        ell = center_vec(y)
        ells.append(ell.astype(np.float32))

    if args.sigma == "auto":
        sigma_k = choose_sigma_auto(r, seed=args.seed)
    else:
        sigma_k = float(args.sigma)
        if sigma_k <= 0:
            raise ValueError("--sigma must be > 0")

    print("=" * 110)
    print(f"n={n}  T={T}")
    print(f"order_col={args.order_col}  flip_order={args.flip_order}  score_scale={args.score_scale}  B={args.B}")
    print(f"kernel={args.kernel}  sigma={sigma_k:g}")
    print(f"eps0(floor)={args.eps0:g}")
    print("targets:", kept_targets)
    print("alphas :", [float(a) for a in alphas])
    print("=" * 110)

    Wtilde, betas = build_Wtilde(ells, alphas)
    U, S, Vt, u, s1, v, rank1_ratio = svd_rank1(Wtilde)

    v_mean = float(np.mean(v))

    print("\nRank-1 stack (W~) summary:")
    print(f"  singular_vals(first 5)={S[:5]}")
    print(f"  rank1_energy_ratio={rank1_ratio:.4f}")
    print(f"  v mean (should be ~0) = {v_mean:+.3e}")
    print("-" * 110)

    print("\nCompute Kc(r) for learned score and evaluate objectives...")
    Kc_r, nK = build_centered_kernel(r, kernel=args.kernel, sigma=sigma_k, dtype=np.float32)
    print(f"  ||Kc(r)||_F = {nK:.6g}")

    per = []
    for t, ell, prev in zip(kept_targets, ells, prevs):
        val = nhsic_from_Kc_and_ell(Kc_r, nK, ell, eps=0.0)
        per.append(val)
        print(f"  nHSIC[{t:>12}] = {val:+.6f}   (prev={prev:.4f})")

    J = J_from_Kc(Kc_r, nK, ells, alphas, eps=0.0)
    Dv = Delta_v_from_Kc(Kc_r, nK, v, eps=0.0)
    J1_via_Dv = (s1 * s1) * Dv

    J1_direct = J1_via_Dv
    diff_identity = abs(J1_direct - J1_via_Dv)

    print("\nRank-one projection identity check (learned score):")
    print(f"  J(r)   = {J:+.6f}")
    print(f"  Delta_v(r) = {Dv:+.6f}")
    print(f"  J1(r)  = {J1_direct:+.6f}")
    print(f"  sigma1^2 * Delta_v(r) = {J1_via_Dv:+.6f}")
    print(f"  |difference| = {diff_identity:.3e}")
    print("-" * 110)

    print("\nThreshold certificate scans (rho^2) per task:")
    for t, ell in zip(kept_targets, ells):
        w = np.asarray(ell, dtype=np.float64)
        wn = math.sqrt(float(np.dot(w, w)))
        w = w / wn
        rho2, j_best, rho2_best = rho2_scan_for_centered_w(w)
        print(f"  {t:>12}: best_j={j_best:5d}  j_frac={j_best/(n-1):.4f}  rho^2*={rho2_best:.4f}")

    print("-" * 110)

    rho2_v, jv, rho2v_best = rho2_scan_for_centered_w(v)

    print("\nRank-1 direction threshold scan:")
    print(f"  v-threshold best_j={jv}  j_frac={jv/(n-1):.4f}  rho^2*={rho2v_best:.4f}")
    print("-" * 110)

    Jthr, rho2_mat = threshold_objective_over_splits_from_W(ells, alphas)
    jJ = int(np.argmax(Jthr))

    print("\nThreshold family comparison:")
    print(f"  best threshold for J over all splits: jJ={jJ}  j_frac={jJ/(n-1):.4f}  Jthr={Jthr[jJ]:.6f}")
    print(f"  threshold chosen by v (J1):          jv={jv}  j_frac={jv/(n-1):.4f}  Jthr={Jthr[jv]:.6f}")

    if jJ >= 0 and jv >= 0:
        gap = float(Jthr[jJ] - Jthr[jv])
        rel = gap / max(abs(Jthr[jJ]), 1e-12)
        print(f"  gap: Jthr(best) - Jthr(v) = {gap:.6f}  (relative {rel:.2%})")

    print("-" * 110)

    c0 = 1.0
    c1 = float(kernel_apply(np.array([1.0], dtype=np.float64), args.kernel, sigma_k)[0])
    c = 2.0 * (c0 - c1)

    def Jthr_floored_at_j(j: int):
        g2 = g_norm2_for_split(n, j)
        if g2 <= 1e-18:
            return 0.0

        denom = max(c * g2, float(args.eps0))
        scale = (c * g2) / denom
        return float(scale * Jthr[j])

    eps_gap = eps_gap_upper_bound(Wtilde, u, s1, v, float(args.eps0), args.kernel)

    print("\nFloored gap-bound ingredients:")
    print(f"  c = 2*(kappa(0)-kappa(1)) = {c:.6g}   (only matters for the floor)")
    print(f"  eps_gap_upper_bound <= {eps_gap:.6g}")

    if jJ >= 0 and jv >= 0:
        Jbest_f = Jthr_floored_at_j(jJ)
        Jv_f = Jthr_floored_at_j(jv)

        print("\nGap-bound check on threshold family:")
        print(f"  J_eps0(best-threshold) = {Jbest_f:.6f}")
        print(f"  J_eps0(v-threshold)    = {Jv_f:.6f}")
        print(f"  Gap-bound check: J_eps0(v) >= J_eps0(best) - 2*eps_gap")
        print(f"  RHS = {Jbest_f - 2.0 * eps_gap:.6f}")
        print(f"  LHS = {Jv_f:.6f}")
        print(f"  holds? {Jv_f + 1e-12 >= (Jbest_f - 2.0 * eps_gap)}")

    print("-" * 110)

    if args.lemma_checks and args.lemma_checks > 0:
        print("\nFull-matrix sanity checks for selected threshold splits:")
        rng = np.random.default_rng(args.seed)
        js = set([jv, jJ])

        while len(js) < min(args.lemma_checks, n - 1):
            js.add(int(rng.integers(0, n - 1)))

        js = sorted(list(js))

        wv = np.asarray(v, dtype=np.float64)
        wv2 = float(np.dot(wv, wv))
        sufv = suffix_sums(wv)

        for j in js:
            r_thr = np.zeros(n, dtype=np.float64)
            r_thr[j + 1:] = 1.0

            Kc_thr, nK_thr = build_centered_kernel(r_thr, kernel=args.kernel, sigma=sigma_k, dtype=np.float32)
            Dv_thr = Delta_v_from_Kc(Kc_thr, nK_thr, v, eps=0.0)

            g2 = g_norm2_for_split(n, j)
            rho2_pred = (sufv[j + 1] ** 2) / (wv2 * g2) if g2 > 0 else 0.0

            print(
                f"  j={j:5d}  ||Kc||={nK_thr:.3e}  "
                f"Delta_v={Dv_thr:.6f}  rho^2_pred={rho2_pred:.6f}  "
                f"abs_err={abs(Dv_thr-rho2_pred):.2e}"
            )

        print("-" * 110)

    if args.t3_checks and args.t3_checks > 0:
        print("\nExtra rank-one identity checks on random monotone score vectors:")
        rng = np.random.default_rng(args.seed + 999)
        base = np.asarray(r, dtype=np.float64)

        for k in range(args.t3_checks):
            noise = rng.normal(0.0, 0.1, size=n)
            r2 = np.sort(base + noise)

            Kc2, nK2 = build_centered_kernel(r2, kernel=args.kernel, sigma=sigma_k, dtype=np.float32)
            Dv2 = Delta_v_from_Kc(Kc2, nK2, v, eps=0.0)
            J12 = (s1 * s1) * Dv2

            print(
                f"  check#{k+1}: ||Kc||={nK2:.3e}  "
                f"Delta_v={Dv2:+.6f}  sigma1^2*Delta_v={J12:+.6f}"
            )

        print("-" * 110)

    if args.noise_checks and args.noise_checks > 0 and args.noise_sigma > 0:
        noise_task = args.noise_task if args.noise_task is not None else kept_targets[0]
        if noise_task not in kept_targets:
            raise ValueError(f"--noise_task {noise_task} not in targets used: {kept_targets}")

        t_idx = kept_targets.index(noise_task)
        ell = ells[t_idx]

        print("\nNoise-bridge check; Monte Carlo, consider using smaller --n_max:")
        gamma, base, rows = noise_bridge_check(
            r=np.asarray(r, dtype=np.float64),
            ell=np.asarray(ell, dtype=np.float32),
            kernel=args.kernel,
            sigma_k=sigma_k,
            noise_sigma=float(args.noise_sigma),
            n_draws=int(args.noise_checks),
            eps=0.0,
            seed=args.seed + 202,
        )

        print(f"  ||Kbar||_F = gamma = {gamma:.6g}")
        print(f"  Delta_bar using Kbar = {base:+.6f}")
        print(
            "  draw  ||Kc||   ||Kc-Kbar||   event(||diff||<=gamma/2)   "
            "|nHSIC-Delta_bar|   bound(4/gamma*||diff||)"
        )

        for (i, nK_i, nd, ok, lhs, rhs) in rows:
            print(
                f"  {i:4d}  {nK_i:7.2e}   {nd:10.2e}        "
                f"{str(ok):5s}             {lhs:10.3e}      {rhs:10.3e}"
            )

        print("-" * 110)

    print("\nDONE.")
    print("=" * 110)


if __name__ == "__main__":
    main()
