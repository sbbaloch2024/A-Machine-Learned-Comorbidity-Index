"""
NHSIC Threshold Certificate and Rank-1 Approximation Checker
===========================================================

Overview
--------
This script analyzes a learned scalar score against one or more binary clinical
targets using normalized Hilbert-Schmidt Independence Criterion (nHSIC) and
threshold-certificate diagnostics. It is designed as a post-hoc diagnostic
checker for a CSV that already contains a scalar ordering column, such as
learned_score, and binary outcome columns.

The script loads a CSV, keeps usable targets, filters rows to the intersection of
valid labels across those retained targets, optionally subsamples rows, sorts
admissions by the chosen scalar score, builds centered label vectors, constructs a
centered radial kernel matrix on the score, and evaluates weighted multi-target
nHSIC objectives.

It also implements several diagnostic checks:

    - Per-task threshold scans based on rho^2.
    - Rank-1 approximation of the weighted label stack W~.
    - Rank-one projection identity check:
          J1(r) = sigma1^2 * Delta_v(r)
    - Threshold-family comparison between the best full-J threshold and the
      threshold induced by the rank-1 direction v.
    - Floored epsilon-gap upper-bound calculation.
    - Optional full-matrix threshold sanity checks.
    - Optional extra monotone-score identity checks.
    - Optional noise-bridge Monte Carlo check.

This script does not train a model. It evaluates structure in an existing learned
score or ordering variable.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --csv: input CSV containing learned score and target columns
        - --targets: one or more binary target columns
        - --order_col: scalar score column used for sorting, default learned_score
        - --flip_order: optionally reverse the score direction
        - --n_max: optional maximum number of rows after validity filtering
        - --seed: random seed for subsampling and Monte Carlo checks
        - --kernel: radial kernel type, one of rbf, laplacian, or cauchy
        - --sigma: kernel scale, either a positive float or auto
        - --score_scale: none or minmax
        - --B: score range bound when minmax scaling is used
        - --eps0: denominator floor for floored-objective checks
        - --threshold_checks: number of full threshold kernel sanity checks
        - --identity_checks: number of random monotone score checks to print
        - --require_all_valid: present for clarity; common valid rows are
          required by this script
        - --noise_checks: number of noise draws for optional noise-bridge check
        - --noise_sigma: standard deviation of additive score noise
        - --noise_task: target used for the optional noise-bridge check
        - --alphas: optional target weights

2.  Load the input CSV.

3.  Validate that the requested order_col exists.

4.  Parse target weights:
        - If --alphas is omitted, use weight 1 for every target.
        - If --alphas is provided, its length must match --targets.

5.  Build target-specific validity masks and integer labels:
        - For long_stay, use long_stay_defined when present.
        - For other targets, use non-missing target values as valid.
        - Skip targets that are missing, empty after filtering, or single-class.

6.  Apply intersection validity:
        - Rows must be valid for every retained target.
        - The script uses this common row set for all matrix calculations and
          shared-row diagnostics.

7.  Optionally subsample valid rows:
        - If n_max > 0 and more than n_max rows remain, sample n_max rows without
          replacement using the provided seed.

8.  Sort rows by the ordering score:
        - Use order_col.
        - If flip_order is enabled, sort by the negated score.

9.  Optionally scale scores:
        - If score_scale=minmax, map scores into [-B, B].
        - If the score is constant, replace the scaled score with zeros.

10. Build centered label vectors:
        - For each retained target, read binary labels on the sorted subset.
        - Compute prevalence.
        - Center the label vector:
              ell_t = H y_t = y_t - mean(y_t)

11. Choose kernel bandwidth:
        - If sigma=auto, estimate sigma using the median absolute score
          difference over random score pairs.
        - Otherwise, use the supplied positive sigma.

12. Build the centered score kernel:
        - Compute pairwise score distances.
        - Apply the selected radial kernel.
        - Double-center the kernel:
              Kc = H K H
        - Compute the Frobenius norm ||Kc||_F.

13. Evaluate learned-score nHSIC:
        - Compute per-target nHSIC.
        - Compute weighted objective:
              J(r) = sum_t alpha_t nHSIC_t(r)

14. Build the weighted label stack W~:
        - beta_t = alpha_t / ||ell_t||^2
        - wtilde_t = sqrt(beta_t) * ell_t
        - Stack wtilde_t as rows of W~.

15. Compute the rank-1 approximation of W~:
        - Run SVD on W~.
        - Extract leading singular value sigma1.
        - Extract leading right singular vector v.
        - Report the rank-1 energy ratio.

16. Check the rank-one projection identity on the learned score:
        - Compute Delta_v(r) = v^T Kc v / ||Kc||_F.
        - Compute J1(r) = sigma1^2 * Delta_v(r).
        - Print J(r), Delta_v(r), J1(r), sigma1^2 * Delta_v(r), and their
          numerical difference.

17. Run threshold certificate scans:
        - For each retained target, scan all sorted threshold splits.
        - Compute rho^2 for the centered label direction.
        - Report the best split and best rho^2 per target.

18. Scan the rank-1 direction:
        - Scan threshold splits for v.
        - Report the v-threshold best split and rho^2.

19. Compare threshold families:
        - Compute exact threshold objective Jthr(j) for every split.
        - Find the best threshold for full J.
        - Compare it with the threshold selected by v.
        - Report absolute and relative gap.

20. Compute the floored epsilon-gap bound:
        - Build rank-1 approximation W1 = sigma1 * u v^T.
        - Compute residual E = W~ - W1.
        - Use the L1-based upper-bound expression.
        - Print the threshold comparison under a floored objective.

21. Optionally run full-matrix threshold sanity checks:
        - Build explicit two-level threshold score vectors.
        - Construct centered kernels for selected threshold splits.
        - Compare direct Delta_v values against predicted rho^2 values.

22. Optionally run extra monotone-score identity checks:
        - Perturb the learned score with small noise.
        - Sort the perturbed scores to keep a monotone score vector.
        - Rebuild Kc.
        - Print Delta_v and sigma1^2 * Delta_v for each check.

23. Optionally run noise-bridge checks:
        - Add Gaussian noise to scores multiple times.
        - Build centered kernels for noisy scores.
        - Estimate the Monte Carlo average kernel Kbar.
        - Compare nHSIC deviations against the 4/gamma bound.

24. Print DONE.

Input CSV Requirements
----------------------
The input CSV must contain:

    - the ordering score column specified by --order_col
    - at least one usable target column from --targets
    - long_stay_defined if long_stay is used and validity should be based on it

Typical input columns:

    hadm_id, learned_score, mortality, mortality_30d, long_stay,
    long_stay_defined, icu_transfer

Missing target columns are skipped. Targets are also skipped if they are empty or
single-class among valid rows. At least one usable target must remain.

Target Validity Handling
------------------------
For long_stay:

    valid = long_stay_defined == 1

when long_stay_defined is present.

For all other targets:

    valid = target is not missing

Rows must be valid for every retained target. The script uses this common row set
for all checks.

Score Ordering and Scaling
--------------------------
Rows are sorted by the chosen scalar score column:

    order_col = learned_score by default

If --flip_order is supplied, the score is negated before sorting.

If score_scale=minmax, the sorted scores are linearly mapped into [-B, B]:

    z = (r - min(r)) / (max(r) - min(r))
    r_scaled = (2z - 1) * B

If all scores are identical, the scaled score vector is set to zero.

Supported Kernels
-----------------
The script supports three radial kernels on pairwise score distances d:

RBF kernel:

    k(d) = exp(-d^2 / (2 sigma^2))

Laplacian kernel:

    k(d) = exp(-d / sigma)

Cauchy kernel:

    k(d) = 1 / (1 + (d / sigma)^2)

If --sigma auto is used, sigma is chosen as the median absolute pairwise score
difference over random score pairs.

nHSIC Objective
---------------
For a centered kernel Kc and centered label vector ell, per-task nHSIC is:

    nHSIC_t(r) = ell_t^T Kc ell_t / (||Kc||_F * ||ell_t||^2)

The weighted multi-target objective is:

    J(r) = sum_t alpha_t nHSIC_t(r)

The convention is to return 0 when ||Kc|| is zero or the centered label vector has
near-zero norm.

Threshold Scan
--------------
For sorted rows, each threshold split j partitions rows into:

    L_j = {0, ..., j}
    R_j = {j+1, ..., n-1}

The centered threshold step vector is:

    g_j = H 1_{R_j}

For a centered vector w, the script computes:

    rho_j = (w^T g_j) / (||w|| ||g_j||)

and scans rho_j^2 over all valid threshold splits.

For binary threshold scores and radial kernels, the per-task threshold nHSIC
equals rho_j^2 in the unfloored case.

Rank-1 Label Stack
------------------
The script builds the weighted label stack W~ using:

    beta_t = alpha_t / ||ell_t||^2
    wtilde_t = sqrt(beta_t) * ell_t

Then it computes the SVD:

    W~ = U S V^T

The leading right singular vector v defines the rank-1 direction used for the
rank-1 threshold certificate.

Rank-One Projection Identity Check
----------------------------------
The script checks the identity:

    J1(r) = sigma1^2 * Delta_v(r)

where:

    Delta_v(r) = v^T Kc v / ||Kc||_F

and sigma1 is the leading singular value of W~.

The output prints J(r), Delta_v(r), J1(r), sigma1^2 * Delta_v(r), and their
absolute difference.

Floored Epsilon-Gap Bound
-------------------------
The script computes an upper bound of the form:

    |J_eps0(r) - J1_eps0(r)|
        <= (kappa_abs_max / eps0)
           * sum_t (||w_t||_1 + ||w1_t||_1) ||e_t||_1

where:

    W1 = sigma1 * u v^T
    E = W~ - W1
    e_t is row t of E

For the supported kernels, kappa_abs_max is treated as 1.

The script also checks a threshold comparison using the floored threshold
objective.

Optional Noise Bridge Check
---------------------------
When --noise_checks > 0 and --noise_sigma > 0, the script performs a Monte Carlo
noise bridge check:

    r_noisy = r + eta,  eta ~ Normal(0, noise_sigma^2)

It estimates:

    Kbar = E[Kc(r + eta)]

using Monte Carlo samples, then checks whether each draw satisfies:

    ||Kc - Kbar|| <= gamma / 2

where:

    gamma = ||Kbar||_F

For each draw, it prints the observed nHSIC deviation and the bound:

    (4 / gamma) * ||Kc - Kbar||

Key Configuration Defaults
--------------------------
  order_col=learned_score
  n_max=5000
  seed=12345
  kernel=rbf
  sigma=auto
  score_scale=minmax
  B=1.0
  eps0=1e-6
  threshold_checks=0
  identity_checks=2
  require_all_valid=True
  noise_checks=0
  noise_sigma=0.0
  alphas=all ones

Outputs
-------
The script prints:

    - retained sample size and number of usable targets
    - score ordering/scaling configuration
    - kernel type and sigma
    - retained target names and weights
    - rank-1 SVD summary for W~
    - learned-score per-target nHSIC
    - weighted J(r)
    - rank-one projection identity check
    - per-target threshold rho^2 certificates
    - rank-1 direction threshold certificate
    - full-J threshold versus v-threshold comparison
    - floored epsilon-gap bound
    - optional full-matrix threshold checks
    - optional random monotone identity checks
    - optional noise bridge results

This script does not write output files by default.

Dependencies
------------
    pip install numpy pandas

Running
-------
Run example:

    python3 validate_rank1_nhsic_theory_mimiciii.py \
      --csv "/path/to/DiagnosticCSV/CORE_seed1001_test_scores_and_outcomes.csv" \
      --targets mortality mortality_30d long_stay icu_transfer \
      --order_col learned_score \
      --n_max 11909 --seed 12345 \
      --kernel rbf --sigma auto \
      --eps0 1e-6 \
      --threshold_checks 5 \
      --identity_checks 3
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
    Returns (valid_mask, y_int01, err).
    For long_stay, uses long_stay_defined if present.
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
    For centered w, compute rho_j^2 for each split j.

    rho_j = (w^T g_j) / (||w|| ||g_j||)

    Since w is centered, w^T g_j equals the sum of w over Rj.

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
# Rank-1 stack W~
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
# Floored gap bound
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
    ap.add_argument("--threshold_checks", type=int, default=0, help="build full threshold Kc for selected splits")
    ap.add_argument("--identity_checks", type=int, default=2, help="number of random monotone score vectors for identity checks")
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

    if args.threshold_checks and args.threshold_checks > 0:
        print("\nFull-matrix sanity checks for selected threshold splits:")

        rng = np.random.default_rng(args.seed)
        js = set([jv, jJ])

        while len(js) < min(args.threshold_checks, n - 1):
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

    if args.identity_checks and args.identity_checks > 0:
        print("\nExtra rank-one identity checks on random monotone score vectors:")

        rng = np.random.default_rng(args.seed + 999)
        base = np.asarray(r, dtype=np.float64)

        for k in range(args.identity_checks):
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
