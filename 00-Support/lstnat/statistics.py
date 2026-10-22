"""Pairwise filtering, rank correlations, and curve-based knee detection."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import spearmanr, rankdata, t as student_t


def pair_filter(frame, lst, nat, multiplier=2):
    """Determine BOTH IQR intervals on the same complete-pair sample."""
    values = frame[[lst, nat]].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(values).all(axis=1)
    keep = complete.copy()
    for column in [lst, nat]:
        series = values.loc[complete, column]
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        width = q3 - q1
        # A zero IQR is a zero-width interval, not a reason to skip filtering.
        keep &= values[column].between(q1 - multiplier * width, q3 + multiplier * width)
    return frame.loc[keep].copy()


def correlations(frame, lst, nat, controls=()):
    cols = [lst, nat, *controls]
    clean = frame.loc[np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1)]
    n = len(clean)
    empty = {"n": n, "rho": np.nan, "p": np.nan, "partial_rho": np.nan,
             "partial_p": np.nan, "control_rank": 0}
    if n < 3 or clean[lst].nunique() < 2 or clean[nat].nunique() < 2:
        return empty
    rho, p = spearmanr(clean[lst], clean[nat])
    result = {**empty, "rho": float(rho), "p": float(p)}
    if not controls:
        return result
    ranked = np.column_stack([rankdata(clean[c], method="average") for c in cols])
    design = np.column_stack([np.ones(n), ranked[:, 2:]])
    rank = np.linalg.matrix_rank(design)
    residuals = ranked[:, :2] - design @ np.linalg.lstsq(design, ranked[:, :2], rcond=None)[0]
    if np.min(residuals.std(axis=0)) < 1e-12:
        return result
    partial = float(np.clip(np.corrcoef(residuals.T)[0, 1], -1, 1))
    df = n - rank - 1
    partial_p = (2 * student_t.sf(abs(partial) * np.sqrt(df / max(1 - partial**2, 1e-15)), df)
                 if df > 0 else np.nan)
    return {**result, "partial_rho": partial, "partial_p": float(partial_p),
            "control_rank": int(rank - 1)}


def balanced_correlations(frame, lst, nat, controls=(), iterations=1000, seed=42):
    """Smallest zone is the benchmark; larger zones are bootstrapped to its size."""
    cols = [lst, nat, *controls]
    frame = frame.loc[np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1)]
    groups = {zone: group for zone, group in frame.groupby("UrG")}
    if not groups:
        return []
    n = min(map(len, groups.values()))
    rng = np.random.default_rng(seed)
    rows = []
    for zone, group in groups.items():
        row = {"zone": zone, "available_n": len(group), "matched_n": n,
               **correlations(group, lst, nat, controls)}
        samples = []
        if len(group) > n:
            for _ in range(iterations):
                sample = group.iloc[rng.integers(0, len(group), n)]
                stat = correlations(sample, lst, nat, controls)
                samples.append([stat["rho"], stat["partial_rho"]])
        for i, key in enumerate(["rho", "partial_rho"]):
            values = np.asarray(samples)[:, i] if samples else np.array([])
            values = values[np.isfinite(values)]
            row[f"balanced_{key}"] = float(np.median(values)) if len(values) else row[key]
            row[f"{key}_ci_low"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{key}_ci_high"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
        # P values above refer to the observed full zone, not the bootstrap median.
        row["bootstrap_iterations"] = iterations if len(group) > n else 0
        rows.append(row)
    return rows


def knee(scales, values, window=7, order=2):
    """Largest interior local maximum of (normalized response - diagonal).

    Implements the normalization/deviation rule in manuscript Section 2.3.2.
    No ranked manual alternatives, shifts, monotonic regression, or fallback knee.
    """
    x, y = np.asarray(scales, float), np.asarray(values, float)
    if len(x) != len(y) or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Knee detection requires a complete finite response curve")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Scales must increase strictly")
    if window % 2 != 1 or window > len(y) or not 0 <= order < window:
        raise ValueError("Invalid Savitzky-Golay settings")
    smooth = savgol_filter(y, window, order, mode="interp")
    result = {"scale": None, "value": None, "raw_value": None,
              "residual_increase_pct": None, "reason": "no interior positive maximum"}
    if np.ptp(smooth) < 1e-12:
        result["reason"] = "constant response"
        return smooth, result
    distance = (smooth - smooth.min()) / np.ptp(smooth) - (x - x.min()) / np.ptp(x)
    peaks, _ = find_peaks(distance)
    peaks = peaks[distance[peaks] > 1e-12]
    if not len(peaks):
        return smooth, result
    idx = int(peaks[np.argmax(distance[peaks])])
    span = float(smooth.max() - smooth.min())
    result.update(scale=int(x[idx]), value=float(smooth[idx]), raw_value=float(y[idx]),
                  residual_increase_pct=float((smooth.max() - smooth[idx]) / span * 100),
                  reason="largest local maximum")
    return smooth, result
