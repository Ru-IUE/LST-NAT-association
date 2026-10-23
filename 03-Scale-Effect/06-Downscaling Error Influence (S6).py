"""Downscaling-error sensitivity of the global bivariate Lee's L scale curves.

This historical filename is retained for compatibility. It is a LEGACY
TABLE DIAGNOSTIC, not Supplement S5. Use
``06-Downscaling Error Influence (S5).py --config <config.json>`` for S5's
pixelwise Gaussian-noise experiment. Legacy mode requires --legacy-table.

For each time-of-day case and each relative LST perturbation
(0%, 5%, 10%), the script perturbs the scale-window LST values in the legacy
multiscale table, rebuilds the adaptive Gaussian KNN weights (K = 8), and
recomputes global bivariate Lee's L at every spatial scale.  It then extracts
an elbow from the smoothed curve and draws the 2 x 2 combined figure.

The legacy diagnostic uses UNIFORM noise on table values, not pixelwise Gaussian noise.
All three error levels use the same Savitzky-Golay settings:
window length = 7, polynomial order = 2.  The same settings are used for the
elbow search and plotting. Curves are smoothed independently without shape anchoring.

Inputs
------
``<scale-dir>`` must contain the four legacy multiscale GeoPackages::

    Day Terra Multiscale.gpkg
    Day Aqua Multiscale.gpkg
    Night Terra Multiscale.gpkg
    Night Aqua Multiscale.gpkg

Each table must contain the NAT column (TEM10/TEM13/TEM22/TEM01) and the
window-mean LST columns L10, L30, ..., L1210.
This table-based diagnostic does not reconstruct physical sampling supports;
its input scale labels have unverified ground-distance meaning.

Outputs
-------
By default, results are written to ``./RES`` next to this script:

    lee_curves.csv
    elbow_summary.csv
    Plot/Combined_Sensitivity.svg
    Plot/Combined_Sensitivity.pdf
    Plot/Combined_Sensitivity.png

Usage
-----
    python "06-Downscaling Error Influence (S6).py" --legacy-table
    python "06-Downscaling Error Influence (S6).py" --legacy-table --scale-dir "D:/data/RES"
    python "06-Downscaling Error Influence (S6).py" --legacy-table --plot-only
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.lee import Spatial_Pearson
from libpysal.weights import W
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*was renamed to.*ensure_all_finite.*",
    category=FutureWarning,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCALE_RES = SCRIPT_DIR / "RES"
OUT_DIR = SCRIPT_DIR / "RES"
PLOT_DIR = OUT_DIR / "Plot"


def discover_scale_dir() -> Path:
    """Return the first candidate folder that contains the legacy tables."""
    candidates = [
        SCRIPT_DIR / "RES",
        SCRIPT_DIR.parent / "Scale Effect" / "RES",
        SCRIPT_DIR.parents[1] / "Scale Effect" / "RES",
        Path.cwd() / "Scale Effect" / "RES",
    ]
    for candidate in candidates:
        if (candidate / "Day Terra Multiscale.gpkg").is_file():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find 'Day Terra Multiscale.gpkg'. Searched:\n" + searched
    )


def configure_paths(scale_dir: Path | None, out_dir: Path | None) -> None:
    global SCALE_RES, OUT_DIR, PLOT_DIR
    SCALE_RES = (
        Path(scale_dir).expanduser().resolve()
        if scale_dir is not None
        else discover_scale_dir()
    )
    OUT_DIR = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else SCRIPT_DIR / "RES"
    )
    PLOT_DIR = OUT_DIR / "Plot"


# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
NEIGHBOR_SIZES = list(range(10, 1211, 20))  # 61 labels, 10, 30, ..., 1210
SCALE_COLS = [f"L{size}" for size in NEIGHBOR_SIZES]
PLOT_MAX_SCALE = 1210
REL_ERRORS = (0.0, 0.05, 0.10)
RNG_SEED = 108

BASE_K = 8
IQR_MULTIPLIER = 2
DECAY_KERNEL = "gaussian"
DISTANCE_UNIT = "km"
SYMMETRIZE_WEIGHTS = True
SYMMETRY_METHOD = "mean"

# Savitzky-Golay settings used everywhere in this script.
SG_WINDOW = 7
SG_ORDER = 2

# Every scenario uses the same smoothing and the highest-deviation knee.
SMOOTH_PARAMS = {key: {sigma: (SG_WINDOW, SG_ORDER, 1.0) for sigma in REL_ERRORS}
                 for key in ("T1", "T2", "T3", "T4")}
SHAPE_ANCHOR_TO_ORIGINAL = False
ELBOW_SMOOTH_WINDOW = SG_WINDOW
ELBOW_SMOOTH_POLYORDER = SG_ORDER
ELBOW_RANK = {key: {sigma: 1 for sigma in REL_ERRORS} for key in SMOOTH_PARAMS}

COLOR_LEE = "#9673A6"
COLOR_ERR05 = "#5B92C8"
COLOR_ERR10 = "#C47A3A"
INK = "#222222"
SCATTER_ALPHA = {0.0: 0.45, 0.05: 0.22, 0.10: 0.16}

CASES = (
    {
        "key": "T1",
        "title": "10:00–11:00",
        "file": "Day Terra Multiscale.gpkg",
        "tem": "TEM10",
        "ylim_pad_top": 0.007,
    },
    {
        "key": "T2",
        "title": "13:00–14:00",
        "file": "Day Aqua Multiscale.gpkg",
        "tem": "TEM13",
        "ylim_pad_top": 0.01,
    },
    {
        "key": "T3",
        "title": "22:00–23:00",
        "file": "Night Terra Multiscale.gpkg",
        "tem": "TEM22",
        "ylim_pad_top": 0.01,
    },
    {
        "key": "T4",
        "title": "01:00–02:00",
        "file": "Night Aqua Multiscale.gpkg",
        "tem": "TEM01",
        "ylim_pad_top": 0.0115,
    },
)


# ---------------------------------------------------------------------------
# IQR filtering and spatial weights
# ---------------------------------------------------------------------------
def identify_iqr_outliers(
    frame: gpd.GeoDataFrame,
    columns: list[str],
    iqr_multiplier: float = 2,
    geometry_column: str = "geometry",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    complete_mask = frame[geometry_column].notna()
    for column in columns:
        complete_mask &= frame[column].notna()

    complete_cases = frame.loc[complete_mask, columns].astype("float64")
    outlier_mask = pd.Series(False, index=frame.index)
    if complete_cases.empty:
        analysis_mask = complete_mask & ~outlier_mask
        return complete_mask, outlier_mask, analysis_mask

    keep_mask = pd.Series(True, index=complete_cases.index)
    for column in columns:
        series = complete_cases[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        keep_mask &= series.between(lower, upper, inclusive="both")

    outlier_mask.loc[complete_cases.index] = ~keep_mask
    analysis_mask = complete_mask & ~outlier_mask
    return complete_mask, outlier_mask, analysis_mask


def choose_k(n: int, target_k: int = 8) -> int | None:
    if n < 2:
        return None
    return min(target_k, n - 1)


def adaptive_kernel(
    distances: np.ndarray,
    bandwidth: float,
    kernel: str = "gaussian",
    eps: float = 1e-9,
) -> np.ndarray:
    bandwidth = max(float(bandwidth), eps)
    distances = np.asarray(distances, dtype="float64")
    scaled = distances / bandwidth
    if kernel == "bisquare":
        weights = np.square(np.clip(1.0 - np.square(scaled), 0.0, None))
    elif kernel == "gaussian":
        weights = np.exp(-0.5 * np.square(scaled))
    elif kernel == "inverse_distance":
        weights = 1.0 / np.power(np.maximum(distances, eps), 1.0)
    else:
        raise ValueError(f"Unsupported decay kernel: {kernel}")
    if np.allclose(weights.sum(), 0.0):
        weights = np.ones_like(distances, dtype="float64")
    return weights


def symmetrize_neighbor_weights(
    neighbor_ids: dict,
    neighbor_weights: dict,
    id_order: list,
    method: str = "mean",
) -> tuple[dict, dict]:
    edge_weights = {}
    for focal_id in id_order:
        for neighbor_id, weight in zip(
            neighbor_ids[focal_id], neighbor_weights[focal_id]
        ):
            edge_weights[(focal_id, neighbor_id)] = float(weight)

    symmetric_neighbor_ids = {row_id: [] for row_id in id_order}
    symmetric_neighbor_weights = {row_id: [] for row_id in id_order}
    undirected_pairs = {
        tuple(sorted((src_id, dst_id)))
        for src_id, dst_id in edge_weights
        if src_id != dst_id
    }

    for left_id, right_id in undirected_pairs:
        left_weight = edge_weights.get((left_id, right_id))
        right_weight = edge_weights.get((right_id, left_id))
        if left_weight is not None and right_weight is not None:
            if method == "mean":
                shared_weight = (left_weight + right_weight) / 2.0
            elif method == "max":
                shared_weight = max(left_weight, right_weight)
            elif method == "min":
                shared_weight = min(left_weight, right_weight)
            else:
                raise ValueError(f"Unsupported symmetry_method: {method}")
        else:
            shared_weight = (
                left_weight if left_weight is not None else right_weight
            )
        symmetric_neighbor_ids[left_id].append(right_id)
        symmetric_neighbor_weights[left_id].append(float(shared_weight))
        symmetric_neighbor_ids[right_id].append(left_id)
        symmetric_neighbor_weights[right_id].append(float(shared_weight))
    return symmetric_neighbor_ids, symmetric_neighbor_weights


def build_adaptive_knn_weights(
    frame: gpd.GeoDataFrame,
    base_k: int = 8,
    kernel: str = "gaussian",
    distance_unit: str = "km",
    symmetrize: bool = True,
    symmetry_method: str = "mean",
) -> W | None:
    k = choose_k(len(frame), base_k)
    if k is None:
        return None

    coords_local = np.column_stack(
        [frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()]
    ).astype("float64")
    if distance_unit == "km":
        coords_local = coords_local / 1000.0
    elif distance_unit != "m":
        raise ValueError(f"Unsupported distance_unit: {distance_unit}")

    tree = cKDTree(coords_local)
    distances, indices = tree.query(coords_local, k=k + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    frame_ids = frame.index.to_list()
    neighbor_ids = {row_id: [] for row_id in frame_ids}
    neighbor_weights = {row_id: [] for row_id in frame_ids}

    for row_pos, row_id in enumerate(frame_ids):
        row_distances = np.asarray(distances[row_pos]).ravel()
        row_indices = np.asarray(indices[row_pos]).ravel()
        valid = np.isfinite(row_distances) & (row_indices != row_pos)
        neighbor_pos = row_indices[valid][:k].astype(int)
        neighbor_dist = row_distances[valid][:k].astype("float64")
        if len(neighbor_pos) == 0:
            continue
        bandwidth = float(np.nanmax(neighbor_dist))
        if bandwidth <= 0:
            kernel_weights = np.ones(len(neighbor_pos), dtype="float64")
        else:
            kernel_weights = adaptive_kernel(
                neighbor_dist, bandwidth=bandwidth, kernel=kernel
            )
        positive = kernel_weights > 0
        if not np.any(positive):
            kernel_weights = np.ones(len(neighbor_pos), dtype="float64")
            positive = kernel_weights > 0
        neighbor_pos = neighbor_pos[positive]
        kernel_weights = kernel_weights[positive]
        neighbor_ids[row_id] = [frame_ids[pos] for pos in neighbor_pos]
        neighbor_weights[row_id] = kernel_weights.tolist()

    if symmetrize:
        neighbor_ids, neighbor_weights = symmetrize_neighbor_weights(
            neighbor_ids=neighbor_ids,
            neighbor_weights=neighbor_weights,
            id_order=frame_ids,
            method=symmetry_method,
        )

    w = W(neighbor_ids, neighbor_weights, id_order=frame_ids)
    w.transform = "R"
    return w


def compute_global_bivariate_lee(
    frame: gpd.GeoDataFrame,
    temp_col: str,
    lst_col: str,
    w: W,
) -> float:
    model = Spatial_Pearson(connectivity=w.sparse, permutations=0).fit(
        frame[[temp_col]].to_numpy(dtype="float64"),
        frame[[lst_col]].to_numpy(dtype="float64"),
    )
    return float(model.association_[0, 1])


def global_lee_at_scale(
    frame: gpd.GeoDataFrame, temp_col: str, lst_col: str
) -> tuple[float, int]:
    working = frame[[temp_col, lst_col, "geometry"]].copy()
    _, _, analysis_mask = identify_iqr_outliers(
        working, columns=[temp_col, lst_col], iqr_multiplier=IQR_MULTIPLIER
    )
    analysis_gdf = working.loc[analysis_mask].copy()
    n = len(analysis_gdf)
    if n < 2:
        return np.nan, n
    if (
        analysis_gdf[temp_col].nunique(dropna=True) < 2
        or analysis_gdf[lst_col].nunique(dropna=True) < 2
    ):
        return np.nan, n
    w = build_adaptive_knn_weights(
        analysis_gdf,
        base_k=BASE_K,
        kernel=DECAY_KERNEL,
        distance_unit=DISTANCE_UNIT,
        symmetrize=SYMMETRIZE_WEIGHTS,
        symmetry_method=SYMMETRY_METHOD,
    )
    if w is None:
        return np.nan, n
    return compute_global_bivariate_lee(analysis_gdf, temp_col, lst_col, w), n


# ---------------------------------------------------------------------------
# Perturbation and smoothing
# ---------------------------------------------------------------------------
def perturb_scale_lst(
    gdf: gpd.GeoDataFrame,
    scale_cols: list[str],
    rel_error: float,
    rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """Apply a relative uniform error to every scale-window LST value."""
    out = gdf.copy()
    if rel_error <= 0:
        return out
    for col in scale_cols:
        values = out[col].to_numpy(dtype="float64")
        finite = np.isfinite(values)
        noise = np.zeros(values.shape, dtype="float64")
        n_finite = int(finite.sum())
        if n_finite:
            noise[finite] = rng.uniform(-rel_error, rel_error, size=n_finite)
        out[col] = values * (1.0 + noise)
    return out


def find_elbow_point(
    x_values: np.ndarray,
    y_values: np.ndarray,
    rank: int = 1,
    min_separation: int = 2,
) -> int:
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    finite_idx = np.flatnonzero(valid_mask)
    if len(finite_idx) < 3:
        return int(finite_idx[0]) if len(finite_idx) else 0

    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]
    x_span = x_valid.max() - x_valid.min()
    y_span = y_valid.max() - y_valid.min()
    if x_span == 0 or y_span == 0:
        fallback_rank = min(max(rank, 1), len(finite_idx)) - 1
        return int(finite_idx[fallback_rank])

    x_norm = (x_valid - x_valid.min()) / x_span
    y_norm = (y_valid - y_valid.min()) / y_span
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    a_value = y2 - y1
    b_value = x1 - x2
    c_value = x2 * y1 - x1 * y2
    denominator = np.sqrt(a_value ** 2 + b_value ** 2)
    if denominator == 0:
        fallback_rank = min(max(rank, 1), len(finite_idx)) - 1
        return int(finite_idx[fallback_rank])

    distances = (
        np.abs(a_value * x_norm + b_value * y_norm + c_value) / denominator
    )
    ranked_local_indices = np.argsort(distances)[::-1]
    selected_indices: list[int] = []
    target_rank = max(rank, 1)
    for candidate_idx in ranked_local_indices:
        candidate_idx = int(candidate_idx)
        if all(
            abs(candidate_idx - chosen_idx) >= min_separation
            for chosen_idx in selected_indices
        ):
            selected_indices.append(candidate_idx)
        if len(selected_indices) >= target_rank:
            break
    if not selected_indices:
        return int(finite_idx[0])
    chosen_local_idx = selected_indices[
        min(target_rank, len(selected_indices)) - 1
    ]
    return int(finite_idx[chosen_local_idx])


def sg_smooth(
    y_values: np.ndarray,
    window_length: int = SG_WINDOW,
    polyorder: int = SG_ORDER,
) -> np.ndarray:
    y_values = np.asarray(y_values, dtype="float64")
    finite_mask = np.isfinite(y_values)
    if finite_mask.sum() < 3:
        return y_values.copy()
    valid_values = y_values[finite_mask]
    n = len(valid_values)
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < 3:
        return y_values.copy()
    if window_length % 2 == 0:
        window_length -= 1
    if polyorder >= window_length:
        polyorder = window_length - 1
    smoothed = y_values.copy()
    smoothed[finite_mask] = savgol_filter(
        valid_values,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp",
    )
    return smoothed


def lookup_smooth_params(panel_key: str, rel_error: float) -> tuple[int, int, float]:
    panel = SMOOTH_PARAMS[panel_key]
    for key, params in panel.items():
        if np.isclose(rel_error, key):
            if len(params) == 2:
                window, polyorder = params
                residual_mix = 1.0
            else:
                window, polyorder, residual_mix = params
            return int(window), int(polyorder), float(residual_mix)
    return SG_WINDOW, SG_ORDER, 1.0


def lookup_elbow_rank(panel_key: str, rel_error: float) -> int:
    panel = ELBOW_RANK[panel_key]
    for key, rank in panel.items():
        if np.isclose(rel_error, key):
            return max(int(rank), 1)
    return 1


def scatter_alpha_for(rel_error: float) -> float:
    for key, alpha in SCATTER_ALPHA.items():
        if np.isclose(rel_error, key):
            return float(alpha)
    return 0.35


def smooth_with_shape_anchor(
    lee: np.ndarray,
    window: int,
    polyorder: int,
    lee_ref: np.ndarray | None = None,
    residual_mix: float = 1.0,
) -> np.ndarray:
    """SG-smooth ``lee``.  If ``lee_ref`` is given, keep its rise-plateau shape."""
    if lee_ref is None or not SHAPE_ANCHOR_TO_ORIGINAL:
        return sg_smooth(lee, window_length=window, polyorder=polyorder)

    ref_sg = sg_smooth(
        lee_ref, window_length=SG_WINDOW, polyorder=SG_ORDER
    )
    mask = np.isfinite(lee) & np.isfinite(ref_sg)
    if int(mask.sum()) < 3:
        return sg_smooth(lee, window_length=window, polyorder=polyorder)

    design = np.column_stack(
        [ref_sg[mask], np.ones(int(mask.sum()), dtype="float64")]
    )
    coef, *_ = np.linalg.lstsq(design, lee[mask], rcond=None)
    fitted = coef[0] * ref_sg + coef[1]
    residual = lee - fitted
    residual_s = sg_smooth(residual, window_length=window, polyorder=polyorder)
    return fitted + float(residual_mix) * residual_s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.visible": True,
            "ytick.minor.visible": False,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), dpi=1200, bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=1200, bbox_inches="tight", transparent=True)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )


def style_axes_like_02(
    ax: plt.Axes,
    *,
    x_direction: str = "in",
    y_direction: str = "out",
    top: bool = True,
    right: bool = True,
) -> None:
    ax.minorticks_off()
    ax.tick_params(
        axis="x",
        which="major",
        direction=x_direction,
        top=top,
        bottom=True,
        labelsize=9,
        pad=3.5,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction=y_direction,
        left=True,
        right=right,
        labelcolor="black",
        labelsize=9,
        pad=2,
    )
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_major_locator(MultipleLocator(0.03))


def scatter_elbow_marker(ax: plt.Axes, scale: float, lee: float, color: str) -> None:
    ax.scatter(
        scale,
        lee,
        color=color,
        s=36,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=5,
    )


def plot_combined(records: pd.DataFrame) -> None:
    mm = 1 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(170 * mm, 118 * mm), dpi=300)
    letters = "abcd"
    series = (
        (0.10, COLOR_ERR10, "10% Perturbation"),
        (0.05, COLOR_ERR05, "5% Perturbation"),
        (0.0, COLOR_LEE, "Unperturbed"),
    )

    for ax, case, letter in zip(axes.ravel(), CASES, letters):
        panel_rows = records[records["time"] == case["key"]]
        ref_sub = panel_rows[
            np.isclose(panel_rows["rel_error"], 0.0)
        ].sort_values("scale_m")
        lee_ref = ref_sub["lee_l"].to_numpy(dtype="float64")
        panel_lee: list[np.ndarray] = []
        ref_elbow_scale: float | None = None

        for rel_error, color, label in series:
            sub = panel_rows[
                np.isclose(panel_rows["rel_error"], rel_error)
            ].sort_values("scale_m")
            scales = sub["scale_m"].to_numpy(dtype="float64")
            lee = sub["lee_l"].to_numpy(dtype="float64")
            plot_mask = scales <= PLOT_MAX_SCALE
            plot_scales = scales[plot_mask]
            plot_lee = lee[plot_mask]
            panel_lee.append(plot_lee)

            window, polyorder, residual_mix = lookup_smooth_params(
                case["key"], rel_error
            )
            lee_sg = smooth_with_shape_anchor(
                lee,
                window=window,
                polyorder=polyorder,
                lee_ref=None if np.isclose(rel_error, 0.0) else lee_ref,
                residual_mix=residual_mix,
            )
            ax.scatter(
                plot_scales,
                plot_lee,
                color=color,
                s=6.5,
                alpha=scatter_alpha_for(rel_error),
                zorder=2,
                edgecolors="none",
            )
            ax.plot(
                plot_scales,
                lee_sg[plot_mask],
                color=color,
                linewidth=1.2,
                label=label,
                zorder=3,
            )

            elbow_sg = sg_smooth(
                lee,
                window_length=ELBOW_SMOOTH_WINDOW,
                polyorder=ELBOW_SMOOTH_POLYORDER,
            )
            elbow_idx = find_elbow_point(
                scales,
                elbow_sg,
                rank=lookup_elbow_rank(case["key"], rel_error),
            )
            elbow_scale = float(scales[elbow_idx])
            elbow_lee = float(lee_sg[elbow_idx])
            scatter_elbow_marker(ax, elbow_scale, elbow_lee, color)
            if np.isclose(rel_error, 0.0):
                ref_elbow_scale = elbow_scale

        if ref_elbow_scale is not None:
            ax.axvline(
                x=ref_elbow_scale,
                color="gray",
                linestyle="--",
                alpha=0.7,
                linewidth=0.8,
                zorder=1,
            )

        stacked = np.concatenate(panel_lee)
        ax.set_xlim(0, PLOT_MAX_SCALE)
        ax.set_ylim(
            float(np.nanmin(stacked)) - 0.01,
            float(np.nanmax(stacked)) + float(case["ylim_pad_top"]),
        )
        style_axes_like_02(
            ax, x_direction="in", y_direction="out", top=False, right=False
        )
        ax.set_ylabel("Global Bivariate Lee's L", fontsize=9, labelpad=6)
        if case["key"] in {"T3", "T4"}:
            ax.set_xlabel("Input scale label (ground distance unverified)", fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        order = [
            labels.index(name)
            for name in ("Unperturbed", "5% Perturbation", "10% Perturbation")
        ]
        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc="lower right",
            ncol=1,
            frameon=True,
            fancybox=False,
            edgecolor="none",
            facecolor="white",
            framealpha=0.92,
            fontsize=9,
            handlelength=1.5,
            handletextpad=0.45,
            labelspacing=0.3,
            borderpad=0.35,
            borderaxespad=0.5,
        )

    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.99), h_pad=1.6, w_pad=0.6)
    fig.subplots_adjust(wspace=0.28)
    for ax in axes.ravel():
        ax.yaxis.set_label_coords(-0.16, 0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in zip(axes.ravel(), letters):
        bbox = ax.yaxis.label.get_window_extent(renderer)
        x_axes, _ = ax.transAxes.inverted().transform(
            ((bbox.x0 + bbox.x1) / 2.0, bbox.y1)
        )
        ax.text(
            x_axes,
            0.96,
            letter,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="black",
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=7,
        )
    save_figure(fig, PLOT_DIR / "Combined_Sensitivity")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading and experiment
# ---------------------------------------------------------------------------
def load_case_gdf(case: dict) -> gpd.GeoDataFrame:
    path = SCALE_RES / case["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path.name} has no CRS.")
    missing = [
        col for col in SCALE_COLS + [case["tem"]] if col not in gdf.columns
    ]
    if missing:
        raise KeyError(f"{path.name} missing columns: {missing}")
    return gdf.to_crs(epsg=3857).copy()


def run_case(case: dict, rng: np.random.Generator) -> pd.DataFrame:
    gdf = load_case_gdf(case)
    rows: list[dict] = []
    for rel_error in REL_ERRORS:
        perturbed = perturb_scale_lst(gdf, SCALE_COLS, rel_error, rng)
        print(f"  {case['key']}  rel_error={error_label(rel_error)}")
        for col in SCALE_COLS:
            lee_l, n = global_lee_at_scale(perturbed, case["tem"], col)
            scale_m = int(col[1:])
            rows.append(
                {
                    "time": case["key"],
                    "title": case["title"],
                    "rel_error": rel_error,
                    "scale_col": col,
                    "scale_m": scale_m,
                    "scale_support": "unverified_input_labels",
                    "lee_l": lee_l,
                    "n": n,
                }
            )
            if scale_m in {10, 210, 230, 1210}:
                print(f"    {col}: L={lee_l:.4f}, N={n}")
    return pd.DataFrame(rows)


def error_label(rel_error: float) -> str:
    return f"{int(round(rel_error * 100))}%"


def write_elbow_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for case in CASES:
        for rel_error in REL_ERRORS:
            sub = records[
                (records["time"] == case["key"])
                & np.isclose(records["rel_error"], rel_error)
            ].sort_values("scale_m")
            scales = sub["scale_m"].to_numpy(dtype="int64")
            lee = sub["lee_l"].to_numpy(dtype="float64")
            lee_sg = sg_smooth(
                lee,
                window_length=ELBOW_SMOOTH_WINDOW,
                polyorder=ELBOW_SMOOTH_POLYORDER,
            )
            elbow_rank = lookup_elbow_rank(case["key"], rel_error)
            elbow_idx = find_elbow_point(
                scales.astype("float64"), lee_sg, rank=elbow_rank
            )
            plot_mask = scales <= PLOT_MAX_SCALE
            idx_1210 = np.flatnonzero(scales == 1210)
            lee_min = float(np.nanmin(lee[plot_mask]))
            lee_opt = float(lee[elbow_idx])
            lee_1210 = float(lee[idx_1210[0]]) if len(idx_1210) else np.nan
            increment = (
                (lee_opt - lee_min) / (lee_1210 - lee_min)
                if np.isfinite(lee_1210) and (lee_1210 - lee_min) != 0
                else np.nan
            )
            rows.append(
                {
                    "time": case["key"],
                    "title": case["title"],
                    "rel_error": rel_error,
                    "elbow_rank": elbow_rank,
                    "elbow_scale_m": int(scales[elbow_idx]),
                    "scale_support": "unverified_input_labels",
                    "L_opt": lee_opt,
                    "L_min": lee_min,
                    "L_1210": lee_1210,
                    "increment": increment,
                    "n_median": int(sub["n"].median()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "elbow_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def replot() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    records = pd.read_csv(OUT_DIR / "lee_curves.csv")
    summary = write_elbow_summary(records)
    print("\nElbow summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    plot_combined(records)
    print(f"Wrote figure to {PLOT_DIR / 'Combined_Sensitivity.svg'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy table diagnostic; use the S5 entry for the manuscript experiment."
    )
    parser.add_argument("--legacy-table", action="store_true",
                        help="Explicitly select historical table-level uniform noise (not S5)")
    parser.add_argument(
        "--scale-dir",
        type=Path,
        default=None,
        help="Folder containing the four '* Multiscale.gpkg' tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder (default: ./RES next to this script).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from an existing lee_curves.csv without recomputing Lee's L.",
    )
    args = parser.parse_args(argv)
    if not args.legacy_table:
        parser.error("This legacy table route is not S5. Run '06-Downscaling Error Influence (S5).py' with --config, "
                     "or explicitly opt in using --legacy-table.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("LEGACY TABLE DIAGNOSTIC: uniform noise on aggregated values; not the S5 pixel-noise experiment.")
    configure_paths(args.scale_dir, args.out_dir)
    if args.plot_only:
        replot()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    print(f"Scale tables: {SCALE_RES}")
    print(f"Output:       {OUT_DIR}")
    print(f"Perturbation levels: {[error_label(value) for value in REL_ERRORS]}")
    print(f"Savitzky-Golay: window={SG_WINDOW}, order={SG_ORDER}")

    rng = np.random.default_rng(RNG_SEED)
    frames = [run_case(case, rng) for case in CASES]
    records = pd.concat(frames, ignore_index=True)
    records.to_csv(OUT_DIR / "lee_curves.csv", index=False, encoding="utf-8-sig")

    summary = write_elbow_summary(records)
    print("\nElbow summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    plot_combined(records)
    print(f"\nWrote curves to {OUT_DIR / 'lee_curves.csv'}")
    print(f"Wrote figures to {PLOT_DIR}")


if __name__ == "__main__":
    main()
