"""Scale-response tables and knee summaries."""
import numpy as np
import pandas as pd
from .settings import TIMES, SCALES, ZONES, input_path, output_path, setting, save_run
from .statistics import correlations, knee
from .association import complete_spatial, weights
from .spatial import lee
from .raster import require_scale_support, configured_support


def multiscale_path(cfg, time, legacy=False):
    return input_path(cfg, "legacy_multiscale", time) if legacy else output_path(cfg, "multiscale", f"{time}.gpkg")


def scale_frame(frame, nat, cfg, label="original", legacy=False):
    if not legacy:
        require_scale_support(frame, configured_support(cfg))
    rows = []
    for scale in SCALES:
        col = f"L{scale}"
        clean = complete_spatial(frame, col, nat)
        w, meta = weights(clean)
        # No local permutations needed to obtain means of local contributions.
        value, p, local, _ = lee(clean[nat], clean[col], w,
                                 setting(cfg, "permutations", 999), setting(cfg, "seed", 42), local_permutations=0)
        rho = correlations(clean, col, nat)
        rows.append({"scenario": label, "scale_m": scale, "zone": "GA", "n": len(clean),
                     "metric": "lee", "value": value, "p": p})
        rows.append({"scenario": label, "scale_m": scale, "zone": "GA", "n": len(clean),
                     "metric": "spearman", "value": rho["rho"], "p": rho["p"]})
        for zone in ZONES:
            vals = local[clean.UrG.eq(zone).to_numpy()]
            rows.append({"scenario": label, "scale_m": scale, "zone": ZONES[zone], "n": len(vals),
                         "metric": "mean_local_lee", "value": float(vals.mean()) if len(vals) else np.nan,
                         "p": np.nan})
    result = pd.DataFrame(rows)
    result["scale_support"] = "unverified_input_labels" if legacy else configured_support(cfg)
    return result


def summarize_curves(curves, cfg):
    summaries, smooth_frames = [], []
    if "scale_support" not in curves or curves.scale_support.isna().any():
        raise ValueError("Scale support metadata is required")
    for keys, group in curves.groupby(["time", "scenario", "zone", "metric"]):
        group = group.sort_values("scale_m").copy()
        if group.scale_m.tolist() != list(SCALES):
            raise ValueError("Incomplete scale grid")
        if group.scale_support.nunique() != 1:
            raise ValueError("Mixed scale definitions in one curve")
        smooth, result = knee(group.scale_m, group.value,
                               setting(cfg, "sg_window", 7), setting(cfg, "sg_order", 2))
        group["smoothed"] = smooth
        smooth_frames.append(group)
        summaries.append({**dict(zip(["time", "scenario", "zone", "metric"], keys)), **result,
                          "scale_support": group.scale_support.iloc[0]})
    return pd.concat(smooth_frames, ignore_index=True), pd.DataFrame(summaries)


def scale_analysis(cfg, legacy=False):
    import geopandas as gpd
    frames = []
    for time, spec in TIMES.items():
        frame = gpd.read_file(multiscale_path(cfg, time, legacy))
        result = scale_frame(frame, spec["nat"], cfg, legacy=legacy)
        result["time"] = time
        frames.append(result)
    curves, knees = summarize_curves(pd.concat(frames, ignore_index=True), cfg)
    folder = "legacy_diagnostic" if legacy else "scale"
    curves.to_csv(output_path(cfg, folder, "curves.csv"), index=False)
    knees.to_csv(output_path(cfg, folder, "knees.csv"), index=False)
    save_run(cfg, "scale", legacy_inputs=legacy,
             scale_support="unverified_input_labels" if legacy else configured_support(cfg),
             note="Legacy inputs have no certified residual-correction provenance" if legacy else "corrected fields")
