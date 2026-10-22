"""Manuscript association analyses: rank correlations, Lee's L, and statistic comparison."""
import numpy as np
import pandas as pd
from .settings import TIMES, ZONES, COVARIATES, input_path, output_path, setting, save_run
from .raster import read_stations
from .statistics import pair_filter, correlations, balanced_correlations
from .spatial import adaptive_weights, lee, clusters, alternative_statistics


def complete_spatial(frame, lst, nat):
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    return pair_filter(frame, lst, nat).to_crs(3857)


def weights(frame, k=8, kernel="gaussian"):
    coords = np.column_stack([frame.geometry.x, frame.geometry.y]) / 1000
    return adaptive_weights(coords, k, kernel)


def correlation_analysis(cfg):
    gdf = read_stations(input_path(cfg, "station_covariates"))
    rows, balanced = [], []
    for time, spec in TIMES.items():
        frame = pair_filter(gdf, spec["lst"], spec["nat"])
        # rho and rho' in each reported row use identical complete covariate cases.
        for zone, group in [("GA", frame), *frame.groupby("UrG")]:
            stats = correlations(group, spec["lst"], spec["nat"], COVARIATES)
            rows.append({"time": time, "zone": ZONES.get(zone, zone), **stats,
                         "delta_rho": stats["rho"] - stats["partial_rho"]})
        b = balanced_correlations(frame, spec["lst"], spec["nat"], COVARIATES,
                                  setting(cfg, "bootstrap", 1000), setting(cfg, "seed", 42))
        balanced.extend({**row, "time": time, "zone": ZONES[row["zone"]]} for row in b)
    pd.DataFrame(rows).to_csv(output_path(cfg, "association", "correlations.csv"), index=False)
    pd.DataFrame(balanced).to_csv(output_path(cfg, "association", "balanced_correlations.csv"), index=False)
    save_run(cfg, "correlation")


def spatial_analysis(cfg, sensitivity=False):
    frame = read_stations(input_path(cfg, "station_native"))
    summary, zone_rows, composition_rows, independent_zone_rows = [], [], [], []
    settings = [(8, "gaussian")]
    if sensitivity:
        settings = [(4, "gaussian"), (8, "gaussian"), (12, "gaussian"), (8, "bisquare"), (8, "inverse_distance")]
    seed, permutations = setting(cfg, "seed", 42), setting(cfg, "permutations", 999)
    rng = np.random.default_rng(seed)
    for time, spec in TIMES.items():
        clean = complete_spatial(frame, spec["lst"], spec["nat"])
        for k, kernel in settings:
            w, meta = weights(clean, k, kernel)
            value, p, local, local_p = lee(clean[spec["nat"]], clean[spec["lst"]], w, permutations, seed)
            labels = clusters(clean[spec["nat"]], clean[spec["lst"]], w, local_p)
            summary.append({"time": time, "n": len(clean), **meta, "lee": value, "p": p,
                            **{f"pct_{c}": float(np.mean(labels == c)*100) for c in np.unique(labels)}})
            if sensitivity:
                continue
            result = clean.copy()
            result["local_lee"], result["p_value"], result["cluster"] = local, local_p, labels
            result.to_file(output_path(cfg, "association", f"{time}_lisa.gpkg"), driver="GPKG", layer="stations")
            minimum = int(clean.groupby("UrG").size().min())
            for zone, group in clean.groupby("UrG"):
                mask = clean.UrG.eq(zone).to_numpy()
                vals = local[mask]
                bootstrap = [float(np.mean(rng.choice(vals, minimum, replace=True)))
                             for _ in range(setting(cfg, "bootstrap", 1000))] if len(vals) > minimum else []
                # Table 2 uses local contributions from the complete spatial graph.
                zone_rows.append({"time": time, "zone": ZONES[zone], "n": len(group),
                                  "mean_local_lee": float(vals.mean()),
                                  "balanced_mean_local_lee": float(np.median(bootstrap)) if bootstrap else float(vals.mean()),
                                  "ci_low": float(np.percentile(bootstrap, 2.5)) if bootstrap else np.nan,
                                  "ci_high": float(np.percentile(bootstrap, 97.5)) if bootstrap else np.nan})
                zone_w, zone_meta = weights(group)
                zone_l, zone_p, _, _ = lee(group[spec["nat"]], group[spec["lst"]], zone_w,
                                          permutations, seed, local_permutations=0)
                independent_zone_rows.append({"time": time, "zone": ZONES[zone], "n": len(group),
                                              **zone_meta, "within_zone_global_lee": zone_l, "p": zone_p})
            for cluster in ["High-High", "Low-Low", "High-Low", "Low-High"]:
                total = int(np.sum(labels == cluster))
                for zone in ZONES:
                    count = int(np.sum((labels == cluster) & clean.UrG.eq(zone).to_numpy()))
                    composition_rows.append({"time": time, "cluster": cluster, "zone": ZONES[zone],
                                             "count": count, "cluster_total": total,
                                             "column_pct": count/total*100 if total else np.nan})
    name = "weights_sensitivity" if sensitivity else "global_lee"
    pd.DataFrame(summary).to_csv(output_path(cfg, "association", name+".csv"), index=False)
    if not sensitivity:
        pd.DataFrame(zone_rows).to_csv(output_path(cfg, "association", "zone_local_lee.csv"), index=False)
        pd.DataFrame(composition_rows).to_csv(output_path(cfg, "association", "cluster_composition.csv"), index=False)
        pd.DataFrame(independent_zone_rows).to_csv(output_path(cfg, "association", "within_zone_global_lee.csv"), index=False)
    save_run(cfg, name)
def compare_statistics(cfg):
    rows, local_rows = [], []
    frame = read_stations(input_path(cfg, "station_native"))
    p, seed = setting(cfg, "permutations", 999), setting(cfg, "seed", 42)
    for time, spec in TIMES.items():
        clean = complete_spatial(frame, spec["lst"], spec["nat"])
        w, _ = weights(clean)
        value, pv, local, lp = lee(clean[spec["nat"]], clean[spec["lst"]], w, p, seed)
        measures = {"lee": {"value": value, "p": pv, "local": local, "local_p": lp},
                    **alternative_statistics(clean[spec["nat"]], clean[spec["lst"]], w, p, seed)}
        from .spatial import zscore
        for name, result in measures.items():
            rows.append({"time": time, "metric": name, "n": len(clean), "value": result["value"], "p": result["p"]})
            if name == "lee":
                labels = clusters(clean[spec["nat"]], clean[spec["lst"]], w, result["local_p"])
            else:
                # Preserve source convention: focal NAT and lagged LST for Moran/Geary.
                a, b = zscore(clean[spec["nat"]]), w @ zscore(clean[spec["lst"]])
                labels = np.full(len(a), "Not significant", dtype=object)
                sig = result["local_p"] < 0.05
                labels[sig] = "Undefined"
                for sa, sb, label in [(1, 1, "High-High"), (-1, -1, "Low-Low"), (1, -1, "High-Low"), (-1, 1, "Low-High")]:
                    labels[sig & (b*sa > 0) & (a*sb > 0)] = label
            for label in np.unique(labels):
                local_rows.append({"time": time, "metric": name, "cluster": label,
                                   "count": int(np.sum(labels == label)), "percent": float(np.mean(labels == label)*100)})
    pd.DataFrame(rows).to_csv(output_path(cfg, "sensitivity", "statistics.csv"), index=False)
    pd.DataFrame(local_rows).to_csv(output_path(cfg, "sensitivity", "statistic_clusters.csv"), index=False)
    save_run(cfg, "compare-statistics")
