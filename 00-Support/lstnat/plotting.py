"""Figures generated exclusively from exported analysis tables; no value offsets."""
import numpy as np
import pandas as pd
from .settings import TIMES, ZONES, output_path, save_run
from .raster import require_scale_support, configured_support, MANUSCRIPT_SUPPORT


def run(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    for folder in ["scale", "legacy_diagnostic", "sensitivity"]:
        name = "noise_curves.csv" if folder == "sensitivity" else "curves.csv"
        path = output_path(cfg, folder, name)
        if not path.exists():
            continue
        # Preserve the natural-area label "NA" instead of parsing it as missing.
        data = pd.read_csv(path, keep_default_na=False, na_values=[""])
        knees = pd.read_csv(output_path(cfg, folder, "noise_knees.csv" if folder == "sensitivity" else "knees.csv"), keep_default_na=False, na_values=[""])
        if folder != "legacy_diagnostic":
            require_scale_support(data, configured_support(cfg))
            require_scale_support(knees, configured_support(cfg))
        scale_label = "Input scale label (ground distance unverified)" if folder == "legacy_diagnostic" else (
            "Square side length (EPSG:3857 m)" if configured_support(cfg) == MANUSCRIPT_SUPPORT else "Local square side length (AEQD m)")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, time in zip(axes.ravel(), TIMES):
            subset = data[(data.time == time) & (data.zone == "GA") & (data.metric == "lee")]
            for scenario, group in subset.groupby("scenario"):
                group = group.sort_values("scale_m")
                line, = ax.plot(group.scale_m, group.smoothed, label=scenario)
                ax.scatter(group.scale_m, group.value, s=6, alpha=.25, color=line.get_color())
                k = knees[(knees.time == time) & (knees.zone == "GA") & (knees.metric == "lee") & (knees.scenario == scenario)].iloc[0]
                if pd.notna(k.scale):
                    ax.scatter(k.scale, k.value, color=line.get_color(), s=25)
                    ax.axvline(k.scale, color=line.get_color(), lw=.6, ls="--", alpha=.6)
            ax.set(title=time, xlabel=scale_label, ylabel="Global Lee's L")
            ax.legend(frameon=False)
        for suffix in ["svg", "png"]:
            fig.savefig(output_path(cfg, "figures", folder+"_global."+suffix), dpi=300)
        plt.close(fig)
        if folder == "scale":
            # Supplementary S4: use identical raw tables and knee settings for both metrics.
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            for ax, time in zip(axes.ravel(), TIMES):
                right = ax.twinx()
                for target, metric, color in [(ax, "spearman", "#456A9D"), (right, "lee", "#72567F")]:
                    group = data[(data.time == time) & (data.zone == "GA") & (data.metric == metric)].sort_values("scale_m")
                    target.plot(group.scale_m, group.smoothed, color=color)
                    k = knees[(knees.time == time) & (knees.zone == "GA") & (knees.metric == metric)].iloc[0]
                    if pd.notna(k.scale):
                        target.scatter(k.scale, k.value, color=color, marker="s" if metric == "spearman" else "o")
                    target.set_ylabel("Spearman's rho" if metric == "spearman" else "Global Lee's L", color=color)
                ax.set(title=time, xlabel=scale_label)
            fig.savefig(output_path(cfg, "figures", "S4_metric_comparison.svg"))
            plt.close(fig)
        if folder == "scale":
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
            for ax, time in zip(axes.ravel(), TIMES):
                for zone in ZONES.values():
                    group = data[(data.time == time) & (data.zone == zone) & (data.metric == "mean_local_lee")].sort_values("scale_m")
                    ax.plot(group.scale_m, group.smoothed, label=zone)
                ax.set(title=time, xlabel=scale_label, ylabel="Mean local Lee's L")
                ax.legend(frameon=False)
            fig.savefig(output_path(cfg, "figures", "zones.svg"))
            plt.close(fig)
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
            palette = ["#C47A3A", "#9673A6", "#5B92C8", "#709A72"]
            for row, (label, times) in enumerate([("Day", ["T1", "T2"]), ("Night", ["T3", "T4"]) ]):
                ax, scale_ax = axes[row]
                distributions, positions, colors = [], [], []
                for i, zone in enumerate(ZONES.values()):
                    for j, time in enumerate(times):
                        group = data[(data.time == time) & (data.zone == zone) & (data.metric == "mean_local_lee")]
                        distributions.append(group.value.to_numpy())
                        positions.append(i*3+j)
                        colors.append(palette[i])
                        key = knees[(knees.time == time) & (knees.zone == zone) & (knees.metric == "mean_local_lee")].iloc[0]
                        if pd.notna(key.scale):
                            scale_ax.scatter(key.scale, i+(j-.5)*.18, color=palette[i], marker="o" if j == 0 else "s")
                violin = ax.violinplot(distributions, positions=positions, widths=.8, showmeans=True, showextrema=False)
                for body, color in zip(violin["bodies"], colors):
                    body.set_facecolor(color); body.set_alpha(.5)
                ax.set_xticks(positions, [time+" "+zone for zone in ZONES.values() for time in times], rotation=30)
                ax.set(title=label+" scale response distributions", ylabel="Mean local Lee's L")
                scale_ax.set_yticks(range(4), list(ZONES.values()))
                scale_ax.set(title=label+" knees", xlabel=scale_label)
            fig.savefig(output_path(cfg, "figures", "zone_distributions_and_knees.svg"))
            plt.close(fig)
    path = output_path(cfg, "association", "correlations.csv")
    if path.exists():
        data = pd.read_csv(path, keep_default_na=False, na_values=[""])
        balanced = pd.read_csv(output_path(cfg, "association", "balanced_correlations.csv"),
                               keep_default_na=False, na_values=[""])
        fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), constrained_layout=True)
        for ax, zone in zip(axes, ["GA", *ZONES.values()]):
            # S1.2 compares zones at equal sample size; GA remains the observed
            # full-region coefficient. Preserve raw zone estimates in the CSV.
            source = data if zone == "GA" else balanced
            sub = source[source.zone == zone].set_index("time").reindex(TIMES)
            rho = sub.rho if zone == "GA" else sub.balanced_rho
            partial = sub.partial_rho if zone == "GA" else sub.balanced_partial_rho
            ax.plot(list(TIMES), rho, marker="o", label="Spearman")
            ax.plot(list(TIMES), partial, marker="s", label="Partial Spearman")
            ax.set(title=zone, ylabel="Rank association")
        axes[0].legend(frameon=False, fontsize=8)
        fig.savefig(output_path(cfg, "figures", "correlations.svg"))
        plt.close(fig)
    # Training fit panels are available only after TensorFlow training has completed.
    paths = {time: output_path(cfg, "validation", time+"_evaluation.csv") for time in TIMES}
    if all(path.exists() for path in paths.values()):
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        for ax, (time, path) in zip(axes.ravel(), paths.items()):
            table = pd.read_csv(path)
            density = ax.hexbin(table.observed, table.predicted, gridsize=70, mincnt=1, cmap="viridis", rasterized=True)
            lo, hi = table[["observed", "predicted"]].min().min(), table[["observed", "predicted"]].max().max()
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=.8)
            ax.set(title=time, xlabel="Observed LST (C)", ylabel="Predicted LST (C)")
            fig.colorbar(density, ax=ax, label="Pixel count")
        fig.savefig(output_path(cfg, "figures", "S1_model_fit.svg"))
        plt.close(fig)
    save_run(cfg, "plot")
