"""S5: perturb the fine field ONCE, then aggregate the same field at every scale."""
import numpy as np
import pandas as pd
from .settings import TIMES, input_path, output_path, setting, save_run
from .raster import read_stations, sample_raster
from .scaling import scale_frame, summarize_curves


def perturb_raster(source, destination, sigma, seed):
    import rasterio
    rng = np.random.default_rng(seed)
    with rasterio.open(source) as src:
        if src.tags().get("residual_corrected") != "true":
            raise ValueError("Sensitivity requires the corrected baseline field")
        profile = src.profile.copy()
        with rasterio.open(destination, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                array = src.read(1, window=window, masked=True).astype(float)
                noise = rng.normal(0, sigma, size=array.shape)
                perturbed = array * (1 + noise)
                dst.write(perturbed.filled(src.nodata).astype(profile["dtype"]), 1, window=window)
            dst.update_tags(noise="pixelwise_multiplicative_Gaussian", sigma=str(sigma),
                            seed=str(seed), corrected_baseline="true")


def run(cfg):
    import rasterio
    stations = read_stations(input_path(cfg, "stations"))
    curves = []
    for i, (time, spec) in enumerate(TIMES.items()):
        original = output_path(cfg, "downscaling", time+"_corrected.tif")
        for j, sigma in enumerate([0.0, .05, .10]):
            path = original
            if sigma:
                path = output_path(cfg, "sensitivity", f"{time}_noise_{int(sigma*100):02d}.tif")
                perturb_raster(original, path, sigma, setting(cfg, "seed", 42)+100*i+j)
            with rasterio.open(path) as src:
                frame = sample_raster(src, stations)
            frame.to_file(output_path(cfg, "sensitivity", f"{time}_noise_{int(sigma*100):02d}.gpkg"), driver="GPKG", layer="stations")
            result = scale_frame(frame, spec["nat"], cfg, f"sigma={sigma:.2f}")
            result["time"] = time
            curves.append(result)
    curves, knees = summarize_curves(pd.concat(curves, ignore_index=True), cfg)
    curves.to_csv(output_path(cfg, "sensitivity", "noise_curves.csv"), index=False)
    knees.to_csv(output_path(cfg, "sensitivity", "noise_knees.csv"), index=False)
    save_run(cfg, "sensitivity", sigma=[0, .05, .10], noise_support="fine pixels before aggregation")
