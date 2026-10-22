"""Run from the repository root; see README.md for the complete order."""
import argparse
import importlib.util
import json
from pathlib import Path
from lstnat.settings import ROOT, TIMES, load, input_path, output_path, save_run


def check(cfg):
    packages = ["numpy", "pandas", "scipy", "geopandas", "rasterio", "sklearn", "esda", "libpysal", "matplotlib", "tensorflow", "ee"]
    print("Dependencies:")
    for package in packages:
        print(f"  {package}: {'available' if importlib.util.find_spec(package) else 'missing'}")
    print("Inputs (some are created by preceding steps):")
    for key, value in cfg["inputs"].items():
        for time in value if isinstance(value, dict) else [None]:
            path = input_path(cfg, key, time)
            print(f"  {key}{'.'+time if time else ''}: {'exists' if path.exists() else 'missing'}")
    print(f"Output directory: {output_path(cfg, '_location').parent}")


def main():
    parser = argparse.ArgumentParser(description="LST–NAT manuscript workflows")
    parser.add_argument("command", choices=["check", "extract-native", "correlation", "spatial", "weight-sensitivity",
                                            "compare-statistics", "train", "predict", "validate-product", "sample",
                                            "scale", "sensitivity", "legacy-scale", "sample-csv", "plot"])
    parser.add_argument("--config", default=str(ROOT/"00-Support"/"config"/"example.json"))
    parser.add_argument("--time", choices=["all", *TIMES], default="all")
    parser.add_argument("--permutations", type=int, help="Override for diagnostics; use 999 for manuscript analysis")
    parser.add_argument("--bootstrap", type=int, help="Override for diagnostics; use 1000 for manuscript analysis")
    args = parser.parse_args()
    cfg = load(args.config)
    cfg.setdefault("analysis", {})
    for name in ["permutations", "bootstrap"]:
        value = getattr(args, name)
        if value is not None:
            if value < 0:
                parser.error(name+" must be nonnegative")
            cfg["analysis"][name] = value
    if cfg["analysis"].get("permutations", 999) != 999 or cfg["analysis"].get("bootstrap", 1000) != 1000:
        print("DIAGNOSTIC RUN: settings differ from manuscript; see recorded configuration.")
    times = list(TIMES) if args.time == "all" else [args.time]
    if args.command == "check":
        check(cfg)
    elif args.command == "extract-native":
        from lstnat.raster import extract_native
        extract_native(cfg)
    elif args.command in ["correlation", "spatial", "weight-sensitivity", "compare-statistics", "scale", "legacy-scale"]:
        from lstnat import association, scaling
        if args.command == "correlation":
            association.correlation_analysis(cfg)
        elif args.command in ["spatial", "weight-sensitivity"]:
            association.spatial_analysis(cfg, sensitivity=args.command == "weight-sensitivity")
        elif args.command == "compare-statistics":
            association.compare_statistics(cfg)
        else:
            if args.command == "legacy-scale":
                print("LEGACY DIAGNOSTIC: original multiscale inputs are not certified residual-corrected fields.")
            scaling.scale_analysis(cfg, legacy=args.command == "legacy-scale")
    elif args.command in ["train", "predict", "validate-product"]:
        from lstnat import downscaling
        if args.command == "validate-product":
            downscaling.validate_product(cfg)
        else:
            getattr(downscaling, args.command)(cfg, times)
    elif args.command == "sample":
        from lstnat.raster import sample
        sample(cfg)
    elif args.command == "sample-csv":
        import pandas as pd
        from lstnat.raster import read_stations, sample_csv_file
        stations = read_stations(input_path(cfg, "stations"))
        for time in times:
            folder = input_path(cfg, "legacy_prediction_csv", time)
            rows = [sample_csv_file(p, stations) for p in sorted(folder.glob("*.csv"))]
            if not rows:
                raise ValueError(f"No CSVs in {folder}")
            table = pd.DataFrame(rows)
            if table.station_id.duplicated().any():
                raise ValueError("Duplicate station CSVs")
            merged = stations.merge(table, on="station_id", how="left", validate="one_to_one")
            merged.to_file(output_path(cfg, "legacy_diagnostic", time+"_csv_scales.gpkg"), driver="GPKG", layer="stations")
        save_run(cfg, "sample-csv", note="Legacy unverified fine-grid predictions; not the corrected manuscript workflow")
    elif args.command == "sensitivity":
        from lstnat.sensitivity import run
        run(cfg)
    elif args.command == "plot":
        from lstnat.plotting import run
        run(cfg)


if __name__ == "__main__":
    main()
