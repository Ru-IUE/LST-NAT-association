"""S1.3 covariates; Earth Engine reads with local output only."""
import argparse
import sys
from pathlib import Path
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "00-Support" / "lstnat").is_dir())
sys.path.insert(0, str(ROOT_DIR / "00-Support"))
from lstnat.settings import ROOT, load, input_path, output_path, save_run


def main():
    import ee
    import json
    import pandas as pd
    from lstnat.raster import read_stations
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT/"00-Support/config/example.json"))
    args = parser.parse_args()
    cfg = load(args.config)
    ee.Initialize(project=cfg["earth_engine"]["project"])
    frame = read_stations(input_path(cfg, "station_native")).to_crs(4326)
    frame["_uid"] = range(len(frame))
    points = ee.FeatureCollection(json.loads(frame[["_uid", "geometry"]].to_json()))
    proj = ee.Projection("EPSG:3857")
    def square(feature):
        geom = feature.geometry().transform(proj, 1).buffer(500).bounds(1, proj)
        return ee.Feature(geom.transform("EPSG:4326", 1)).copyProperties(feature)
    windows = points.map(square)
    surface = ee.ImageCollection("MODIS/061/MOD09A1").filterDate("2020-10-01", "2020-11-01")
    def ndvi(image):
        return image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
    def ndbi(image):
        return image.normalizedDifference(["sur_refl_b06", "sur_refl_b02"]).rename("NDBI")
    annual = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual").mosaic().rename([str(y) for y in range(2000, 2023)])
    impervious = annual.select("2020").eq(190).multiply(100).rename("GAIA").toFloat()
    # The manuscript specifies the void-filled approximately 90 m SRTM product.
    elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation").rename("ALTI")
    slope = ee.Terrain.slope(elevation).rename("Slope")
    soil = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate("2020-10-01", "2020-11-01").select("soil").mean().multiply(0.1).rename("Soil")
    configs = [("NDVI", surface.map(ndvi).mean(), 500), ("NDBI", surface.map(ndbi).mean(), 500),
               ("GAIA", impervious, 300), ("ALTI", elevation, 90), ("Slope", slope, 90), ("Soil", soil, 4638.3)]
    for name, image, scale in configs:
        stats = image.reduceRegions(collection=windows, reducer=ee.Reducer.mean(), scale=scale)
        table = pd.DataFrame([feature["properties"] for feature in stats.getInfo()["features"]])
        value = "mean" if "mean" in table else name
        frame = frame.drop(columns=[name], errors="ignore").merge(
            table[["_uid", value]].rename(columns={value: name}), on="_uid", how="left", validate="one_to_one")
    frame = frame.drop(columns="_uid")
    frame.to_file(output_path(cfg, "preprocessing", "stations_covariates.gpkg"), driver="GPKG", layer="stations")
    save_run(cfg, "covariates", ndvi_ndbi_qa_filter=False, elevation_processing_scale_m=90)


if __name__ == "__main__":
    main()
