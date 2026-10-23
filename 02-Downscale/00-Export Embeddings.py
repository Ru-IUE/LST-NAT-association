"""Prepare Earth Engine exports; submission requires explicit --submit at runtime.

No export or authentication is performed on import. Download the fine export to
the configured embedding_fine path, preserving band order A1 ... A64.
"""
import argparse
import sys
from pathlib import Path
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "00-Support" / "lstnat").is_dir())
sys.path.insert(0, str(ROOT_DIR / "00-Support"))
from lstnat.settings import load, input_path, ROOT, TIMES


def main():
    import ee
    import rasterio
    from rasterio.warp import transform_bounds
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT/"00-Support/config/example.json"))
    parser.add_argument("--submit", action="store_true", help="Start Google Drive export tasks")
    args = parser.parse_args()
    cfg = load(args.config)
    ee.Initialize(project=cfg["earth_engine"]["project"])
    with rasterio.open(input_path(cfg, "native_lst", "T1")) as native:
        bounds = transform_bounds(native.crs, "EPSG:3857", *native.bounds, densify_pts=41)
    # Full native extent, not clipped station windows: required for full-block residuals.
    region = ee.Geometry.Rectangle(list(bounds), proj="EPSG:3857", geodesic=False)
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2020-01-01", "2021-01-01").filterBounds(region)
    first = ee.Image(collection.first())
    bands = [f"A{i:02d}" for i in range(64)]
    names = [f"A{i}" for i in range(1, 65)]
    embedding = collection.mosaic().select(bands, names).setDefaultProjection(first.select(0).projection())
    # EPSG:3857 matches the window-coordinate convention in Supplement S1/S3.
    # These projected coordinate lengths must not be described as geodesic distances.
    fine = embedding.reproject(crs="EPSG:3857", scale=10)
    folder = cfg["earth_engine"].get("drive_folder", "LST_NAT_exports")
    tasks = [("embedding_10m", ee.batch.Export.image.toDrive(
        image=fine.toFloat(), description="embedding_10m", folder=folder,
        fileNamePrefix="embedding_10m", region=region, crs="EPSG:3857", scale=10,
        maxPixels=1e13, fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True}))]
    # S3.1 training consumes these reduceResolution(mean) exports through
    # inputs.embedding_coarse. Each export is aligned to its native TRIMS grid.
    for time in TIMES:
        with rasterio.open(input_path(cfg, "native_lst", time)) as native:
            tr = native.transform
            transform = [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]
            crs = native.crs.to_wkt()
        aggregated = fine.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65536).reproject(crs=crs, crsTransform=transform)
        tasks.append((time+"_embedding_mean", ee.batch.Export.image.toDrive(
            image=aggregated.toFloat(), description=time+"_embedding_mean", folder=folder,
            fileNamePrefix=time+"_embedding_mean", region=region, crs=crs, crsTransform=transform,
            maxPixels=1e13, fileFormat="GeoTIFF")))
    for name, task in tasks:
        if args.submit:
            task.start()
            print(name, "submitted", task.id)
        else:
            print(name, "prepared; not submitted")


if __name__ == "__main__":
    main()
