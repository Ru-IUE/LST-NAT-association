"""Read-only source raster access, native extraction, and nested fine-grid sampling."""
from pathlib import Path
import numpy as np
import pandas as pd


def read_stations(path):
    import geopandas as gpd
    frame = gpd.read_file(path)
    if frame.crs is None or frame.empty or not frame.geometry.geom_type.eq("Point").all():
        raise ValueError("Stations must be nonempty points with a defined CRS")
    if "UrG" not in frame or not frame["UrG"].isin([1, 2, 3, 4]).all():
        raise ValueError("Supply merged UrG codes 1=UA, 2=SA, 3=RA, 4=NA")
    if "station_id" not in frame:
        frame["station_id"] = [f"station_{i+1:03d}" for i in range(len(frame))]
    if frame["station_id"].isna().any() or frame["station_id"].duplicated().any():
        raise ValueError("station_id must be nonmissing and unique")
    return frame


def weighted_native(src, square):
    """Intersection-area-weighted valid cells; square is already in raster CRS."""
    from rasterio.features import geometry_window
    from rasterio.errors import WindowError
    from shapely.geometry import Polygon
    try:
        window = geometry_window(src, [square.__geo_interface__])
    except WindowError:
        return np.nan
    data = src.read(1, window=window, masked=True)
    tr = src.window_transform(window)
    numerator = denominator = 0.0
    for r, c in np.ndindex(data.shape):
        value = data[r, c]
        if np.ma.is_masked(value) or not np.isfinite(value) or value == 0:
            continue
        polygon = Polygon([tr * p for p in [(c, r), (c+1, r), (c+1, r+1), (c, r+1)]])
        area = square.intersection(polygon).area
        numerator += float(value) * area
        denominator += area
    return numerator / denominator if denominator else np.nan


def extract_native(cfg):
    import rasterio
    import geopandas as gpd
    from shapely.geometry import box
    from .settings import TIMES, input_path, output_path, save_run
    frame = read_stations(input_path(cfg, "stations"))
    metric = frame.to_crs(3857)
    squares = gpd.GeoSeries([box(p.x-500, p.y-500, p.x+500, p.y+500) for p in metric.geometry], crs=3857)
    for time, spec in TIMES.items():
        with rasterio.open(input_path(cfg, "native_lst", time)) as src:
            if not src.crs or not src.crs.is_projected:
                raise ValueError("Native area-weighted extraction requires a projected raster")
            projected_squares = squares.to_crs(src.crs)
            frame[spec["lst"]] = [weighted_native(src, square) for square in projected_squares]
    path = output_path(cfg, "preprocessing", "stations_native.gpkg")
    frame.to_file(path, driver="GPKG", layer="stations")
    save_run(cfg, "extract-native", n=len(frame), result=str(path))


def nested_means(grid, center_row, center_col, scales):
    """Each scale is a side length in metres on a 10 m lattice; no edge truncation."""
    from .settings import SCALES
    if tuple(scales) != SCALES:
        raise ValueError("Expected 61 scales from 10 to 1210 m")
    array = np.asarray(grid, float)
    sums = np.pad(np.where(np.isfinite(array), array, 0).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = np.pad(np.isfinite(array).astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    result = {}
    for scale in scales:
        radius = (scale // 10) // 2
        r0, r1 = center_row-radius, center_row+radius+1
        c0, c1 = center_col-radius, center_col+radius+1
        if r0 < 0 or c0 < 0 or r1 > array.shape[0] or c1 > array.shape[1]:
            result[f"L{scale}"] = np.nan
            continue
        def total(prefix):
            return prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]
        count = total(counts)
        result[f"L{scale}"] = float(total(sums) / count) if count else np.nan
    return result


def validate_fine(src):
    if src.crs != "EPSG:3857" or not np.allclose(src.res, [10, 10]) or src.transform.b or src.transform.d:
        raise ValueError("Fine LST must use a north-up EPSG:3857 grid with 10 m pixels")


def sample_raster(src, stations):
    from rasterio.windows import Window
    from .settings import SCALES
    validate_fine(src)
    metric = stations.to_crs(src.crs)
    rows = []
    for p in metric.geometry:
        r, c = src.index(p.x, p.y)
        grid = src.read(1, window=Window(c-60, r-60, 121, 121), boundless=True,
                        masked=True).astype(float).filled(np.nan)
        rows.append(nested_means(grid, 60, 60, SCALES))
    frame = stations.copy()
    for column in rows[0]:
        frame[column] = [row[column] for row in rows]
    return frame


def sample(cfg):
    import rasterio
    from .settings import TIMES, input_path, output_path, save_run
    stations = read_stations(input_path(cfg, "stations"))
    for time in TIMES:
        path = output_path(cfg, "downscaling", f"{time}_corrected.tif")
        with rasterio.open(path) as src:
            if src.tags().get("residual_corrected") != "true":
                raise ValueError("A certified residual-corrected fine field is required")
            frame = sample_raster(src, stations)
        frame.to_file(output_path(cfg, "multiscale", f"{time}.gpkg"), driver="GPKG", layer="stations")
    save_run(cfg, "sample")


def sample_csv_file(csv_path, stations):
    """Legacy station CSV diagnostic; no claim that inputs are residual corrected."""
    from .settings import SCALES
    df = pd.read_csv(csv_path)
    if df["station_id"].nunique() != 1:
        raise ValueError(f"CSV must contain exactly one station_id: {csv_path}")
    sid = str(df["station_id"].iloc[0])
    matches = stations[stations.station_id.eq(sid)]
    if len(matches) != 1:
        raise ValueError(f"Cannot match {sid} uniquely")
    if df.duplicated(["Lon", "Lat"]).any():
        raise ValueError("Duplicate pixel coordinates")
    # Pivot keeps missing lattice cells as NaN rather than guessing a rectangular reshape.
    grid = df.pivot(index="Lat", columns="Lon", values="Pred_LST").sort_index().sort_index(axis=1)
    for axis in [grid.index.to_numpy(float), grid.columns.to_numpy(float)]:
        if len(axis) > 2 and not np.allclose(np.diff(axis), np.median(np.diff(axis)), rtol=1e-4, atol=1e-10):
            raise ValueError("Legacy CSV coordinates are not a regular lattice")
    point = matches.to_crs(4326).geometry.iloc[0]
    r = int(np.abs(grid.index.to_numpy(float)-point.y).argmin())
    c = int(np.abs(grid.columns.to_numpy(float)-point.x).argmin())
    return {"station_id": sid, **nested_means(grid.to_numpy(float), r, c, SCALES)}
