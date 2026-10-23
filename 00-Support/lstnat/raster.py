"""Read-only source raster access, native extraction, and nested fine-grid sampling."""
from pathlib import Path
import numpy as np
import pandas as pd


# Keep the existing AEQD helpers available for explicitly selected diagnostics.
SCALE_SUPPORT = "station_aeqd_wgs84_v1"
MANUSCRIPT_SUPPORT = "station_epsg3857_v1"


def configured_support(cfg):
    support = cfg.get("analysis", {}).get("spatial_support", MANUSCRIPT_SUPPORT)
    if support not in [MANUSCRIPT_SUPPORT, SCALE_SUPPORT]:
        raise ValueError("Unknown spatial_support: " + str(support))
    return support


def support_metadata(support):
    return {"scale_support": support,
            "scale_definition": ("square side length in EPSG:3857 coordinate metres"
                                 if support == MANUSCRIPT_SUPPORT else
                                 "square side length in station-centered WGS84 AEQD metres"),
            "sampling_resolution_m": 10,
            "sampling_resampling": "nearest" if support == MANUSCRIPT_SUPPORT else "average"}

SUPPORT_METADATA = {
    "scale_support": SCALE_SUPPORT,
    "scale_definition": "square side length in station-centered WGS84 AEQD metres",
    "sampling_resolution_m": 10,
    "sampling_resampling": "average",
}


def station_support(lon, lat, side_m, support=SCALE_SUPPORT):
    """Construct a station-centered square in the selected coordinate system."""
    from pyproj import CRS
    from shapely.geometry import box
    from shapely import segmentize
    if not np.isfinite([lon, lat, side_m]).all() or not -90 < lat < 90 or side_m <= 0:
        raise ValueError("Expected finite station coordinates and a positive support size")
    if support == MANUSCRIPT_SUPPORT:
        from pyproj import Transformer
        x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
        return CRS.from_epsg(3857), segmentize(
            box(x-side_m/2, y-side_m/2, x+side_m/2, y+side_m/2), 25)
    if support != SCALE_SUPPORT:
        raise ValueError("Unknown spatial support")
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat:.12f} +lon_0={lon:.12f} +datum=WGS84 +units=m +no_defs")
    # Densify before transforming: projected window edges need not stay straight.
    square = segmentize(box(-side_m/2, -side_m/2, side_m/2, side_m/2), 25)
    return crs, square


def local_grid(src, lon, lat, side_m=1210, pixel_m=10, support=SCALE_SUPPORT):
    """Place the field on a station-centered grid with an explicit support CRS."""
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    crs, square = station_support(lon, lat, side_m, support)
    size = int(round(side_m/pixel_m))
    if size < 1 or not np.isclose(size*pixel_m, side_m):
        raise ValueError("Support side must contain a whole number of local pixels")
    transform = from_origin(square.bounds[0], square.bounds[3], pixel_m, pixel_m)
    with WarpedVRT(src, crs=crs, transform=transform, width=size, height=size,
                   resampling=(Resampling.nearest if support == MANUSCRIPT_SUPPORT else Resampling.average),
                   dtype="float64", nodata=np.nan,
                   tolerance=1e-6) as vrt:
        array = vrt.read(1, masked=True).astype(float).filled(np.nan)
    return array, transform, crs


def require_scale_support(frame, expected=None):
    """Reject unknown, mixed, or configuration-incompatible scale definitions."""
    if (frame.empty or "scale_support" not in frame or frame.scale_support.nunique(dropna=False) != 1
            or not frame.scale_support.isin([SCALE_SUPPORT, MANUSCRIPT_SUPPORT]).all()
            or (expected is not None and not frame.scale_support.eq(expected).all())):
        raise ValueError("Spatial support provenance is missing or incompatible; rerun sample before scale or plot")


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


def weighted_native(src, square, support_crs=None):
    """Weight intersections in support_crs; default accepts a source-CRS polygon."""
    from rasterio.features import geometry_window
    from rasterio.errors import WindowError
    from shapely.geometry import Polygon
    from shapely.ops import transform as transform_geometry
    from shapely import segmentize
    from pyproj import Transformer
    search_square = square
    to_support = None
    if support_crs is not None:
        to_source = Transformer.from_crs(support_crs, src.crs, always_xy=True).transform
        to_support = Transformer.from_crs(src.crs, support_crs, always_xy=True).transform
        search_square = transform_geometry(to_source, square)
    try:
        window = geometry_window(src, [search_square.__geo_interface__])
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
        if to_support is not None:
            # Subdivide each native-cell edge before its nonlinear transformation.
            edge = max(np.hypot(tr.a, tr.d), np.hypot(tr.b, tr.e))/32
            polygon = transform_geometry(to_support, segmentize(polygon, edge))
        area = square.intersection(polygon).area
        numerator += float(value) * area
        denominator += area
    return numerator / denominator if denominator else np.nan


def extract_native(cfg):
    import rasterio
    from .settings import TIMES, input_path, output_path, save_run
    frame = read_stations(input_path(cfg, "stations"))
    points = frame.to_crs(4326).geometry
    support = configured_support(cfg)
    supports = [station_support(p.x, p.y, 1000, support) for p in points]
    for time, spec in TIMES.items():
        with rasterio.open(input_path(cfg, "native_lst", time)) as src:
            if not src.crs or not src.crs.is_projected:
                raise ValueError("Native area-weighted extraction requires a projected raster")
            frame[spec["lst"]] = [weighted_native(src, square, crs) for crs, square in supports]
    frame["native_support"] = support
    frame["native_side_m"] = 1000
    path = output_path(cfg, "preprocessing", "stations_native.gpkg")
    frame.to_file(path, driver="GPKG", layer="stations")
    save_run(cfg, "extract-native", n=len(frame), result=str(path),
             support=support, side_m=1000, area_weighting="intersection area in support CRS")


def nested_means(grid, center_row, center_col, scales):
    """Aggregate an already prepared local 10 m grid; this function has no CRS."""
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
    from pyproj import CRS
    crs = CRS.from_user_input(src.crs) if src.crs else None
    metric = crs is not None and crs.is_projected and all(
        np.isclose(axis.unit_conversion_factor, 1) for axis in crs.axis_info[:2])
    if (not metric or not np.allclose(src.res, [10, 10]) or src.transform.b or src.transform.d
            or src.transform.a <= 0 or src.transform.e >= 0):
        raise ValueError("Fine raster must be north-up with 10-unit pixels in a projected metre CRS; "
                         "storage pixel sizes are not ground-distance support sizes")


def sample_raster(src, stations, support=SCALE_SUPPORT):
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as transform_geometry
    from .settings import SCALES
    validate_fine(src)
    geographic = stations.to_crs(4326)
    rows = []
    footprint = box(*src.bounds)
    for p in geographic.geometry:
        grid, _, crs = local_grid(src, p.x, p.y, SCALES[-1], support=support)
        values = nested_means(grid, 60, 60, SCALES)
        to_source = Transformer.from_crs(crs, src.crs, always_xy=True).transform
        _, outer = station_support(p.x, p.y, SCALES[-1], support)
        if not footprint.covers(transform_geometry(to_source, outer)):
            # An extent-clipped support must not be reported as a complete square.
            for side in SCALES:
                _, square = station_support(p.x, p.y, side, support)
                if not footprint.covers(transform_geometry(to_source, square)):
                    values[f"L{side}"] = np.nan
        rows.append(values)
    frame = stations.copy()
    for column in rows[0]:
        frame[column] = [row[column] for row in rows]
    frame["scale_support"] = support
    frame["support_lon"] = geographic.geometry.x.to_numpy()
    frame["support_lat"] = geographic.geometry.y.to_numpy()
    frame["source_crs"] = src.crs.to_string()
    frame["support_res_m"] = 10
    frame["resampling"] = support_metadata(support)["sampling_resampling"]
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
            frame = sample_raster(src, stations, configured_support(cfg))
        frame.to_file(output_path(cfg, "multiscale", f"{time}.gpkg"), driver="GPKG", layer="stations")
    save_run(cfg, "sample", **support_metadata(configured_support(cfg)))


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
    return {"station_id": sid, "scale_support": "unverified_input_labels",
            **nested_means(grid.to_numpy(float), r, c, SCALES)}
