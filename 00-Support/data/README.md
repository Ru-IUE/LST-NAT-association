# Input data contract

Supply input data at the paths configured in `00-Support/config/example.json`; source files are opened for reading only.

| Input | Required content |
| --- | --- |
| stations | Point GeoPackage or Shapefile with defined CRS; unique `station_id` preferred; `UrG` codes 1=UA, 2=SA, 3=RA, 4=NA; monthly hour fields TEM10, TEM13, TEM22, TEM01 |
| native_lst | Four projected, single-band native TRIMS rasters, in degrees Celsius, with a defined nodata flag and transform |
| station_native | Output from `extract-native`, including TD_LST, AD_LST, TN_LST, AN_LST; prepared tables with these fields can also be supplied |
| station_covariates | station_native plus NDVI, NDBI, GAIA, ALTI, Slope, Soil; GAIA stores GLC_FCS30D impervious percentage |
| embedding_fine | Complete-region, north-up GeoTIFF in a projected metre CRS with 10-coordinate-unit pixels and exactly 64 bands ordered A1...A64, or directory A1.tif...A64.tif with identical grids; storage spacing does not define ground-distance supports |
| embedding_coarse | Four GEE reduceResolution(mean) 64-band exports on the exact respective native TRIMS grids (CRS, transform, rows and columns). The example configuration uses these for S3 training. Omit this mapping only to explicitly use the recorded local GDAL fallback. |
| legacy_multiscale | Optional GeoPackages for table diagnostics, containing UrG, geometry, hour-matched NAT, and L10, L30, ..., L1210; extra scales are ignored |
| legacy_prediction_csv | Optional per-station CSV folders containing station_id, Lon, Lat, Pred_LST on a regular lattice; used only by sample-csv diagnostics |

Station inputs must contain nonempty point geometries, a defined CRS, and valid UrG codes. Analyses report pair-specific complete-case sizes after IQR filtering.

If station_id is missing, the code assigns station_001, station_002, ... in input row order. Station IDs must be nonmissing and unique. When using per-station prediction CSVs, keep their IDs consistent with the station input. Supply the four-class UrG codes listed above.

Provide quality-controlled station composites with day alignment completed before analysis. The time-field mapping is T1=TEM10 (10:00–11:00), T2=TEM13 (13:00–14:00), T3=TEM22 (22:00–23:00), and T4=TEM01 (01:00–02:00 on the following day). The code does not repair day alignment in already averaged station inputs.

Fine raster correction requires complete coarse blocks. Cropped station CSVs cannot supply the true mean of all fine predictions in a coarse pixel. Reconstructing a constant residual independently in every station window would yield inconsistent values in overlapping windows and is deliberately not used.

The default support is `station_epsg3857_v1`: native LST and covariates use 1000-unit EPSG:3857 squares, product checks use 1500-unit squares, and nested scales use a station-centered grid with 10-coordinate-metre spacing and nearest resampling. L10 through L1210 are projected square side labels, not geodesic ground distances. The alternative `station_aeqd_wgs84_v1` uses local distance-based squares and average resampling; select it explicitly in `analysis.spatial_support`. Sampled tables carry `scale_support`, `support_lon`, `support_lat`, `source_crs`, `support_res_m`, and `resampling`. Main scale analysis requires a known marker matching the active configuration. Rerun sampling and analysis after changing support; never relabel cached results. Optional table/CSV diagnostics retain unverified input scale labels.

Nodata conventions: masked cells, non-finite values, and embedding sentinel -9999 are invalid. Native station LST extraction additionally excludes zero. DNN training retains finite zero LST.
