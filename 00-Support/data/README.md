# Input data contract

Supply input data at the paths configured in `00-Support/config/example.json`; source files are opened for reading only.

| Input | Required content |
| --- | --- |
| stations | Point GeoPackage or Shapefile with defined CRS; unique `station_id` preferred; `UrG` codes 1=UA, 2=SA, 3=RA, 4=NA; monthly hour fields TEM10, TEM13, TEM22, TEM01 |
| native_lst | Four projected, single-band native TRIMS rasters, in degrees Celsius, with a defined nodata flag and transform |
| station_native | Output from `extract-native`, including TD_LST, AD_LST, TN_LST, AN_LST; prepared tables with these fields can also be supplied |
| station_covariates | station_native plus NDVI, NDBI, GAIA, ALTI, Slope, Soil; GAIA stores GLC_FCS30D impervious percentage |
| embedding_fine | Complete-region EPSG:3857, north-up 10 m GeoTIFF with exactly 64 bands ordered A1...A64, or directory A1.tif...A64.tif with identical grids |
| legacy_multiscale | Optional GeoPackages for table diagnostics, containing UrG, geometry, hour-matched NAT, and L10, L30, ..., L1210; extra scales are ignored |
| legacy_prediction_csv | Optional per-station CSV folders containing station_id, Lon, Lat, Pred_LST on a regular lattice; used only by sample-csv diagnostics |

Station inputs must contain nonempty point geometries, a defined CRS, and valid UrG codes. Analyses report pair-specific complete-case sizes after IQR filtering.

If station_id is missing, the code assigns station_001, station_002, ... in input row order. Station IDs must be nonmissing and unique. When using per-station prediction CSVs, keep their IDs consistent with the station input. Supply the four-class UrG codes listed above.

Provide quality-controlled station composites with day alignment completed before analysis. The time-field mapping is T1=TEM10, T2=TEM13, T3=TEM22, and T4=TEM01.

Fine raster correction requires complete coarse blocks. Cropped station CSVs cannot supply the true mean of all fine predictions in a coarse pixel. Reconstructing a constant residual independently in every station window would yield inconsistent values in overlapping windows and is deliberately not used.

Nodata conventions: masked cells, non-finite values, and embedding sentinel -9999 are invalid. Native station LST extraction additionally excludes zero. DNN training retains finite zero LST.
