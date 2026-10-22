# Input data contract

Data are not bundled. Paths are configured in `00-Support/config/example.json`; source files are opened for reading only.

| Input | Required content |
| --- | --- |
| stations | Point GeoPackage or Shapefile with defined CRS; unique `station_id` preferred; `UrG` codes 1=UA, 2=SA, 3=RA, 4=NA; monthly hour fields TEM10, TEM13, TEM22, TEM01 |
| native_lst | Four projected, single-band native TRIMS rasters, in degrees Celsius, with a defined nodata flag and transform |
| station_native | Output from `extract-native`, including TD_LST, AD_LST, TN_LST, AN_LST; existing prepared tables can be used explicitly for diagnostics |
| station_covariates | station_native plus NDVI, NDBI, GAIA, ALTI, Slope, Soil; GAIA is the historical column name for GLC_FCS30D impervious percentage |
| embedding_fine | Complete-region EPSG:3857, north-up 10 m GeoTIFF with exactly 64 bands ordered A1...A64, or directory A1.tif...A64.tif with identical grids |
| legacy_multiscale | Optional old GeoPackages containing UrG, geometry, hour-matched NAT, and L10, L30, ..., L1210; extra scales are ignored |
| legacy_prediction_csv | Optional per-station CSV folders containing station_id, Lon, Lat, Pred_LST on a regular lattice; used only by sample-csv diagnostics |

Expected station frame after the supplied upstream quality control: 1,427 stations, with UA=249, SA=281, RA=416, NA=481. Code does not invent, delete, or rebalance raw stations to force these counts. Analyses report pair-specific complete-case sizes after IQR filtering.

If station_id is missing, the code assigns station_001, station_002, ... in input row order, matching the original export notebook. Preserve this order when matching legacy CSVs, or supply the explicit ID mapping used during export. The original local station table has already merged UC and UA; no five-class numeric-code mapping is guessed.

The supplied native station records are October 2020 hourly/monthly composites. Raw timestamp records and the original station screening logs are absent, so the initial quality-control and next-day alignment stages remain input requirements. The code uses TEM22 consistently at T3, rather than mixing TEM22 and TEM23 between workflows.

Fine raster correction requires complete coarse blocks. Cropped station CSVs cannot supply the true mean of all fine predictions in a coarse pixel. Reconstructing a constant residual independently in every station window would yield inconsistent values in overlapping windows and is deliberately not used.

Nodata conventions: masked cells, non-finite values, and embedding sentinel -9999 are invalid. Native station LST extraction additionally excludes zero as specified in S1.1. DNN training retains finite zero LST because S3.1 does not specify that exclusion for the training sample.
