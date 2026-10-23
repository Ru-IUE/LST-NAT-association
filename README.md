# LST-NAT association across urban–rural gradients

This directory contains code for analyzing associations between land surface temperature (LST) and near-surface air temperature (NAT) across urban–rural gradients. The workflow covers preprocessing, association analysis, downscaling, scale analysis, and sensitivity checks. Input data and trained weights must be supplied through the configuration files.

## Repository layout

```text
00-Support/
  run.py                   Command-line entry point for every step
  lstnat/                  Shared implementations (module map in 00-Support/README.md)
    settings.py            Constants, configuration, output containment, run records
    raster.py              Station input, area-weighted extraction, nested-scale sampling
    statistics.py          Pair filtering, rank correlations, knee detection
    spatial.py             Adaptive KNN weights, Lee/Moran/Geary statistics
    association.py         Correlation, Lee's L, and statistic-comparison workflows
    downscaling.py         DNN training, inference, complete-block residual correction
    scaling.py             Scale-response curves and knee summaries
    sensitivity.py         S5 pixel-noise experiment
    plotting.py            Figures generated from exported tables
  config/                  example.json (configuration template)
  data/README.md           Input data contract
  requirements/            base.txt, downscaling.txt, gee.txt
  tests/                   Method checks on synthetic data
  results/                 Generated analysis outputs and run records
01-Relationship/
  00-Collect Auxiliary Variables.py
  01-Correlation.ipynb
  02-Bivariate Lee's L.ipynb
02-Downscale/
  00-Export Embeddings.py
  01-Modeling Day Terra.ipynb
  02-Modeling Day Aqua.ipynb
  03-Modeling Night Terra.ipynb
  04-Modeling Night Aqua.ipynb
  05-Predict & Verify.ipynb
03-Scale-Effect/
  01-Sampling.py
  02-Scale Stats.ipynb / 02-Scale Stats (Lee).ipynb
  03-UrG Scale Effect.ipynb / 03-UrG Scale Effect (Lee).ipynb
  04-Plot UrG Scale Effect.ipynb
  05-Calc Residual Delta.py
  06-Downscaling Error Influence (S5).py
  06-Downscaling Error Influence (S6).py  Legacy table diagnostic only
```

The top-level folders follow execution order: `00-Support` holds the shared code and configuration, followed by the three workflow steps `01-Relationship` → `02-Downscale` → `03-Scale-Effect`. Inside each step folder the `NN-` prefix gives the execution order. `00-Collect Auxiliary Variables.py` and `00-Export Embeddings.py` provide the preprocessing entry points. Shared implementations live in `00-Support/lstnat`. Paired `(Lee)` and non-`(Lee)` notebooks are equivalent entry points that write the same unified tables; run either variant. `06-Downscaling Error Influence (S5).py` calls the shared pixelwise-noise workflow and plotting code. The old `(S6)` filename is retained only for historical table diagnostics and requires `--legacy-table`; it does not reproduce Supplement S5.

Notebooks are thin entry points: each one runs the matching `00-Support/run.py` command or reads the tables that command produced, so the notebook and command-line routes yield identical results. The module-by-module map is in [00-Support/README.md](00-Support/README.md).

## Environment

Python 3.10 or newer is required. `00-Support/requirements/` separates the analysis (`base.txt`), TensorFlow (`downscaling.txt`), and Earth Engine (`gee.txt`) groups; they specify minimum dependency versions. TensorFlow is only needed for training and inference, and `earthengine-api` only for the two scripts that query Earth Engine.

Install what you need (`pip install -r 00-Support/requirements/base.txt`, plus `downscaling.txt` or `gee.txt`), then run from the repository root:

```powershell
$py = 'C:\Users\15093\.conda\envs\Processing\python.exe'  # replace with your analysis environment
& $py 00-Support/run.py check --config 00-Support/config/example.json
& $py -m unittest discover -s 00-Support/tests -v
```

`check` reports which dependencies and configured inputs are available. Fill in the input paths before running the analysis; unavailable files are reported as `missing`.

## Inputs and configuration

Use `00-Support/config/example.json` as the public template. All relative paths resolve from this repository root, regardless of the working directory. Configure the files described in [00-Support/data/README.md](00-Support/data/README.md). Four time keys are used throughout:

| Key | Satellite observation | NAT input field | Native LST field |
| --- | --- | --- | --- |
| T1 | Terra day, 10:30 | TEM10 | TD_LST |
| T2 | Aqua day, 13:30 | TEM13 | AD_LST |
| T3 | Terra night, 22:30 | TEM22 | TN_LST |
| T4 | Aqua night, 01:30 | TEM01 | AN_LST |

Provide quality-controlled October 2020 station composites. The analysis frame in the manuscript contains 1,427 stations: UA 249, SA 281, RA 416, and NA 481, after merging UC into UA. The code accepts the prepared four-class `UrG` values rather than reclassifying stations. T1–T4 correspond to hourly NAT intervals 10:00–11:00, 13:00–14:00, 22:00–23:00 and 01:00–02:00. T4 is on the following day relative to the satellite day; prepare that alignment before monthly averaging. The code reads the configured hour fields directly and does not repeat raw-station quality control.

All generated products are restricted to `output_dir` inside `Upload Code`. The configuration template sets this to `00-Support/results`.

## Spatial support and distance units

The default `analysis.spatial_support` is `station_epsg3857_v1`, matching the EPSG:3857 window convention in Supplement S1.3 and S3.1. Station coordinates are projected to EPSG:3857 before constructing centered squares. Native LST and covariates use 1,000 × 1,000 coordinate-metre windows; product validation uses 1,500 × 1,500 windows (half-width 750). Corrected LST is sampled on a station-centered 10-unit grid and averaged over 61 nested squares with side labels 10, 30, ..., 1210. Nearest resampling preserves the source values when placing the fine field on the centered grid. Native LST extraction instead weights intersecting native cells by their area in the support CRS, excluding masked, non-finite, and zero values.

EPSG:3857 coordinate metres are not ellipsoidal ground metres: map scale varies with latitude. The scale labels follow the manuscript's projected-window convention and should be reported with that convention. The station-centered AEQD implementation remains available by explicitly setting `analysis.spatial_support` to `station_aeqd_wgs84_v1`; it uses local WGS84 distance-based windows and average resampling. This is an alternative support definition and can yield different knee scales. It is not an interchangeable reproduction of the submitted EPSG:3857 workflow.

Sampled tables record the support marker, station longitude/latitude, source CRS, resolution, and resampling method. `scale` and `plot` reject unknown or mixed markers and markers that disagree with the active configuration. Rerun `sample`, `scale`, and `sensitivity` when changing support definitions; existing AEQD outputs are not relabeled as EPSG:3857. Spatial KNN weights separately use Euclidean distances in EPSG:3857 kilometres, as specified in S2.

## Methods and supplementary sections

| Manuscript section | Implementation and reporting |
| --- | --- |
| S1 | Prepared hour-matched station composites; native LST extraction; six covariates: NDVI, NDBI, impervious percentage, elevation, slope, and soil moisture. Internal `GAIA` means GLC_FCS30D class-190 percentage, not the GAIA product. |
| S2 | Complete-pair filtering followed by inclusive 2×IQR bounds for both variables; adaptive KNN Gaussian weights with K=8, arithmetic-mean symmetrization and row standardization; 999 permutations, local p < 0.05; K=4/8/12 and kernel/statistic sensitivity. |
| S3 | Four independent 64-feature DNNs; GEE spatial-mean embeddings on the exact native grid; random 80/20 split with seed 3407; time-specific architectures and AdamW settings; block-constant residual correction. Both model and product accuracy checks use uncorrected predictions. |
| S4 | Global Spearman and Lee scale curves on the same 61-scale grid, using the same SG settings and normalized-deviation knee rule. Zone scale curves are means of local Lee contributions from the full-station graph. |
| S5 | One pixelwise multiplicative Gaussian perturbation of the corrected field per time and noise level (σ=0, 0.05, 0.10), followed by the common sampling, filtering, SG and knee procedure. |

The smallest filtered zone is the benchmark for equal-size resampling with replacement (B=1000) of larger zones. For zone comparisons, `balanced_correlations.csv` and `zone_local_lee.csv` expose the resampled estimates alongside full-zone estimates. Correlation figures use the equal-size estimates for UA–NA and the observed global coefficient for GA. `within_zone_global_lee.csv` is a separate calculation with within-zone weights, not the zone mean of full-graph local contributions. Cluster composition uses each cluster's total as its denominator (`column_pct`).

SG uses window length 7 and polynomial order 2 in the supplied configuration. Every time, zone, metric and perturbation uses the same configured settings. The selected knee is the interior positive maximum of normalized response minus normalized scale. No knee is reported for a flat or straight response. No scenario-specific ranks, curve-shape anchoring, or published-result offsets are applied in the manuscript workflow.

The published 210–230 m global range and the ±40 m noise-test displacements are reference results, not enforced outputs. They must be recomputed from the matching original inputs. The station subsets, raster grids, trained weights, and support convention affect reproduction; passing synthetic method tests alone does not verify the published numerical results.

## Running the workflow

Run the following from this directory after preparing your configuration. Replace `00-Support/config/example.json` with your own configuration as appropriate.

1. Extract station-centered, area-weighted native LST and retrieve the six covariates.

   ```powershell
   & $py 00-Support/run.py extract-native --config 00-Support/config/example.json
   & $py "01-Relationship/00-Collect Auxiliary Variables.py" --config 00-Support/config/example.json
   ```

2. Compute raw/partial Spearman correlations, Lee's L and local clusters, neighborhood/kernel sensitivity, and alternative statistics.

   ```powershell
   & $py 00-Support/run.py correlation --config 00-Support/config/example.json
   & $py 00-Support/run.py spatial --config 00-Support/config/example.json
   & $py 00-Support/run.py weight-sensitivity --config 00-Support/config/example.json
   & $py 00-Support/run.py compare-statistics --config 00-Support/config/example.json
   ```

3. Prepare the complete 10 m embedding raster and the four GEE mean-embedding exports, train the models, infer uncorrected LST, and apply complete coarse-cell residuals. The Earth Engine export script prepares tasks without starting them. Configure your project and pass `--submit` only when you intend to submit exports. Downloads may be split into tiles; merge them while preserving all 64 bands. Put the time-specific mean exports at `inputs.embedding_coarse.T1`–`T4`; their CRS, transform, shape and band count must match the respective native target. Configurations omitting `embedding_coarse` use a recorded local GDAL mean-aggregation fallback rather than GEE reduceResolution. Do not claim that fallback as an exact GEE reproduction.

   ```powershell
   & $py "02-Downscale/00-Export Embeddings.py" --config 00-Support/config/example.json
   # After intentionally exporting and downloading the embeddings:
   & $py 00-Support/run.py train --config 00-Support/config/example.json --time all
   & $py 00-Support/run.py predict --config 00-Support/config/example.json --time all
   & $py 00-Support/run.py validate-product --config 00-Support/config/example.json
   ```

   Training and prediction require TensorFlow and substantial raster storage. Native-grid model checks use the same 20% evaluation set that monitors early stopping. Product checks use uncorrected predictions within 1500-unit windows in the configured support CRS, paired with nearest native LST samples on the same fine lattice and then averaged over valid pairs. Neither is independent 10 m thermal validation.

4. Extract 61 nested supports, identify global and UrG-specific knees, and run the pixel-noise stress test.

   ```powershell
   & $py 00-Support/run.py sample --config 00-Support/config/example.json
   & $py 00-Support/run.py scale --config 00-Support/config/example.json
   & $py 00-Support/run.py sensitivity --config 00-Support/config/example.json
   & $py 00-Support/run.py plot --config 00-Support/config/example.json
   ```

   The four global scale curves use global Lee's L. Zone curves use the mean of local contributions from the same full-station spatial graph. Spearman curves provide the S4 sensitivity comparison. The S5 stress test uses the same corrected baseline, configured aggregation windows, SG settings, and knee rule for 0%, 5%, and 10% pixelwise multiplicative Gaussian noise. Noise is applied once to source storage pixels, then every scenario is placed on the same station-centered grids. `noise_knees.csv` includes the computed baseline knee and `knee_shift_m`; a positive shift means a coarser knee.

5. Alternatively, run the S5 entry point to compute the same pixel-noise experiment and plot the exported tables. It shares the configuration and implementations with step 4.

   ```powershell
   & $py "03-Scale-Effect/06-Downscaling Error Influence (S5).py" --config 00-Support/config/example.json
   ```

   The historical `(S6)` file requires `--legacy-table`. It applies uniform noise to already aggregated table values, which cannot reproduce pixel-noise attenuation through aggregation. Its tables and figures are diagnostics only. No S6 section exists in the supplied Supplementary Material.

## Optional table and CSV diagnostics

The configuration keys `legacy_multiscale` and `legacy_prediction_csv` accept multiscale GeoPackages and per-station prediction CSVs, respectively. Configure these optional inputs in your configuration file before running the corresponding diagnostics.

```powershell
& $py 00-Support/run.py legacy-scale --config 00-Support/config/example.json
& $py 00-Support/run.py sample-csv --config 00-Support/config/example.json
```

`legacy-scale` computes scale statistics from the supplied multiscale tables. `sample-csv` samples per-station CSVs using station IDs. Their output scale labels have unverified ground-distance meaning. The main `sample` workflow instead requires fine rasters tagged as residual-corrected and samples them using the explicitly configured support definition; neither diagnostic applies complete-block residual correction.

## Output and reporting

Results are written as CSV, GeoPackage, GeoTIFF, JSON, SVG, and PNG. Commands using the shared run-record mechanism save the effective configuration, package versions, and UTC time under `<output_dir>/runs` (default: `00-Support/results/runs`). Diagnostic overrides such as `--permutations 19 --bootstrap 20` are marked explicitly. The configuration template uses 999 permutations and 1,000 bootstrap resamples. With 999 permutations, the minimum attainable permutation p value is 0.001.
