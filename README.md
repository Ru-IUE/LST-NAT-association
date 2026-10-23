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
  06-Downscaling Error Influence (S6).py
```

The top-level folders follow execution order: `00-Support` holds the shared code and configuration, followed by the three workflow steps `01-Relationship` → `02-Downscale` → `03-Scale-Effect`. Inside each step folder the `NN-` prefix gives the execution order. `00-Collect Auxiliary Variables.py` and `00-Export Embeddings.py` provide the preprocessing entry points. Shared implementations live in `00-Support/lstnat`. Paired `(Lee)` and non-`(Lee)` notebooks are equivalent entry points that write the same unified tables; run either variant. `06-Downscaling Error Influence (S6).py` is a standalone downscaling-error experiment; it reads multiscale GeoPackages directly and does not import `00-Support`.

Notebooks are thin entry points: each one runs the matching `00-Support/run.py` command or reads the tables that command produced, so the notebook and command-line routes yield identical results. The module-by-module map is in [00-Support/README.md](00-Support/README.md).

## Environment

Python 3.10 or newer is required. `00-Support/requirements/` separates the analysis (`base.txt`), TensorFlow (`downscaling.txt`), and Earth Engine (`gee.txt`) groups; they specify minimum dependency versions. TensorFlow is only needed for training and inference, and `earthengine-api` only for the two scripts that query Earth Engine.

Install what you need (`pip install -r 00-Support/requirements/base.txt`, plus `downscaling.txt` or `gee.txt`), then run from the repository root:

```powershell
$py = 'python'    # or a full path such as C:\Envs\Conda\envs\myenv\python.exe
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

Provide prepared hourly/monthly station fields, with station quality control and day alignment completed before analysis. The code reads the configured hour fields directly.

All generated products are restricted to `output_dir` inside `Upload Code`. The configuration template sets this to `00-Support/results`.

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

3. Prepare the complete 10 m embedding raster, train the four models, infer uncorrected LST, and apply complete coarse-cell residuals. The Earth Engine export script prepares tasks without starting them. Configure your project and pass `--submit` only when you intend to submit exports. Downloads may be split into tiles; merge them into one grid while preserving all 64 bands.

   ```powershell
   & $py "02-Downscale/00-Export Embeddings.py" --config 00-Support/config/example.json
   # After intentionally exporting and downloading the embeddings:
   & $py 00-Support/run.py train --config 00-Support/config/example.json --time all
   & $py 00-Support/run.py predict --config 00-Support/config/example.json --time all
   & $py 00-Support/run.py validate-product --config 00-Support/config/example.json
   ```

   Training and prediction require TensorFlow and substantial raster storage. Native-grid model checks use the same 20% evaluation set that monitors early stopping. Product checks use uncorrected predictions. Neither is independent 10 m thermal validation.

4. Extract 61 nested supports, identify global and UrG-specific knees, and run the pixel-noise stress test.

   ```powershell
   & $py 00-Support/run.py sample --config 00-Support/config/example.json
   & $py 00-Support/run.py scale --config 00-Support/config/example.json
   & $py 00-Support/run.py sensitivity --config 00-Support/config/example.json
   & $py 00-Support/run.py plot --config 00-Support/config/example.json
   ```

   The four global scale curves use global Lee's L. Zone curves use the mean of local contributions from the same full-station spatial graph. Spearman curves provide the S4 sensitivity comparison. The S5 stress test uses the same corrected baseline, aggregation scales, SG settings, and knee rule for 0%, 5%, and 10% pixelwise multiplicative Gaussian noise.

5. Optionally run the standalone S6 downscaling-error experiment. Review its input and output path settings before execution. It reads multiscale GeoPackages directly and runs independently of `00-Support/run.py`.

   ```powershell
   & $py "03-Scale-Effect/06-Downscaling Error Influence (S6).py"
   ```

## Optional table and CSV diagnostics

The configuration keys `legacy_multiscale` and `legacy_prediction_csv` accept multiscale GeoPackages and per-station prediction CSVs, respectively. Configure these optional inputs in your configuration file before running the corresponding diagnostics.

```powershell
& $py 00-Support/run.py legacy-scale --config 00-Support/config/example.json
& $py 00-Support/run.py sample-csv --config 00-Support/config/example.json
```

`legacy-scale` computes scale statistics from the supplied multiscale tables. `sample-csv` samples per-station CSVs using station IDs. The main `sample` workflow instead requires fine rasters tagged as residual-corrected; neither diagnostic applies complete-block residual correction.

## Output and reporting

Results are written as CSV, GeoPackage, GeoTIFF, JSON, SVG, and PNG. Commands using the shared run-record mechanism save the effective configuration, package versions, and UTC time under `<output_dir>/runs` (default: `00-Support/results/runs`). Diagnostic overrides such as `--permutations 19 --bootstrap 20` are marked explicitly. The configuration template uses 999 permutations and 1,000 bootstrap resamples. With 999 permutations, the minimum attainable permutation p value is 0.001.
