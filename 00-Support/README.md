# Shared implementation and local support files

Entry points live in `01-Relationship`, `02-Downscale`, and `03-Scale-Effect`; this folder holds the shared code, configuration templates, tests, and dependency lists.

```text
run.py                    command-line entry point for every workflow step
config/                   example.json (configuration template)
data/README.md            input data contract; configured input files are read or placed here
requirements/             base.txt, downscaling.txt, gee.txt
tests/test_methods.py     method checks on synthetic data
lstnat/
  settings.py             constants, config loading, path containment, run records
  raster.py               EPSG:3857 manuscript windows; optional AEQD; area-weighted extraction and nested scales
  statistics.py           pair filtering, Spearman/partial rank correlation, knee detection
  spatial.py              adaptive KNN weights, Lee/Moran/Geary statistics
  association.py          correlation, Lee's L, cluster, and statistic-comparison workflows
  downscaling.py          DNN training, inference, complete-block residual correction
  scaling.py              scale-response curves and knee summaries
  sensitivity.py          S5 pixel-noise experiment
  plotting.py             figures generated from exported tables
```

Run from the `Upload Code` root:

```powershell
$py = 'C:\Users\15093\.conda\envs\Processing\python.exe'
& $py 00-Support/run.py check --config 00-Support/config/example.json
& $py -m unittest discover -s 00-Support/tests -v
```

`analysis.spatial_support` defaults to `station_epsg3857_v1` to match Supplement S1/S3. The existing `station_aeqd_wgs84_v1` support remains an explicit alternative and its outputs cannot be mixed with manuscript-window outputs. S5's named entry point calls the same pixelwise Gaussian-noise implementation as `run.py sensitivity`; the historical S6 file requires explicit legacy-table mode.

Relative input paths resolve from the `Upload Code` root. Configure the required files using `config/example.json` and the input contract in `data/README.md`. Generated outputs stay under the configured `output_dir`, which defaults to `00-Support/results`.
