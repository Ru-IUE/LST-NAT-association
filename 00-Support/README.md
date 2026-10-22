# Shared implementation and local support files

Entry points live in `01-Relationship`, `02-Downscale`, and `03-Scale-Effect`; this folder holds the shared code, configuration templates, tests, and dependency lists.

```text
run.py                    command-line entry point for every workflow step
config/                   example.json (public template) and local.json (local diagnostics)
data/README.md            input data contract; configured input files are read or placed here
requirements/             base.txt, downscaling.txt, gee.txt
tests/test_methods.py     method checks on synthetic data
lstnat/
  settings.py             constants, config loading, path containment, run records
  raster.py               station input, area-weighted native extraction, nested-scale sampling
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
& $py 00-Support/run.py check --config 00-Support/config/local.json
& $py -m unittest discover -s 00-Support/tests -v
```

Inputs remain relative to `Upload Code`, so the supplied local configurations continue to read the untouched sibling project folders. Generated outputs stay under `00-Support/results`.
