"""Portable configuration and output containment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMES = {
    "T1": {"name": "Day Terra", "lst": "TD_LST", "nat": "TEM10"},
    "T2": {"name": "Day Aqua", "lst": "AD_LST", "nat": "TEM13"},
    "T3": {"name": "Night Terra", "lst": "TN_LST", "nat": "TEM22"},
    "T4": {"name": "Night Aqua", "lst": "AN_LST", "nat": "TEM01"},
}
SCALES = tuple(range(10, 1211, 20))
ZONES = {1: "UA", 2: "SA", 3: "RA", 4: "NA"}
COVARIATES = ["NDVI", "NDBI", "GAIA", "ALTI", "Slope", "Soil"]


def load(path=None):
    path = Path(path or ROOT / "00-Support" / "config" / "example.json").resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg["_config"] = str(path)
    return cfg


def input_path(cfg, key, time=None):
    value = cfg["inputs"][key]
    if time is not None:
        value = value[time]
    if not value:
        raise ValueError(f"Configure inputs.{key}" + (f".{time}" if time else ""))
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_path(cfg, *parts):
    base = (ROOT / cfg.get("output_dir", "00-Support/results")).resolve()
    path = base.joinpath(*parts).resolve()
    # Inputs may live anywhere; all generated files must stay inside this repository.
    if not base.is_relative_to(ROOT) or base == ROOT or not path.is_relative_to(base):
        raise ValueError("output_dir must be a subdirectory of Upload Code")
    # Generated intermediate products are also inputs to later steps. They are
    # intentionally allowed here; containment protects all external originals.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setting(cfg, name, default):
    return cfg.get("analysis", {}).get(name, default)


def save_run(cfg, command, **details):
    import importlib.metadata
    import platform
    from datetime import datetime, timezone
    versions = {}
    for name in ["numpy", "pandas", "scipy", "geopandas", "rasterio", "esda", "libpysal", "tensorflow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {"command": command, "utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "versions": versions,
              "configuration": cfg, **details}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = output_path(cfg, "runs", f"{stamp}_{command}.json")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
