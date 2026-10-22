"""Report residual increases from the unified, unshifted knee results."""
from pathlib import Path
import argparse
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "00-Support" / "lstnat").is_dir())
sys.path.insert(0, str(ROOT / "00-Support"))


def main():
    import pandas as pd
    from lstnat.settings import load, output_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "00-Support/config/example.json"))
    parser.add_argument("--legacy", action="store_true", help="Read the explicitly labeled legacy diagnostic results")
    args = parser.parse_args()
    cfg = load(args.config)
    folder = "legacy_diagnostic" if args.legacy else "scale"
    path = output_path(cfg, folder, "knees.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Run 02-Scale Stats first; missing knee table: {path}")
    table = pd.read_csv(path, keep_default_na=False, na_values=[""])
    rows = table[table.metric.isin(["lee", "mean_local_lee"])]
    print(rows[["time", "zone", "metric", "scale", "residual_increase_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
