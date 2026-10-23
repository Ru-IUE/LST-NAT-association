"""Supplement S5: pixelwise Gaussian noise before the common 61-scale analysis."""
import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "00-Support" / "lstnat").is_dir())
sys.path.insert(0, str(ROOT / "00-Support"))


def main():
    from lstnat.settings import load
    from lstnat.sensitivity import run
    from lstnat.plotting import run as plot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "00-Support/config/example.json"))
    parser.add_argument("--plot-only", action="store_true", help="Plot already computed shared result tables")
    args = parser.parse_args()
    cfg = load(args.config)
    if not args.plot_only:
        run(cfg)
    plot(cfg)


if __name__ == "__main__":
    main()
