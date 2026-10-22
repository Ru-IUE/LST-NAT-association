"""Original-named entry point for the corrected 61-scale sampling workflow."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "00-Support" / "lstnat").is_dir())
sys.path.insert(0, str(ROOT / "00-Support"))

if __name__ == "__main__":
    from run import main
    sys.argv.insert(1, "sample")
    main()
