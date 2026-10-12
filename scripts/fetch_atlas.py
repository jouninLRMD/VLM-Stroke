"""
Downloads the arterial territory atlas of Liu et al. (2023) in MNI space, as distributed with the SOOP demonstration
repository (github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo, BSD 2-Clause license), and checks its SHA-256.

Writes data/resources/ArterialAtlas136.nii.gz and data/resources/ArterialAtlas136.txt.
Usage (from the repository root):  python scripts/fetch_atlas.py
"""
import hashlib, os, sys, urllib.request
from pathlib import Path

COMMIT = "48486dd63d014c6937c04331da50080185673119"
BASE = f"https://raw.githubusercontent.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo/{COMMIT}/NIfTI/"
FILES = {
    "ArterialAtlas136.nii.gz": "eede82c58229c08b7576e514be259ccd41b597cec7347e1dbd84037fb706ca1b",
    "ArterialAtlas136.txt": "fbac7328a58fe87dc18a0f04b9fc15f10f472abebdffdc134dcc63ada9338545",
}
REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CALM_DATA", str(REPO / "data"))) / "resources"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        dest = OUT / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
            print(f"{name}: already present")
            continue
        with urllib.request.urlopen(BASE + name, timeout=120) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != sha:
            sys.exit(f"{name}: checksum mismatch, file not saved")
        dest.write_bytes(data)
        print(f"{name}: saved to {dest}")


if __name__ == "__main__":
    main()
