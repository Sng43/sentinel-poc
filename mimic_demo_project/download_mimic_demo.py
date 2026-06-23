"""Download the open-access MIMIC-IV Clinical Database Demo (v2.2).

The demo is de-identified and open-access — no PhysioNet credentialing and no
AWS/boto3 required. It is fetched as a single zip over HTTPS and extracted,
preserving the ``hosp/`` and ``icu/`` table layout that ``src.data_loader``
expects.

Usage
-----
    python download_mimic_demo.py [dest_dir]

``dest_dir`` defaults to ``./mimic_iv_demo``. From the project, the notebooks
point the loader at ``project_sentinel/data/raw/mimic_iv_demo``.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

# Open-access demo zip (redirects to the signed get-zip endpoint).
DEMO_URL = "https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"


def download_mimic_demo(dest_dir: str = "mimic_iv_demo") -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Skip the download if the tables are already extracted.
    if any(dest.rglob("icustays.csv.gz")):
        print(f"✓ MIMIC-IV demo already present in {dest} — nothing to do.")
        return dest

    zip_path = dest / "mimic-iv-clinical-database-demo-2.2.zip"
    print(f"Downloading MIMIC-IV demo (~16 MB) → {zip_path} ...")
    urllib.request.urlretrieve(DEMO_URL, zip_path)  # noqa: S310 (trusted host)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()  # tidy up

    n = sum(1 for _ in dest.rglob("*.csv.gz"))
    print(f"✓ Done. {n} tables extracted under {dest}")
    return dest


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "mimic_iv_demo"
    download_mimic_demo(target)
