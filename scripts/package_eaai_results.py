"""Package eaai_final_outputs/ into EAAI_Final_Submission_Technical_Package.zip."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eaai_final_outputs"
ZIP_PATH = ROOT / "EAAI_Final_Submission_Technical_Package.zip"
NOTEBOOK = ROOT / "notebooks" / "EAAI_Final_Wildfire_Pipeline_Verification.ipynb"


def snapshot_data_mtimes() -> dict[str, float]:
    data_dir = ROOT / "data"
    return {
        str(p.relative_to(ROOT)): p.stat().st_mtime
        for p in sorted(data_dir.rglob("*"))
        if p.is_file()
    }


def main() -> None:
    if not OUT.exists():
        raise FileNotFoundError(f"Run the EAAI notebook first: {OUT} missing")

    before = snapshot_data_mtimes()

    pkg = OUT / "package"
    pkg.mkdir(parents=True, exist_ok=True)
    if NOTEBOOK.is_file():
        shutil.copy2(NOTEBOOK, pkg / NOTEBOOK.name)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(ROOT).as_posix())

    after = snapshot_data_mtimes()
    for rel, mt in before.items():
        assert rel in after and abs(after[rel] - mt) < 0.01, f"data/ modified: {rel}"

    n_pdf = len(list((OUT / "figures").glob("*.pdf")))
    n_csv = len(list((OUT / "tables").glob("*.csv")))
    print(f"Zip: {ZIP_PATH}")
    print(f"PDFs: {n_pdf} | CSVs: {n_csv}")
    print("data/ untouched.")


if __name__ == "__main__":
    main()
