"""E7A: import scores from results/e7a/e7a_labeling.xlsx back into the CSVs.

Run after labeling/saving the Excel file:
    venv\\Scripts\\python.exe -u src\\experiments\\import_e7a_excel.py
Then analyze:
    venv\\Scripts\\python.exe -u src\\experiments\\analyze_e7a.py
"""

import os
import re
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E7A_DIR = os.path.join(REPO_ROOT, "results", "e7a")
XLSX = os.path.join(E7A_DIR, "e7a_labeling.xlsx")

SHEETS = {
    "Round1": "e7a_labeling_round1.csv",
    "Round2": "e7a_labeling_round2.csv",
}


def clean_score(v):
    s = str(v).strip()
    m = re.fullmatch(r"([0-9])(?:\.0)?", s)
    return m.group(1) if m else ""


def read_sheet(xl_path, sheet):
    """Read a sheet, auto-detecting the header row (instruction banner above it)."""
    raw = pd.read_excel(xl_path, sheet_name=sheet, header=None)
    hdr = None
    for i, v in enumerate(raw.iloc[:, 0].astype(str)):
        if v.strip() == "review_id":
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"no 'review_id' header row found in sheet '{sheet}'")
    return pd.read_excel(xl_path, sheet_name=sheet, header=hdr)


def main():
    if not os.path.exists(XLSX):
        print(f"missing: {os.path.relpath(XLSX, REPO_ROOT)}")
        return 1
    sheets = {name: read_sheet(XLSX, name) for name in SHEETS}
    for name, csv_name in SHEETS.items():
        if name not in sheets:
            print(f"sheet '{name}' not found in xlsx, skipped")
            continue
        xl = sheets[name]
        csv_path = os.path.join(E7A_DIR, csv_name)
        df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
        xl["review_id"] = xl["review_id"].astype(int)
        xl["score"] = xl["score"].apply(clean_score)
        if "comment" in xl.columns:
            xl["comment"] = xl["comment"].fillna("").astype(str).str.strip()
        n_before = int((df["score"].astype(str) != "").sum())
        for _, row in xl.iterrows():
            mask = df["review_id"] == row["review_id"]
            if row["score"] != "":
                df.loc[mask, "score"] = row["score"]
            if "comment" in xl.columns and str(row.get("comment", "")):
                df.loc[mask, "comment"] = row["comment"]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        n_after = int((df["score"].astype(str) != "").sum())
        print(f"{name}: {csv_name}  scored {n_before} -> {n_after} / {len(df)}")
    print("\nNext: venv\\Scripts\\python.exe -u src\\experiments\\analyze_e7a.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
