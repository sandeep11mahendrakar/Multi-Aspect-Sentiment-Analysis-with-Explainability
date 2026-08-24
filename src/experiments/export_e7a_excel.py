"""E7A: export the labeling sample to a single Excel file for manual labeling.

Creates results/e7a/e7a_labeling.xlsx with two sheets:
    Round1  200 reviews  (main sample)
    Round2   50 reviews  (kappa subset)

Sandeep fills the `score` column (0-9) in Excel and saves the file, then run:
    venv\\Scripts\\python.exe -u src\\experiments\\import_e7a_excel.py
which copies the scores back into the CSVs, then:
    venv\\Scripts\\python.exe -u src\\experiments\\analyze_e7a.py

Score scale: 0-3 negative | 4-6 neutral | 7-9 positive (text sentiment only).
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E7A_DIR = os.path.join(REPO_ROOT, "results", "e7a")
XLSX = os.path.join(E7A_DIR, "e7a_labeling.xlsx")

INSTR = ("Fill ONLY the score column: 0-3 negative | 4-6 neutral | 7-9 positive "
         "(based on text sentiment, ignore star ratings). Save the file when done.")


def main():
    r1 = pd.read_csv(os.path.join(E7A_DIR, "e7a_labeling_round1.csv"),
                     encoding="utf-8-sig", keep_default_na=False)
    r2 = pd.read_csv(os.path.join(E7A_DIR, "e7a_labeling_round2.csv"),
                     encoding="utf-8-sig", keep_default_na=False)
    # never overwrite existing scores in the xlsx
    if os.path.exists(XLSX):
        try:
            old = pd.read_excel(XLSX, sheet_name=None)
            for name, df in (("Round1", r1), ("Round2", r2)):
                if name in old and "score" in old[name].columns:
                    prev = old[name].set_index("review_id")["score"]
                    df["score"] = df["review_id"].map(prev).fillna("").astype(str).replace("nan", "")
                    if "comment" in old[name].columns:
                        com = old[name].set_index("review_id")["comment"]
                        df["comment"] = df["review_id"].map(com).fillna("").astype(str).replace("nan", "")
        except Exception as e:
            print(f"warning: could not read existing xlsx ({e}); starting fresh")

    cols = ["review_id", "Review Text", "score", "comment"]
    with pd.ExcelWriter(XLSX, engine="openpyxl") as xl:
        for name, df in (("Round1", r1[cols]), ("Round2", r2[cols])):
            df.to_excel(xl, sheet_name=name, index=False)
            ws = xl.sheets[name]
            ws.insert_rows(1)
            ws.cell(row=1, column=1, value=INSTR)
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(cols))
            ws.freeze_panes = "A3"
            ws.column_dimensions["A"].width = 10
            ws.column_dimensions["B"].width = 110
            ws.column_dimensions["C"].width = 8
            ws.column_dimensions["D"].width = 30
            for row in ws.iter_rows(min_row=3, min_col=2, max_col=2):
                for cell in row:
                    cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")
            for row in ws.iter_rows(min_row=3, min_col=1, max_col=4):
                for cell in row:
                    cell.alignment = cell.alignment.copy(vertical="top")

    print(f"scored so far: R1 {int((r1['score'].astype(str) != '').sum())}/200 | "
          f"R2 {int((r2['score'].astype(str) != '').sum())}/50")
    print(f"-> {os.path.relpath(XLSX, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
