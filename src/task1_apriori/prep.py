# src/task1_apriori/prep.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.utils.text_normalize import load_synonyms, normalize_token

DEFAULT_SYNONYMS = (Path(__file__).resolve().parents[1] / "utils" / "symptom_synonyms.json")

def detect_symptom_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).lower().startswith("symptom")]
    return cols or [c for c in df.columns if "symptom" in str(c).lower()]

def row_to_transaction(row: pd.Series, cols: list[str], synonyms: dict[str, str]) -> list[str]:
    seen = set()
    for c in cols:
        tok = normalize_token(row.get(c, ""), synonyms)
        if tok:
            seen.add(tok)
    return sorted(seen)

def main():
    ap = argparse.ArgumentParser(description="Prepare disease→symptom transactions for Apriori")
    ap.add_argument("--input", type=Path, required=True, help="Raw disease-symptom CSV")
    ap.add_argument("--output", type=Path, required=True, help="Where to write transactions.csv")
    ap.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS,
                    help="JSON mapping of {variant: canonical}; default resolves from src/utils/")
    ap.add_argument("--symptom-cols", type=str, default=None,
                    help="Comma-separated column names if not in Symptom_* form")
    ap.add_argument("--augment-input", type=Path, action="append", default=[],
                    help="Optional extra CSV(s) of the same schema to append")
    ap.add_argument("--min_symptoms_per_tx", type=int, default=1,
                    help="Drop baskets with fewer than this many symptoms (default=1)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dfs = [pd.read_csv(args.input)]
    for extra in args.augment_input:
        dfs.append(pd.read_csv(extra))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} file(s); total rows: {len(df)}")

    synonyms = load_synonyms(args.synonyms)

    cols = [c.strip() for c in args.symptom_cols.split(",")] if args.symptom_cols else detect_symptom_cols(df)
    if not cols:
        raise SystemExit("Could not detect symptom columns. Pass --symptom-cols explicitly.")
    print(f"Using {len(cols)} symptom columns: {cols[:5]}{' ...' if len(cols)>5 else ''}")

    did_col = next((c for c in df.columns if str(c).lower() in {"disease", "disease_id", "label"}), None)

    rows = []
    for i, row in df.iterrows():
        did = row[did_col] if did_col else i
        tx = row_to_transaction(row, cols, synonyms)
        if len(tx) >= args.min_symptoms_per_tx:
            rows.append({"disease_id": did, "symptoms": ";".join(tx)})

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} transactions to {args.output} using {len(cols)} symptom columns.")

if __name__ == "__main__":
    main()
