#!/usr/bin/env python3
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_project_root() -> Path:
    """Находим корень репо как ближайший предок, где есть папка configs/."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "configs").exists():
            return parent
    # запасной вариант — текущая рабочая директория
    return Path(".").resolve()


def read_raw(root: Path, paths, params):
    raw_path = root / paths["raw_dir"] / paths["raw_filename"]
    rp = params["reader_params"]
    header = 0 if rp.get("header", True) else None
    df = pd.read_csv(
        raw_path,
        encoding=rp.get("encoding", "utf-8"),
        sep=rp.get("delimiter", ","),
        header=header,
        na_values=rp.get("na_values", ["", "NA", "N/A"]),
        dtype=str,  # читаем как строки — ничего не «чиним»
        keep_default_na=True,
    )
    return df, raw_path


def try_cast_series(s: pd.Series, expected: str):
    non_null = s.dropna()
    n = len(non_null)
    if n == 0:
        return 0.0, {"checked": 0, "fails": 0}
    fails = 0
    if expected == "int":
        for v in non_null:
            if not re.fullmatch(r"[+-]?\d+", str(v).strip()):
                fails += 1
    elif expected == "float":
        for v in non_null:
            if not re.fullmatch(r"[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?", str(v).strip()):
                fails += 1
    elif expected == "datetime":
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
        fails = int(parsed.isna().sum())
    elif expected in ("string", "category"):
        fails = 0
    else:
        return 0.0, {"checked": n, "fails": 0}
    return (fails / max(n, 1)), {"checked": int(n), "fails": int(fails)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/schema_check.json", help="куда писать отчёт JSON")
    ap.add_argument("--verbose", action="store_true", help="печать путей и статуса")
    args = ap.parse_args()

    root = find_project_root()
    cfg = root / "configs"
    reports_path = (root / args.out) if not args.out.startswith("/") else Path(args.out)
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    schema = load_yaml(cfg / "schema.yaml")
    params = load_yaml(cfg / "params.yaml")
    paths = load_yaml(cfg / "paths.yaml")

    df, raw_path = read_raw(root, paths, params)
    existing_cols = list(df.columns)

    expected_cols = [c["name"] for c in schema["columns"]]
    missing = [c for c in expected_cols if c not in existing_cols]
    unexpected = [c for c in existing_cols if c not in expected_cols]

    if args.verbose:
        print(f"[validate] root={root}")
        print(f"[validate] raw_path={raw_path} exists={raw_path.exists()}")
        print(f"[validate] out={reports_path}")

    # жёсткий фэйл на критичных
    hard_missing = [c for c in ("customerID", "Churn") if c in missing]
    if hard_missing:
        report = {
            "status": "fail",
            "reason": f"Missing critical columns: {hard_missing}",
            "missing": missing,
            "unexpected": unexpected,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(raw_path),
        }
        reports_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.verbose:
            print(f"[validate] wrote FAIL report → {reports_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        sys.exit(1)

    # мягкие проверки
    soft = params.get("soft_checks", {})
    max_fail = float(soft.get("max_cast_fail_rate", 0.02))
    dtype_mismatch, warnings = [], []
    exp_map = {c["name"]: c["type"] for c in schema["columns"]}
    notes_max_len = int(soft.get("notes_max_len", 4000))

    for col in expected_cols:
        if col not in df.columns:
            continue
        expected = exp_map.get(col, "string")
        fail_rate, details = try_cast_series(df[col], expected)
        if fail_rate > 0:
            entry = {
                "column": col,
                "expected": expected,
                "cast_fail_rate": round(float(fail_rate), 6),
                "fails": details["fails"],
                "checked": details["checked"],
            }
            if fail_rate > max_fail and expected in ("int", "float", "datetime"):
                dtype_mismatch.append(entry)
            else:
                warnings.append({"type_warning": entry})
        if col == "notes":
            too_long = df[col].dropna().astype(str).map(len) > notes_max_len
            cnt = int(too_long.sum())
            if cnt > 0:
                warnings.append({"notes_max_len_exceeded": {"count": cnt, "limit": notes_max_len}})

    report = {
        "status": "ok" if not dtype_mismatch and not missing else "warn",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
        "missing": missing,
        "unexpected": unexpected,
        "dtype_mismatch": dtype_mismatch,
        "warnings": warnings,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
    }
    reports_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.verbose:
        print(f"[validate] wrote report → {reports_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
