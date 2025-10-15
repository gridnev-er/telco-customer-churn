from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_safe(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_input(paths: dict, params: dict) -> tuple[pd.DataFrame, Path]:
    # приоритет: после кастинга → после дедупа → raw
    candidates = [
        DATA / "clean" / "telco_cast.parquet",
        DATA / "clean" / "telco.parquet",
        DATA / "clean" / "telco_dedup.csv",
        ROOT / paths["raw_dir"] / paths["raw_filename"],
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p), p
            else:
                rp = params.get("reader_params", {})
                header = 0 if rp.get("header", True) else None
                df = pd.read_csv(
                    p,
                    encoding=rp.get("encoding", "utf-8"),
                    sep=rp.get("delimiter", ","),
                    header=header,
                    na_values=rp.get("na_values", ["", "NA", "N/A"]),
                    dtype=str,
                    keep_default_na=True,
                )
                return df, p
    print("input not found: ни clean, ни raw", file=sys.stderr)
    sys.exit(2)


def regex_hits(columns: list[str], patterns: list[str]) -> dict[str, list[str]]:
    hits = {}
    for pat in patterns or []:
        try:
            r = re.compile(pat)
        except re.error as e:
            # сломанный паттерн — лучше упасть конфигом
            print(f"config error: bad regex '{pat}': {e}", file=sys.stderr)
            sys.exit(2)
        matched = [c for c in columns if r.search(c)]
        if matched:
            hits[pat] = matched
    return hits


def is_string_like_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)


def main():
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    cols_cfg = load_yaml(CFG / "columns.yaml")
    if any(x is None for x in (params, paths, cols_cfg)):
        print(
            "config error: params.yaml/paths.yaml/columns.yaml пусты или не найдены",
            file=sys.stderr,
        )
        sys.exit(2)

    df, in_path = read_input(paths, params)
    present_cols = set(df.columns)

    id_cols = list(cols_cfg.get("id_cols", []))
    label_col = cols_cfg.get("label_col")
    text_cols = list(cols_cfg.get("text_cols", []))
    tech_cols = list(cols_cfg.get("tech_cols", []))
    nf_regex = list(cols_cfg.get("non_feature_cols_from_regex", []))

    # 1) наличие ключевых колонок
    missing_required = [
        c for c in (id_cols + ([label_col] if label_col else [])) if c and c not in present_cols
    ]
    if missing_required:
        print(f"required columns missing in data: {missing_required}", file=sys.stderr)

    # 2) собрать non_feature_cols
    nf = set()
    nf.update([c for c in id_cols if c])
    if label_col:
        nf.add(label_col)
    nf.update([c for c in tech_cols if c])

    regex_map = regex_hits(list(present_cols), nf_regex)
    for cols in regex_map.values():
        nf.update(cols)

    non_feature_cols = sorted(nf)

    # 3) planned_features из columns.yaml (кандидаты)
    num_cols = list(cols_cfg.get("num_cols", []))
    cat_cols = list(cols_cfg.get("cat_cols", []))
    date_cols = list(cols_cfg.get("date_cols", []))
    planned_features = sorted(set(num_cols + cat_cols + date_cols + text_cols))

    # 4) пересечение
    intersect = sorted(set(non_feature_cols).intersection(planned_features))

    # 5) проверки типов text_cols
    type_warnings = []
    if params.get("guards", {}).get("warn_if_text_not_string", True):
        for c in text_cols:
            if c in present_cols and not is_string_like_dtype(df[c]):
                type_warnings.append(
                    f"text column '{c}' has dtype '{df[c].dtype}', expected string-like"
                )

    # 6) лог и поведение
    clean_log_path = REPORTS / "data_clean_log.json"
    log = load_json_safe(clean_log_path)
    log.setdefault("non_feature_guard", {})
    log["non_feature_guard"].update(
        {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "input_path": str(in_path),
            "non_feature_cols": non_feature_cols,
            "non_feature_regex_hits": regex_map,
            "planned_features": planned_features,
            "non_feature_in_features_attempted": intersect,
            "missing_required_columns": missing_required,
            "text_type_warnings": type_warnings,
        }
    )
    save_json(clean_log_path, log)

    # 7) возврат кода
    fail_nf = params.get("guards", {}).get("fail_if_non_feature_in_X", True) and len(intersect) > 0
    if missing_required:
        print(f"ERROR: required columns missing: {missing_required}", file=sys.stderr)
        sys.exit(1)
    if fail_nf:
        print(
            f"ERROR: non-feature columns leaked into planned features: {intersect}", file=sys.stderr
        )
        sys.exit(1)
    if type_warnings:
        print("WARN:", "; ".join(type_warnings), file=sys.stderr)

    print(
        json.dumps(
            {
                "status": "ok",
                "non_feature_cols": non_feature_cols,
                "leaks_detected": intersect,
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
