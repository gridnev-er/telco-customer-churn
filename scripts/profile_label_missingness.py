from __future__ import annotations

import json
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


def read_input(paths: dict, params: dict) -> tuple[pd.DataFrame, Path]:
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
                rp = params["reader_params"]
                header = 0 if rp.get("header", True) else None
                df = pd.read_csv(
                    p,
                    encoding=rp.get("encoding", "utf-8"),
                    sep=rp.get("delimiter", ","),
                    header=header,
                    na_values=rp.get("na_values", ["", "NA", "N/A"]),
                    dtype=str,  # не мутируем сырые типы
                    keep_default_na=True,
                )
                return df, p
    print("input not found: ни clean, ни raw", file=sys.stderr)
    sys.exit(2)


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


def compute_label_stats(df: pd.DataFrame, cfg: dict) -> tuple[dict, int]:
    col = cfg.get("column", "Churn")
    allowed = list(cfg.get("allowed_values", ["Yes", "No"]))
    pos = cfg.get("positive_class", allowed[0] if allowed else "Yes")
    sample_lim = int(cfg.get("sample_invalid_limit", 5))
    if col not in df.columns:
        return {"column": col, "error": "label column not found"}, 1

    s = df[col]
    null_mask = s.isna() | (s.astype(str) == "")
    null_count = int(null_mask.sum())

    # валидные = строго из списка allowed
    valid_mask = s.isin(allowed)
    invalid_mask = ~valid_mask & ~null_mask
    invalid_vals = s[invalid_mask].astype(str).head(sample_lim).tolist()
    invalid_count = int(invalid_mask.sum())

    counts = {v: int((s == v).sum()) for v in allowed}
    total_valid = sum(counts.values())
    positive_share = (counts.get(pos, 0) / total_valid) if total_valid else 0.0

    label_section = {
        "column": col,
        "allowed_values": allowed,
        "positive_class": pos,
        "counts": counts,
        "positive_share": round(positive_share, 6),
        "null_count": null_count,
        "invalid_count": invalid_count,
        "invalid_samples": invalid_vals,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }
    status = 1 if (cfg.get("fail_on_invalid", True) and invalid_count > 0) else 0
    return label_section, status


def compute_missingness(df: pd.DataFrame, max_notice: float) -> dict:
    # для строк/категорий учитываем пустую строку как пропуск
    df2 = df.copy()
    for c in df2.columns:
        # не трогаем типы, только считаем пустые строки как пропуски для статистики
        # если объект/строка — считаем "" как пропуск
        if (
            pd.api.types.is_object_dtype(df2[c])
            or pd.api.types.is_string_dtype(df2[c])
            or str(df2[c].dtype) == "category"
        ):
            empties = df2[c].astype(str) == ""
            if empties.any():
                df2.loc[empties, c] = pd.NA

    miss_map = {}
    n = len(df2)
    for c in df2.columns:
        mc = int(df2[c].isna().sum())
        miss_map[c] = {
            "missing_count": mc,
            "missing_share": round(mc / n if n else 0.0, 6),
        }
    fully_missing = [c for c, v in miss_map.items() if v["missing_share"] == 1.0]
    high_missing = [c for c, v in miss_map.items() if v["missing_share"] >= max_notice]

    return {
        "missing_by_column": miss_map,
        "fully_missing_cols": fully_missing,
        "high_missing_cols": high_missing,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    if params is None or paths is None:
        print("config error: params.yaml/paths.yaml пусты или не найдены", file=sys.stderr)
        sys.exit(2)

    df, in_path = read_input(paths, params)

    # label
    label_cfg = params.get(
        "label_validation",
        {
            "column": "Churn",
            "allowed_values": ["Yes", "No"],
            "positive_class": "Yes",
            "fail_on_invalid": True,
            "sample_invalid_limit": 5,
        },
    )
    label_section, label_status = compute_label_stats(df, label_cfg)

    # missingness
    soft = params.get("soft_checks", {}) or {}
    max_notice = float(soft.get("max_missing_rate_notice", 0.4))
    missing_section = compute_missingness(df, max_notice)

    # merge into data_profile.json
    profile_path = REPORTS / "data_profile.json"
    profile = load_json_safe(profile_path)
    profile["label"] = label_section
    profile["missingness"] = missing_section
    save_json(profile_path, profile)

    # краткий summary в общий лог (необязательный)
    clean_log_path = REPORTS / "data_clean_log.json"
    log = load_json_safe(clean_log_path)
    log.setdefault("summary", {})
    log["summary"]["label_positive_share"] = label_section.get("positive_share", 0.0)
    # top-5 по missing_share
    miss_items = list(missing_section["missing_by_column"].items())
    top5 = sorted(miss_items, key=lambda kv: kv[1]["missing_share"], reverse=True)[:5]
    log["summary"]["missing_top5"] = [{"column": k, **v} for k, v in top5]
    save_json(clean_log_path, log)

    print(
        json.dumps(
            {
                "status": "ok" if label_status == 0 else "invalid_label_values",
                "input": str(in_path),
                "profile_written": str(profile_path),
            },
            ensure_ascii=False,
        )
    )
    sys.exit(label_status)


if __name__ == "__main__":
    main()
