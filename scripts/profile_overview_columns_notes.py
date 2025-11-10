from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    raise FileNotFoundError("input not found: ни clean, ни raw")


def dtype_bucket(s: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if str(s.dtype) == "category":
        return "category"
    return "string"


def numeric_profile(s: pd.Series, qlist: List[float]) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce")
    n = len(x)
    valid = x.dropna()
    out = {
        "type": "float" if pd.api.types.is_float_dtype(x) else "int",
        "count": n,
        "n_unique": int(valid.nunique()),
        "missing_share": round(1 - len(valid) / n if n else 0.0, 6),
    }
    if len(valid):
        qs = valid.quantile(qlist)
        out.update(
            {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            }
        )
        # именованные квантили p5, p25, p50...
        for q, v in zip(qlist, qs.values):
            name = f"p{int(q*100):d}"
            out[name] = float(v)
    return out


def categorical_profile(s: pd.Series, top_k: int, rare_thr: float, n_rows: int) -> Dict[str, Any]:
    # считаем "" как пропуск только для статистики профиля
    s_work = s.copy()
    if pd.api.types.is_object_dtype(s_work) or pd.api.types.is_string_dtype(s_work):
        s_work = s_work.mask(s_work.astype(str) == "", other=pd.NA)
    vc = s_work.value_counts(normalize=True, dropna=True)
    top = [{"level": str(idx), "share": round(float(fr), 6)} for idx, fr in vc.head(top_k).items()]
    rare_share = float(vc[vc < rare_thr].sum()) if len(vc) else 0.0
    return {
        "type": "category" if str(s.dtype) == "category" else "string",
        "count": n_rows,
        "n_unique": int(s_work.nunique(dropna=True)),
        "top_k": top,
        "rare_share": round(rare_share, 6),
        "missing_share": round(float(s_work.isna().mean()), 6),
    }


def datetime_profile(s: pd.Series) -> Dict[str, Any]:
    x = pd.to_datetime(s, errors="coerce", utc=True)
    n = len(x)
    valid = x.dropna()
    out = {
        "type": "datetime",
        "count": n,
        "missing_share": round(1 - len(valid) / n if n else 0.0, 6),
    }
    if len(valid):
        out.update(
            {
                "min": valid.min().isoformat(),
                "max": valid.max().isoformat(),
            }
        )
    return out


def notes_profile(s: pd.Series, qlist: List[float], np_cfg: dict) -> Dict[str, Any]:
    # строковое представление без изменения смысла
    s_str = s.fillna("").astype(str)
    empty_mask = s_str == ""
    n = len(s_str)

    # длины
    len_chars = s_str.str.len()
    len_words = s_str.str.split().map(len)

    # квантили
    def qdict(series: pd.Series) -> Dict[str, Any]:
        out = {
            "min": int(series.min()) if len(series) else 0,
            "max": int(series.max()) if len(series) else 0,
        }
        if len(series):
            qs = series.quantile(qlist)
            for q, v in zip(qlist, qs.values):
                out[f"p{int(q*100)}"] = int(v)
        return out

    # PII эвристики
    patterns = (np_cfg or {}).get("pii_patterns", {})
    compiled = {}
    for k, pat in patterns.items():
        try:
            compiled[k] = re.compile(pat)
        except re.error:
            continue  # пропускаем битые паттерны

    def has_pat(pat: re.Pattern) -> pd.Series:
        return s_str.str.contains(pat, na=False)

    pii_counters = {}
    for name, rp in compiled.items():
        hit = has_pat(rp)
        pii_counters[f"has_{name}_share"] = round(float(hit.mean()), 6)

    warnings = []
    max_len_warn = (np_cfg or {}).get("max_len_chars_warn", None)
    if max_len_warn is not None:
        over = (len_chars > int(max_len_warn)).mean() if n else 0.0
        if over > 0:
            warnings.append(
                f"notes length > {int(max_len_warn)} chars in {round(float(over), 6)} share of rows"
            )

    return {
        "empty_share": round(float(empty_mask.mean()), 6),
        "non_empty_share": round(1 - float(empty_mask.mean()), 6),
        "length_chars": qdict(len_chars),
        "length_words": qdict(len_words),
        "pii_counters": pii_counters,
        "warnings": warnings,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    if params is None or paths is None:
        raise SystemExit("config error: params.yaml/paths.yaml отсутствуют или пусты")

    df, in_path = read_input(paths, params)
    n_rows, n_cols = len(df), len(df.columns)

    prof_cfg = params.get("profiling", {}) or {}
    qlist = [float(q) for q in prof_cfg.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])]
    top_k = int(prof_cfg.get("top_k_categories", 10))
    rare_thr = float(prof_cfg.get("rare_level_threshold", 0.01))

    # overview
    dtype_map = {"int": 0, "float": 0, "bool": 0, "category": 0, "string": 0, "datetime": 0}
    for c in df.columns:
        dtype_map[dtype_bucket(df[c])] += 1
    overview = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "dtypes": dtype_map,
        "input_path": str(in_path),
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }

    # columns
    columns: Dict[str, Any] = {}
    for c in df.columns:
        bucket = dtype_bucket(df[c])
        if bucket in ("int", "float"):
            columns[c] = numeric_profile(df[c], qlist)
        elif bucket in ("string", "category"):
            columns[c] = categorical_profile(df[c], top_k, rare_thr, n_rows)
        elif bucket == "datetime":
            columns[c] = datetime_profile(df[c])
        else:  # bool и прочее
            # для bool: частоты True/False, missing_share
            s = df[c]
            true_share = float((s == True).mean())  # noqa: E712
            missing_share = float(s.isna().mean())
            columns[c] = {
                "type": "bool",
                "count": n_rows,
                "true_share": round(true_share, 6),
                "missing_share": round(missing_share, 6),
            }

    # notes_profile (если есть колонка 'notes')
    np_cfg = params.get("notes_profiling", {}) or {}
    if "notes" in df.columns:
        np_section = notes_profile(df["notes"], qlist, np_cfg)
    else:
        np_section = {
            "warning": "notes column not found",
            "profiled_at": datetime.now(timezone.utc).isoformat(),
        }

    # merge в reports/data_profile.json (не перезатирая другие разделы)
    profile_path = REPORTS / "data_profile.json"
    profile = load_json_safe(profile_path)
    profile["overview"] = overview
    profile["columns"] = columns
    profile["notes_profile"] = np_section
    save_json(profile_path, profile)

    # короткий вывод
    print(
        json.dumps(
            {"status": "ok", "written": str(profile_path), "rows": n_rows, "cols": n_cols},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
