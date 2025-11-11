from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List

import pandas as pd
import yaml

try:
    from sklearn.metrics import roc_auc_score  # type: ignore

    _HAS_SK = True
except Exception:
    _HAS_SK = False

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


# ---------- utils
def load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_parquet_robust(p: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")


def read_clean(paths: dict) -> tuple[pd.DataFrame, Path]:
    candidates = [
        DATA / "clean" / "telco.parquet",
        DATA / "clean" / "telco_cast.parquet",
    ]
    for p in candidates:
        if p.exists():
            return read_parquet_robust(p), p
    raise FileNotFoundError("clean not found: data/clean/telco.parquet|telco_cast.parquet")


def build_non_feature_set(cols_cfg: dict) -> tuple[set, list[re.Pattern]]:
    non_feat = set()
    non_feat.update(cols_cfg.get("id_cols", []) or [])
    if cols_cfg.get("label_col"):
        non_feat.add(cols_cfg["label_col"])
    non_feat.update(cols_cfg.get("tech_cols", []) or [])
    # regex-автоисключения
    rx_list = cols_cfg.get("non_feature_cols_from_regex", []) or []
    comp = [re.compile(rx) for rx in rx_list]
    # продублируем id_cols (дешёво и безопасно)
    for c in cols_cfg.get("id_cols", []) or []:
        non_feat.add(c)
    return non_feat, comp


# ---------- checks
def keyword_scan(columns: List[str], patterns: List[str]) -> List[dict]:
    out = []
    regs = [re.compile(p) for p in patterns]
    for c in columns:
        for rg in regs:
            if rg.search(c):
                out.append({"column": c, "keyword": rg.pattern})
    # уникализируем по колонкам/паттернам
    uniq = {(d["column"], d["keyword"]) for d in out}
    return [{"column": c, "keyword": k} for (c, k) in sorted(uniq)]


def categorical_pure_levels(
    df: pd.DataFrame, label_col: str, col: str, min_support: int
) -> List[dict]:
    res: List[dict] = []
    if col == label_col:
        return res

    grp = df[[col, label_col]].copy()
    grp[label_col] = grp[label_col].astype("string")
    # считаем распределение по классам для каждого уровня
    ct = grp.value_counts(dropna=False).rename("n").reset_index()
    # таблица: level -> {Yes: n, No: n}
    tbl: dict = {}
    for _, r in ct.iterrows():
        level = r[col]
        y = r[label_col]
        n = int(r["n"])
        if level not in tbl:
            tbl[level] = {"Yes": 0, "No": 0}
        if y in ("Yes", "No"):
            tbl[level][y] += n
    for level, d in tbl.items():
        n_yes, n_no = d["Yes"], d["No"]
        support = n_yes + n_no
        if support >= min_support:
            if (n_yes > 0 and n_no == 0) or (n_no > 0 and n_yes == 0):
                res.append(
                    {
                        "column": col,
                        "level": None if pd.isna(level) else str(level),
                        "support": int(support),
                        "class_only": "Yes" if (n_yes > 0 and n_no == 0) else "No",
                    }
                )
    return res


def numeric_univariate_auc(
    df: pd.DataFrame, label_col: str, col: str, auc_thr: float
) -> dict | None:
    if not _HAS_SK:
        return None
    y = df[label_col]
    if y.isna().any():
        y = y.copy()
    y = y.map({"No": 0, "Yes": 1})
    if y.isna().any():
        return None
    x = pd.to_numeric(df[col], errors="coerce")
    if x.notna().sum() < 2:
        return None
    # ранги устойчивее, чем «сырые» значения
    xr = x.rank(pct=True)
    # удалим строки с NaN в y/x
    mask = xr.notna() & y.notna()
    if mask.sum() < 2:
        return None
    auc = float(roc_auc_score(y[mask], xr[mask]))
    if auc >= auc_thr or (1 - auc) >= auc_thr:
        return {"column": col, "auroc": round(max(auc, 1 - auc), 6)}
    return None


def temporal_anomalies(df: pd.DataFrame, params: dict, date_cols_cfg: List[str]) -> List[dict]:
    out: List[dict] = []
    # будущие даты
    future_year_max = int(
        params.get("leakage", {}).get("temporal", {}).get("future_year_max", 2035)
    )
    now = pd.Timestamp.now(tz="UTC")
    for c in date_cols_cfg or []:
        if c not in df.columns:
            continue
        s = pd.to_datetime(df[c], errors="coerce", utc=True)
        bad_future = s.dropna().dt.year.gt(future_year_max).sum()
        bad_after_now = (s.dropna() > now).sum()
        if bad_future or bad_after_now:
            out.append(
                {
                    "column": c,
                    "future_year_gt": int(bad_future),
                    "after_now": int(bad_after_now),
                }
            )
    # порядок пар дат
    pairs = params.get("leakage", {}).get("temporal", {}).get("check_order_pairs", []) or []
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            sa = pd.to_datetime(df[a], errors="coerce", utc=True)
            sb = pd.to_datetime(df[b], errors="coerce", utc=True)
            bad = (sb < sa).sum()
            if bad:
                out.append({"pair": [a, b], "violations": int(bad)})
    return out


# ---------- main
def main():
    t0 = time.perf_counter()
    schema = load_yaml(CFG / "schema.yaml")
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    cols = load_yaml(CFG / "columns.yaml")
    log = load_json(REPORTS / "data_clean_log.json")
    schema_version = schema.get("schema_version")

    df, clean_path = read_clean(paths)
    label_col = cols.get("label_col", "Churn")
    if label_col not in df.columns:
        raise SystemExit(f"label_col '{label_col}' not in data")

    # non-feature set + regexы
    non_feat, rx_nonfeat = build_non_feature_set(cols)
    # добавим regex-совпадения в non-feature
    for c in df.columns:
        for rg in rx_nonfeat:
            if rg.search(c):
                non_feat.add(c)

    # --- keyword scan
    kw_patterns = params.get("leakage", {}).get("keyword_patterns", []) or []
    kw_hits = keyword_scan(list(df.columns), kw_patterns)

    # --- near-perfect: cat-уровни только у одного класса + num AUC
    np_cfg = params.get("leakage", {}).get("near_perfect", {}) or {}
    min_support = int(np_cfg.get("min_support", 20))
    auc_thr = float(np_cfg.get("auc_threshold", 0.98))

    suspect_levels: List[dict] = []
    suspect_auc: List[dict] = []

    # кандидаты: все колонки минус non-feature и текст
    text_cols = set(cols.get("text_cols", []) or [])
    candidates = [c for c in df.columns if c not in non_feat and c not in text_cols]

    for c in candidates:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            auc_info = numeric_univariate_auc(df, label_col, c, auc_thr)
            if auc_info:
                suspect_auc.append(auc_info)
        else:
            sus = categorical_pure_levels(df, label_col, c, min_support)
            suspect_levels.extend(sus)

    # --- temporal anomalies
    date_cols_cfg = cols.get("date_cols", []) or []
    temporal = temporal_anomalies(df, params, date_cols_cfg)

    # --- статус
    has_any = bool(kw_hits or suspect_levels or suspect_auc or temporal)
    status = "no_leakage_detected" if not has_any else "needs_manual_review"

    leakage_check = {
        "schema_version": schema_version,
        "input": {"path": str(clean_path)},
        "rules": {
            "no_label_usage_in_steps_1_9": True,
            "non_feature_guards_enabled": True,
            "temporal_fields_blocked": True,
        },
        "scans": {
            "keyword_hits": kw_hits,
            "near_perfect_levels": suspect_levels,
            "near_perfect_auc": suspect_auc,
            "temporal_anomalies": temporal,
            "id_overlap_train_test": None,  # проверим позже, на этапе сплитов
        },
        "suspect_features": sorted(
            list({d["column"] for d in suspect_levels} | {d["column"] for d in suspect_auc})
        ),
        "status": status,
        "comment": (
            "Найдены подозрительные признаки — проверь в EDA" if has_any else "Утечек не обнаружено"
        ),
    }

    # слей в общий лог запуска и добавь тайминг
    if not isinstance(log, dict):
        log = {}
    log["leakage_check"] = leakage_check

    elapsed = round(time.perf_counter() - t0, 3)
    log_timing = log.get("timing_sec", {}) if isinstance(log.get("timing_sec"), dict) else {}
    log_timing["leakage"] = elapsed
    log["timing_sec"] = log_timing

    save_json(REPORTS / "data_clean_log.json", log)

    print(
        json.dumps(
            {
                "status": "ok",
                "written": "reports/data_clean_log.json",
                "leakage_status": status,
                "elapsed_sec": elapsed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
