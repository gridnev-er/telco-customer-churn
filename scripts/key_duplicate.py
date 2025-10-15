from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"


def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raw(paths, params) -> pd.DataFrame:
    raw_path = ROOT / paths["raw_dir"] / paths["raw_filename"]
    rp = params["reader_params"]
    header = 0 if rp.get("header", True) else None
    df = pd.read_csv(
        raw_path,
        encoding=rp.get("encoding", "utf-8"),
        sep=rp.get("delimiter", ","),
        header=header,
        na_values=rp.get("na_values", ["", "NA", "N/A"]),
        dtype=str,
        keep_default_na=True,
    )
    return df, raw_path


def main():
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    cols_cfg = load_yaml(CFG / "columns.yaml")

    id_cols = cols_cfg.get("id_cols", [])
    if not id_cols:
        print("config error: configs/columns.yaml → id_cols пустой", file=sys.stderr)
        sys.exit(2)
    key = id_cols[0]

    # Флаги строгости
    dedup_flags = params.get("dedup") or {}
    fail_on_missing_key = bool(dedup_flags.get("fail_on_missing_key", True))
    fail_on_key_conflicts = bool(dedup_flags.get("fail_on_key_conflicts", False))

    # Читаем raw
    df, raw_path = read_raw(paths, params)
    if key not in df.columns:
        msg = f"Ключевая колонка '{key}' отсутствует в raw"
        if fail_on_missing_key:
            print(msg, file=sys.stderr)
            sys.exit(1)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)

    # Подсчёт пустых ключей (NaN или ровно пустая строка)
    key_null_count = (
        int(df[key].isna().sum()) + int((df[key] == "").sum()) if key in df.columns else len(df)
    )

    # Точные дубликаты строк (по всем колонкам)
    dup_mask = df.duplicated(keep="first")  # pandas считает NaN==NaN → нам подходит
    duplicates_exact_removed = int(dup_mask.sum())

    # Кол-во групп точных дублей
    vc = df.value_counts(dropna=False)
    duplicates_exact_groups = int((vc > 1).sum())

    # Результат без точных дублей
    df_nodup = df.loc[~dup_mask].copy()

    # Конфликты по ключу (после удаления точных дублей)
    key_conflict_rows = 0
    key_conflict_groups = 0
    key_conflict_samples = []

    if key in df_nodup.columns:
        grp = df_nodup.groupby(key, dropna=False)
        for k, g in grp:
            if len(g) <= 1:
                continue
            # Различия хоть в одной НЕключевой колонке
            diff_cols = []
            for c in g.columns:
                if c == key:
                    continue
                # считаем различия с учётом NaN: dropna=False
                if g[c].nunique(dropna=False) > 1:
                    diff_cols.append(c)
            if diff_cols:
                key_conflict_groups += 1
                key_conflict_rows += len(g)
                # примеров много не копим
                if len(key_conflict_samples) < 10:
                    key_conflict_samples.append(
                        {
                            key: None if (pd.isna(k) if isinstance(k, float) else False) else k,
                            "diff_columns": diff_cols[:10],
                            "row_count": int(len(g)),
                        }
                    )

    # Пишем чистую таблицу
    CLEAN.mkdir(parents=True, exist_ok=True)
    clean_path = CLEAN / "telco.parquet"
    try:
        df_nodup.to_parquet(clean_path, index=False)  # требует pyarrow/fastparquet
    except Exception as e:
        print(f"Parquet не записан ({e}). Временный fallback в CSV.", file=sys.stderr)
        clean_path = CLEAN / "telco_dedup.csv"
        df_nodup.to_csv(clean_path, index=False, encoding="utf-8")

    # Лог метрик
    REPORTS.mkdir(parents=True, exist_ok=True)
    log = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
        "key_column": key,
        "kept_row_index_strategy": "first_occurrence",
        "duplicates_exact_removed": duplicates_exact_removed,
        "duplicates_exact_groups": duplicates_exact_groups,
        "key_conflict_rows": key_conflict_rows,
        "key_conflict_groups": key_conflict_groups,
        "key_conflict_samples": key_conflict_samples,
        "key_null_count": key_null_count,
        "row_hash_method": "pandas.value_counts on full-row (NaN==NaN)",
        "n_rows_raw": int(len(df)),
        "n_rows_clean": int(len(df_nodup)),
        "n_cols": int(len(df.columns)),
        "status": "fail" if (fail_on_key_conflicts and key_conflict_groups > 0) else "ok",
    }
    (REPORTS / "data_clean_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, ensure_ascii=False, indent=2))

    if fail_on_key_conflicts and key_conflict_groups > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
