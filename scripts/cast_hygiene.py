from __future__ import annotations

import json
import sys
import unicodedata
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


def load_previous_log() -> dict:
    p = REPORTS / "data_clean_log.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_log(obj: dict):
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_clean_log.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_input(paths, params) -> tuple[pd.DataFrame, Path]:
    # приоритет — результат дедупа
    cand = [
        DATA / "clean" / "telco.parquet",
        DATA / "clean" / "telco_dedup.csv",
        ROOT / paths["raw_dir"] / paths["raw_filename"],
    ]
    for p in cand:
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
                    dtype=str,
                    keep_default_na=True,
                )
                return df, p
    print("Не найден входной файл (clean/raw)", file=sys.stderr)
    sys.exit(1)


def normalize_text(val: str) -> str:
    if pd.isna(val):
        return val
    s = str(val)
    # NBSP -> space
    s = s.replace("\u00a0", " ")
    # внешний strip
    s2 = s.strip()
    # Unicode NFKC
    s3 = unicodedata.normalize("NFKC", s2)
    return s3


def canonicalize_case(series: pd.Series, mapping: dict) -> tuple[pd.Series, int]:
    """Привести регистр/форму по каноническому маппингу (ключи — в нижнем регистре)."""
    cnt = 0

    def _map(v):
        nonlocal cnt
        if pd.isna(v):
            return v
        key = str(v).strip().lower()
        if key in mapping:
            newv = mapping[key]
            if newv != v:
                cnt += 1
            return newv
        return v

    return series.map(_map), cnt


def maybe_strip_thousand_sep(col: pd.Series, decimal_sep: str) -> pd.Series:
    """Если формат однородный, убираем тысячные разделители (запятые/пробелы/точки)."""
    s = col.astype(str)
    # быстрые эвристики
    if decimal_sep == ".":
        # если есть запятые и при этом точка правее последней запятой → вероятны тысячные запятые
        mask = s.str.contains(",", regex=False)
        if mask.any():
            s2 = s.str.replace(",", "", regex=False)
            return s2
    elif decimal_sep == ",":
        # европейский формат: 1.234,56
        mask = s.str.contains(r"\.\d{3}(,|$)", regex=True)
        if mask.any():
            s2 = s.str.replace(".", "", regex=False)
            return s2
    return col


def cast_column(sr: pd.Series, expected: str, params: dict, logs: dict, colname: str) -> pd.Series:
    before_dtype = str(sr.dtype)
    invalid_count = 0
    trimmed_count = 0

    # безопасная гигиена строкового представления
    s = sr.map(normalize_text)
    mask_trim = (s != sr) & ~(s.isna() & sr.isna())
    trimmed_count = int(mask_trim.sum())

    if expected in ("string", "category"):
        # только гигиена
        out = s
    elif expected in ("int", "float"):
        dec = (params.get("casting") or {}).get("decimal_sep", ".")
        policy = (params.get("casting") or {}).get("numeric_thousand_sep_policy", "homogeneous")
        s2 = s
        if policy == "homogeneous":
            s2 = maybe_strip_thousand_sep(s2, dec)
        if dec == ",":
            s2 = s2.str.replace(",", ".", regex=False)
        out = pd.to_numeric(s2, errors="coerce")
        invalid_count = int(out.isna().sum() - s2.isna().sum())
        if expected == "int":
            out = out.astype("Int64")
        else:
            out = out.astype("Float64")
    elif expected == "datetime":
        dt_cfg = (params.get("casting") or {}).get("datetime", {})
        fmt_list = dt_cfg.get("formats", None)
        if fmt_list:
            parsed = pd.NaT
            for fmt in fmt_list:
                parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True)
                if parsed.notna().any():
                    break
        else:
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
        invalid_count = int(parsed.isna().sum() - s.isna().sum())
        out = parsed
    elif expected == "bool":
        mapp = (params.get("booleans") or {}).get("mappings", {}).get(colname)
        if not mapp:
            # без явного маппинга — не трогаем
            out = s
        else:
            out = s.map(lambda v: mapp.get(v, None))
            out = out.astype("boolean")
            invalid_count = int(out.isna().sum() - s.isna().sum())
    else:
        out = s

    # логи
    logs["type_casts"].append(
        {
            "column": colname,
            "from_dtype": before_dtype,
            "to_dtype": expected,
        }
    )
    if trimmed_count:
        logs["hygiene_trimmed"][colname] = trimmed_count
    if invalid_count:
        logs["invalid_to_nan"][colname] = invalid_count

    return out


def main():
    schema = load_yaml(CFG / "schema.yaml")
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    _cols = load_yaml(CFG / "columns.yaml")

    # вход
    df_in, in_path = read_input(paths, params)
    df = df_in.copy()

    # подготовка логов
    base_log = load_previous_log()
    logs = base_log.copy()
    logs.update(
        {
            "checked_at_cast": datetime.now(timezone.utc).isoformat(),
            "cast_input_path": str(in_path),
            "type_casts": [],
            "invalid_to_nan": {},
            "hygiene_trimmed": {},
            "category_case_normalized": {},
            "notes_stats": {},
            "warnings": logs.get("warnings", []),
        }
    )

    # карта ожидаемых типов
    exp = {c["name"]: c["type"] for c in schema["columns"]}
    soft = params.get("soft_checks", {}) or {}
    max_fail = float(soft.get("max_cast_fail_rate", 0.02))

    # каноны для категорий
    canon = (params.get("categories") or {}).get("canonical_cases", {})

    # обработка колонок по схеме
    for col, expected in exp.items():
        if col not in df.columns:
            continue

        # особый кейс notes: считаем свою статистику
        if col == "notes":
            before = df[col]
            after = before.map(normalize_text)
            mask_trim = (before != after) & ~(before.isna() & after.isna())
            trimmed = int(mask_trim.sum())
            len_delta = (before.fillna("").map(len) - after.fillna("").map(len)).sum()
            df[col] = after
            logs["notes_stats"] = {
                "trimmed_count": trimmed,
                "len_delta_total": int(len_delta),
                "empty_share_after": float((after.fillna("") == "").mean()),
            }
            continue

        # каст/гигиена по типу
        df[col] = cast_column(df[col], expected, params, logs, col)

        # переводим в pandas 'category' и опциональная канонизация регистра
        if expected == "category":
            if col in canon:
                mapped, cnt = canonicalize_case(df[col], canon[col])
                df[col] = mapped
                if cnt:
                    logs["category_case_normalized"][col] = int(cnt)
            df[col] = df[col].astype("category")

        # проверка доли неудачных кастов
        if expected in ("int", "float", "datetime", "bool"):
            s = df[col]
            # считаем fail_rate относительно непустых исходно
            before_non_null = df_in[col].notna().sum() if col in df_in.columns else len(df)
            after_null = s.isna().sum()
            # грубая эвристика на fail_rate
            if before_non_null:
                fail_rate = max(0.0, (after_null - df_in[col].isna().sum()) / before_non_null)
                if fail_rate > max_fail:
                    logs["warnings"].append(
                        f"cast_fail_rate for {col} = {fail_rate:.4f} > {max_fail}"
                    )

    # запись результата
    out_path = DATA / "clean" / "telco_cast.parquet"
    (DATA / "clean").mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        print(f"Parquet не записан ({e}). Fallback в CSV.", file=sys.stderr)
        out_path = DATA / "clean" / "telco_cast.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

    logs["cast_output_path"] = str(out_path)
    save_log(logs)
    print(json.dumps({"status": "ok", "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
