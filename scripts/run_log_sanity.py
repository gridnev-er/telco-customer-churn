from __future__ import annotations

import hashlib
import io
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

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


def load_json_safe(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


def tooling_versions() -> Dict[str, str]:
    vers = {"pandas": pd.__version__}
    try:
        import pyarrow  # type: ignore

        vers["pyarrow"] = pyarrow.__version__
    except Exception:
        pass
    try:
        import fastparquet  # type: ignore

        vers["fastparquet"] = fastparquet.__version__
    except Exception:
        pass
    return vers


def git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return None


# ---------- IO helpers
def read_parquet_robust(p: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")


def read_input_after(params: dict, paths: dict) -> tuple[pd.DataFrame, Path]:
    candidates = [
        DATA / "clean" / "telco.parquet",
        DATA / "clean" / "telco_cast.parquet",
    ]
    for p in candidates:
        if p.exists():
            return read_parquet_robust(p), p
    raise FileNotFoundError(
        "clean not found: ожидается data/clean/telco.parquet или telco_cast.parquet"
    )


def read_input_before(params: dict, paths: dict) -> tuple[pd.DataFrame, Path, str]:
    dedup_csv = DATA / "clean" / "telco_dedup.csv"
    if dedup_csv.exists():
        rp = params.get("reader_params", {})
        header = 0 if rp.get("header", True) else None
        df = pd.read_csv(
            dedup_csv,
            encoding=rp.get("encoding", "utf-8"),
            sep=rp.get("delimiter", ","),
            header=header,
            na_values=rp.get("na_values", ["", "NA", "N/A"]),
            dtype=str,
            keep_default_na=True,
        )
        return df, dedup_csv, "dedup_csv"
    raw_path = ROOT / paths.get("raw_dir", "data/raw") / paths.get("raw_filename", "telco.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw not found: {raw_path}")
    if raw_path.suffix == ".parquet":
        return read_parquet_robust(raw_path), raw_path, "raw_parquet"
    else:
        rp = params.get("reader_params", {})
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
        return df, raw_path, "raw_csv"


# ---------- metrics
def content_hash_csv(df: pd.DataFrame) -> str:
    # каноничный CSV-поток: index=False, sep=','; пропуски = '', даты — ISO в UTC
    work = df.copy()
    for c in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[c]):
            s = pd.to_datetime(work[c], errors="coerce", utc=True)
            iso = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            work[c] = iso.mask(s.isna(), other="")
    buf = io.StringIO()
    work.to_csv(buf, index=False, sep=",", lineterminator="\n", na_rep="")
    return md5_bytes(buf.getvalue().encode("utf-8"))


def missing_share_per_column(df: pd.DataFrame) -> Dict[str, float]:
    # для строк/object: пустая строка считаем пропуском
    shares: Dict[str, float] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            m = s.isna() | (s.astype(str) == "")
        else:
            m = s.isna()
        shares[c] = float(m.mean()) if len(s) else 0.0
    return shares


# ---------- sanity helpers
def dtypes_match_schema(df: pd.DataFrame, schema: dict) -> bool:
    # Простая проверка: столбцы из схемы присутствуют; числа/даты/строки
    fam: Dict[str, str] = {}
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            fam[c] = "int"
        elif pd.api.types.is_float_dtype(df[c]):
            fam[c] = "float"
        elif pd.api.types.is_bool_dtype(df[c]):
            fam[c] = "bool"
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            fam[c] = "datetime"
        elif str(df[c].dtype) == "category":
            fam[c] = "category"
        else:
            fam[c] = "string"
    ok = True
    for col in schema.get("columns", []):
        name = col.get("name")
        exp = col.get("type")
        if name not in df.columns:
            ok = False
            continue
        got = fam.get(name, "")
        # грубое соответствие по семействам
        if exp in {"int", "float", "bool", "datetime", "category", "string"} and got != exp:
            # допускаем, что category может быть string на этом этапе
            if not (exp == "category" and got == "string"):
                ok = False
    return ok


# ---------- main
def main():
    t0 = time.perf_counter()

    schema = load_yaml(CFG / "schema.yaml")
    params = load_yaml(CFG / "params.yaml")
    paths = load_yaml(CFG / "paths.yaml")
    columns_cfg = load_yaml(CFG / "columns.yaml")
    ingest_meta = load_json_safe(REPORTS / "ingest_meta.json")
    clean_meta = load_json_safe(REPORTS / "clean_meta.json")
    profile = load_json_safe(REPORTS / "data_profile.json")
    log_prev = load_json_safe(REPORTS / "data_clean_log.json")

    # --- read BEFORE / AFTER
    t_read_raw0 = time.perf_counter()
    df_before, before_path, before_src = read_input_before(params, paths)
    read_raw_ok = True
    t_read_raw1 = time.perf_counter()

    t_read_clean0 = time.perf_counter()
    df_after, after_path = read_input_after(params, paths)
    read_clean_ok = True
    t_read_clean1 = time.perf_counter()

    # --- sizes & shares
    miss_before = missing_share_per_column(df_before)
    miss_after = missing_share_per_column(df_after)
    miss_delta = {
        k: round(miss_after.get(k, 0.0) - miss_before.get(k, 0.0), 12) for k in df_after.columns
    }

    # --- label stats (из after)
    label_col = columns_cfg.get("label_col") or "Churn"
    if label_col in df_after.columns:
        counts = df_after[label_col].value_counts(dropna=False)
        yes = int(counts.get("Yes", 0))
        no = int(counts.get("No", 0))
        null_count = int(df_after[label_col].isna().sum())
        invalid_count = int(
            (~df_after[label_col].isin(["Yes", "No"]) & df_after[label_col].notna()).sum()
        )
        total = len(df_after)
        positive_share = (yes / total) if total else 0.0
    else:
        yes = no = null_count = invalid_count = 0
        positive_share = 0.0

    # --- content hash recompute (after)
    t_hash0 = time.perf_counter()
    content_hash_now = content_hash_csv(df_after)
    t_hash1 = time.perf_counter()

    # --- sanity checks
    label_atol = float(params.get("sanity", {}).get("label_share_atol", 1e-6))
    missing_atol = float(params.get("sanity", {}).get("missing_share_atol", 1e-6))

    label_from_profile = profile.get("label", {}).get("positive_share")
    label_share_match = abs((label_from_profile or 0.0) - positive_share) <= label_atol

    missing_prof = profile.get("missingness", {}).get("missing_by_column", {})
    missing_share_close = True
    if isinstance(missing_prof, dict) and missing_prof:
        for c, obj in missing_prof.items():
            after_share = miss_after.get(c, 0.0)
            ref_share = float(obj.get("missing_share", 0.0))
            if abs(after_share - ref_share) > missing_atol:
                missing_share_close = False
                break

    # Типы vs схема (грубая проверка семейств)
    schema_ok = dtypes_match_schema(df_after, schema)

    # Сверка с clean_meta (если есть)
    n_rows_match = True
    n_cols_match = True
    content_hash_match = True
    if clean_meta:
        n_rows_match = int(clean_meta.get("n_rows", -1)) == len(df_after)
        n_cols_match = int(clean_meta.get("n_cols", -1)) == len(df_after.columns)
        ch = clean_meta.get("content_hash")
        if ch:
            content_hash_match = ch == content_hash_now

    # Тест записи/чтения (temp)
    t_write0 = time.perf_counter()
    write_ok = True
    tmp_path = DATA / "clean" / "_sanity_tmp.parquet"
    try:
        df_after.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
        _ = read_parquet_robust(tmp_path)
    except Exception:
        write_ok = False
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
    t_write1 = time.perf_counter()

    # --- assemble log
    run_id = datetime.now(timezone.utc).isoformat()
    git_short = git_commit_short()
    dataset_name = params.get("dataset_name", "telco")
    schema_version = schema.get("schema_version")

    # возьмём существующие разделы из старого лога
    log = log_prev if isinstance(log_prev, dict) else {}

    log["run"] = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "git_commit": git_short,
        "schema_version": schema_version,
        "params_fingerprint": clean_meta.get("params_fingerprint") if clean_meta else None,
    }

    raw_rows = int(len(df_before))
    raw_cols = int(len(df_before.columns))
    log["raw"] = {
        "path": str(before_path.relative_to(ROOT))
        if before_path.is_absolute()
        else str(before_path),
        "source": before_src,
        "md5": ingest_meta.get("md5"),
        "n_rows": raw_rows,
        "n_cols": raw_cols,
    }

    clean_rows = int(len(df_after))
    clean_cols = int(len(df_after.columns))
    clean_path = after_path
    log["clean"] = {
        "path": str(clean_path.relative_to(ROOT)) if clean_path.is_absolute() else str(clean_path),
        "content_hash": content_hash_now,
        "n_rows": clean_rows,
        "n_cols": clean_cols,
    }

    # missing snapshot
    # объёмы могут быть большими — оставим топ-200 по алф.коду
    def pack_missing(d: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        return {k: {"missing_share": round(float(v), 6)} for k, v in d.items()}

    log["missing_snapshot"] = {
        "before": pack_missing(miss_before),
        "after": pack_missing(miss_after),
        "delta": {k: round(float(v), 6) for k, v in miss_delta.items()},
    }

    # label summary
    log["label"] = {
        "column": label_col,
        "positive_class": "Yes",
        "counts": {"Yes": yes, "No": no},
        "positive_share": round(float(positive_share), 6),
        "null_count": null_count,
        "invalid_count": invalid_count,
    }

    # sanity flags
    log["sanity"] = {
        "read_raw_ok": read_raw_ok,
        "write_clean_ok": write_ok,
        "read_clean_ok": read_clean_ok,
        "n_rows_match": n_rows_match,
        "n_cols_match": n_cols_match,
        "content_hash_match": content_hash_match,
        "label_share_match": label_share_match,
        "missing_share_close": missing_share_close,
        "schema_types_match": schema_ok,
    }

    # timing
    log["timing_sec"] = {
        "read_raw": round(t_read_raw1 - t_read_raw0, 3),
        "read_clean": round(t_read_clean1 - t_read_clean0, 3),
        "hash_content": round(t_hash1 - t_hash0, 3),
        "write_tmp": round(t_write1 - t_write0, 3),
        "total": round(time.perf_counter() - t0, 3),
    }

    # tooling версий
    log["tooling"] = tooling_versions()

    # сохранить
    REPORTS.mkdir(parents=True, exist_ok=True)
    save_json(REPORTS / "data_clean_log.json", log)

    print(
        json.dumps({"status": "ok", "written": "reports/data_clean_log.json"}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
