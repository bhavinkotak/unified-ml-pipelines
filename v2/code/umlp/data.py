"""Public tabular datasets: download once from OpenML, cache under data/raw, verify by sha256."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_id: int
    target: str
    positive: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    group_key: str
    agg_col: str
    ratio: tuple[str, str]
    shift_col: str
    drop_col: str
    excluded: tuple[str, ...] = ()
    license: str = "CC BY 4.0 (UCI Machine Learning Repository); OpenML licence field: Public"


SPECS = (
    DatasetSpec(
        name="adult", openml_id=1590, target="class", positive=">50K",
        numeric=("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"),
        categorical=("workclass", "education", "marital-status", "occupation", "relationship", "race",
                     "sex", "native-country"),
        group_key="occupation", agg_col="hours-per-week", ratio=("capital-gain", "hours-per-week"),
        shift_col="age", drop_col="capital-gain"),
    DatasetSpec(
        name="bank-marketing", openml_id=1461, target="Class", positive="2",
        # V1 age, V6 balance, V10 day, V13 campaign, V14 pdays, V15 previous
        numeric=("V1", "V6", "V10", "V13", "V14", "V15"),
        # V2 job, V3 marital, V4 education, V5 default, V7 housing, V8 loan, V9 contact, V11 month, V16 poutcome
        categorical=("V2", "V3", "V4", "V5", "V7", "V8", "V9", "V11", "V16"),
        group_key="V2", agg_col="V6", ratio=("V6", "V13"), shift_col="V1", drop_col="V16",
        # V12 (call duration) is only known after the outcome; the UCI page says to discard it.
        excluded=("V12",)),
    DatasetSpec(
        name="credit-g", openml_id=31, target="class", positive="bad",
        numeric=("duration", "credit_amount", "installment_commitment", "residence_since", "age",
                 "existing_credits", "num_dependents"),
        categorical=("checking_status", "credit_history", "purpose", "savings_status", "employment",
                     "personal_status", "other_parties", "property_magnitude", "other_payment_plans",
                     "housing", "job", "own_telephone", "foreign_worker"),
        group_key="purpose", agg_col="credit_amount", ratio=("credit_amount", "duration"),
        shift_col="age", drop_col="checking_status"),
)
SPEC_BY_NAME = {s.name: s for s in SPECS}


@dataclass
class Dataset:
    spec: DatasetSpec
    X: pd.DataFrame  # numeric columns float (NaN = missing), categorical columns object (None = missing)
    y: np.ndarray    # int 0/1
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _download(spec: DatasetSpec, path: Path) -> None:
    from sklearn.datasets import fetch_openml
    last: Exception | None = None
    # transient OpenML failures/timeouts: retry with both ARFF parsers, then abort with the last error
    for parser in ("auto", "liac-arff", "liac-arff"):
        try:
            frame = fetch_openml(data_id=spec.openml_id, as_frame=True, parser=parser).frame
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
    else:
        raise RuntimeError(f"OpenML download failed for {spec.name}: {last}")
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def load(spec: DatasetSpec) -> Dataset:
    """Cached file is authoritative; a sha256 mismatch against the manifest aborts the run."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{spec.name}.csv.gz"
    manifest = _load_manifest()
    if not path.exists():
        _download(spec, path)
        manifest[spec.name] = {
            "file": path.name, "openml_id": spec.openml_id,
            "url": f"https://www.openml.org/d/{spec.openml_id}",
            "license": spec.license,
            "retrieved_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
            "sha256": sha256_file(path),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    digest = sha256_file(path)
    expected = manifest.get(spec.name, {}).get("sha256")
    if expected is not None and expected != digest:
        raise RuntimeError(f"{path} sha256 {digest} does not match manifest {expected}")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    X = pd.DataFrame(index=raw.index)
    for c in spec.numeric:
        X[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
    for c in spec.categorical:
        X[c] = pd.Series([None if pd.isna(v) or v == "?" else str(v) for v in raw[c]], dtype=object)
    y = (raw[spec.target].astype(str) == spec.positive).astype(int).to_numpy()
    return Dataset(spec=spec, X=X, y=y, sha256=digest)


def write_sources_md() -> None:
    manifest = _load_manifest()
    lines = ["# Data sources", "",
             "All files are cached by `code/run_all.py` via `sklearn.datasets.fetch_openml` and verified by sha256.", "",
             "| File | Dataset | URL | License | Retrieved (UTC) | sha256 |", "|---|---|---|---|---|---|"]
    for spec in SPECS:
        m = manifest.get(spec.name, {})
        lines.append(f"| data/raw/{m.get('file', '')} | {spec.name} (OpenML {spec.openml_id}) | {m.get('url', '')} | "
                     f"{m.get('license', '')} | {m.get('retrieved_utc', '')} | `{m.get('sha256', '')}` |")
    lines += ["", "Original sources: UCI Adult (https://archive.ics.uci.edu/dataset/2/adult), UCI Bank Marketing "
              "(https://archive.ics.uci.edu/dataset/222/bank+marketing), UCI Statlog German Credit "
              "(https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).",
              "Bank Marketing column V12 (call duration) is excluded because it is only known after the outcome."]
    (ROOT / "data" / "SOURCES.md").write_text("\n".join(lines) + "\n")
