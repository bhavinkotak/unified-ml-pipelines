"""Content-hash lineage and the rerun reproducibility experiment."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from . import features
from .skew import make_model

REPRO_RERUNS = 5
REF_SEED = 12345
THREAD_SETTINGS = (1, 2)  # pinned_threads regime alternates these OpenMP thread counts
SIMULATED_DEFAULT_CHANGE = {"l2_regularization": 1.0}  # simulated library default change ("version B")
REGIMES = ("pinned", "pinned_threads", "unpinned_seed", "data_order", "implicit_default")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def data_hash(F: np.ndarray, y: np.ndarray, *, order_sensitive: bool) -> str:
    """Order-insensitive hash identifies the table version; order-sensitive hash identifies the exact input."""
    if order_sensitive:
        return array_hash(F, y)
    rows = sorted(sha256(np.ascontiguousarray(r).tobytes() + bytes([int(t)])) for r, t in zip(F, y, strict=True))
    return sha256("".join(rows).encode())


def code_hash() -> str:
    return sha256((inspect.getsource(features) + inspect.getsource(make_model)).encode())


def lineage_hash(record: dict) -> str:
    return sha256(json.dumps(record, sort_keys=True, default=str).encode())


def model_hash(model) -> str:
    """Hash of the fitted trees and baseline, i.e. the decision function, not pickle metadata."""
    parts = [np.asarray(model._baseline_prediction)]
    parts += [p.nodes for stage in model._predictors for p in stage]
    return array_hash(*parts)


def lineage_records(F, y, explicit_params: dict, model, version: str) -> tuple[dict, dict]:
    """explicit: what a user-written params file records. resolved: concrete seed, full params, exact input."""
    code = code_hash()
    explicit = {"data": data_hash(F, y, order_sensitive=False), "code": code, "params": explicit_params}
    resolved = {"data": data_hash(F, y, order_sensitive=True), "code": code,
                "params": {k: v for k, v in model.get_params().items()},
                "env": {"sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version(),
                        "library_build": version}}
    return explicit, resolved


def _fit(F, y, seed, overrides: dict, threads: int | None):
    model = make_model(seed, **overrides)
    if threads is None:
        return model.fit(F, y)
    with threadpool_limits(limits=threads, user_api="openmp"):
        return model.fit(F, y)


def run_regime(regime: str, F: np.ndarray, y: np.ndarray, F_test: np.ndarray, rerun: int):
    """Returns (model, explicit_lineage, resolved_lineage, predictions) for one rerun of a regime.
    rerun = -1 is the regime's reference configuration."""
    seed, overrides, threads, version = REF_SEED, {}, None, "A"
    explicit_params = {"max_iter": features_params()["max_iter"], "random_state": REF_SEED}
    if regime == "pinned_threads":
        threads = THREAD_SETTINGS[rerun % 2]
    elif regime == "unpinned_seed":
        seed = int(np.random.SeedSequence().generate_state(1)[0])  # fresh OS entropy, as with random_state=None
        explicit_params["random_state"] = None
    elif regime == "data_order" and rerun >= 0:
        perm = np.random.default_rng([REF_SEED, rerun]).permutation(len(y))
        F, y = F[perm], y[perm]
    elif regime == "implicit_default" and rerun >= 0 and rerun % 2 == 0:
        overrides, version = dict(SIMULATED_DEFAULT_CHANGE), "B"
    model = _fit(F, y, seed, overrides, threads)
    explicit, resolved = lineage_records(F, y, explicit_params, model, version)
    return model, lineage_hash(explicit), lineage_hash(resolved), model.predict_proba(F_test)[:, 1]


def features_params() -> dict:
    return make_model(REF_SEED).get_params()


def repro_experiment(F: np.ndarray, y: np.ndarray, F_test: np.ndarray, dataset: str) -> list[dict]:
    rows = []
    for regime in REGIMES:
        ref_model, ref_e, ref_r, ref_pred = run_regime(regime, F, y, F_test, -1)
        ref = dict(model=model_hash(ref_model), pred=array_hash(ref_pred), explicit=ref_e, resolved=ref_r)
        for r in range(REPRO_RERUNS):
            model, lin_e, lin_r, pred = run_regime(regime, F, y, F_test, r)
            rows.append(dict(dataset=dataset, regime=regime, rerun=r,
                             identical_model=model_hash(model) == ref["model"],
                             identical_pred=array_hash(pred) == ref["pred"],
                             max_abs_pred_diff=float(np.max(np.abs(pred - ref_pred))),
                             explicit_lineage_changed=lin_e != ref["explicit"],
                             resolved_lineage_changed=lin_r != ref["resolved"]))
    return rows


def summarise_repro(rows: list[dict]) -> dict:
    out = {}
    for regime in REGIMES:
        rs = [r for r in rows if r["regime"] == regime]
        diff = [r for r in rs if not r["identical_pred"]]
        same = [r for r in rs if r["identical_pred"]]
        out[regime] = dict(
            n_reruns=len(rs),
            n_identical_model=sum(r["identical_model"] for r in rs),
            n_identical_pred=len(same),
            identical_pred_rate=len(same) / len(rs),
            n_nonidentical=len(diff),
            n_nonidentical_explained_explicit=sum(r["explicit_lineage_changed"] for r in diff),
            n_nonidentical_explained_resolved=sum(r["resolved_lineage_changed"] for r in diff),
            n_identical_flagged_resolved=sum(r["resolved_lineage_changed"] for r in same),
            max_abs_pred_diff=max(r["max_abs_pred_diff"] for r in rs),
        )
    return out


def write_lineage_example(path: Path, F, y, model) -> None:
    explicit, resolved = lineage_records(F, y, {"max_iter": features_params()["max_iter"], "random_state": REF_SEED},
                                         model, "A")
    path.write_text(json.dumps({"explicit": explicit, "resolved": resolved,
                                "explicit_hash": lineage_hash(explicit), "resolved_hash": lineage_hash(resolved)},
                               indent=1, sort_keys=True, default=str))
