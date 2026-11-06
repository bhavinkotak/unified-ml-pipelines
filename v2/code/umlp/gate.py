"""Simulated CI/CD quality gate. Every regression below is an injected simulation, not an observed incident."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .features import Divergence, Statics, batch_features, mismatch, serve_requests
from .lineage import array_hash, code_hash
from .skew import SEEDS, Split, make_model, make_split, safe_auc

GATE_FRACTION = 0.5       # first half of the serving requests is the gate's validation window
GATE_AUC_MARGIN = 0.005   # candidate offline AUC may not fall more than this below the incumbent
GATE_SKEW_MAX = 0.01      # max fraction of replayed requests with any offline/online feature mismatch
HARM_MARGIN = 0.005       # ground truth: online AUC on the post-gate window drops by more than this
LABEL_NOISE = 0.2
UNDERFIT_ITER = 5
SEED_OFFSET = 1000


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str                                   # "benign" or "injected"
    train_div: Divergence = Divergence()        # affects training and offline features
    serve_div: Divergence = Divergence()        # affects only the serving implementation
    model_overrides: dict = field(default_factory=dict)
    label_noise: float = 0.0
    drop_column: bool = False
    shuffle_rows: bool = False
    seed_offset: int = 0


CANDIDATES = (
    Candidate("seed_change", "benign", seed_offset=SEED_OFFSET),
    Candidate("learning_rate_0.09", "benign", model_overrides={"learning_rate": 0.09}),
    Candidate("row_shuffle", "benign", shuffle_rows=True),
    Candidate("label_noise", "injected", label_noise=LABEL_NOISE),
    Candidate("dropped_column", "injected", drop_column=True),
    Candidate("underfit", "injected", model_overrides={"max_iter": UNDERFIT_ITER}),
    Candidate("training_pit_leakage", "injected", train_div=Divergence(pit_leakage=True)),
    Candidate("serving_encoding_order", "injected", serve_div=Divergence(encoding_order=True)),
    Candidate("serving_missing_default", "injected", serve_div=Divergence(missing_default=True)),
    Candidate("serving_stale_window", "injected", serve_div=Divergence(stale_window=True)),
    Candidate("serving_precision", "injected", serve_div=Divergence(precision=True)),
)


def decide(checks: dict[str, bool | None]) -> tuple[bool, str]:
    """Fail closed: a check that errored, timed out, or produced no value (None/NaN) blocks promotion."""
    for name, ok in checks.items():
        if ok is None or (isinstance(ok, float) and math.isnan(ok)):
            return False, f"{name}: unknown"
        if not ok:
            return False, f"{name}: failed"
    return True, "promote"


def run_check(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - any error inside a check is an unknown outcome
        return None


class Registry:
    """Minimal model registry: serving keeps the incumbent whenever a version is absent or fails verification."""

    def __init__(self):
        self._store: dict[str, tuple[object, str]] = {}
        self.serving: str | None = None

    def register(self, version: str, model, lineage_model_hash: str) -> None:
        self._store[version] = (model, lineage_model_hash)

    def promote(self, version: str, gate_decision: bool) -> str | None:
        if gate_decision and self.load(version) is not None:
            self.serving = version
        return self.serving

    def load(self, version: str):
        from .lineage import model_hash
        entry = self._store.get(version)
        if entry is None:
            return None
        model, expected = entry
        return model if model_hash(model) == expected else None


def _auc_check(cand_auc: float, inc_auc: float) -> bool | None:
    if math.isnan(cand_auc) or math.isnan(inc_auc):
        return None
    return cand_auc >= inc_auc - GATE_AUC_MARGIN


def _pipeline(sp: Split, seed: int, c: Candidate):
    spec, n = sp.ds.spec, sp.n_train
    X, rows = sp.X, sp.rows
    statics = sp.statics
    if c.drop_column:  # upstream schema change: the column arrives empty in every row
        X = X.copy()
        X[spec.drop_col] = np.nan if spec.drop_col in spec.numeric else None
        rows = [dict(r, **{spec.drop_col: None}) for r in rows]
        statics = Statics.fit(spec, X.iloc[:n], sp.y[:n], sp.file_row[:n])
    y_train = sp.y[:n].copy()
    rng = np.random.default_rng(seed + SEED_OFFSET)
    if c.label_noise > 0:  # corrupted label source feeds training and the batch label aggregates
        flip = rng.random(n) < c.label_noise
        y_train = np.where(flip, 1 - y_train, y_train)
    F_train = batch_features(spec, statics, X.iloc[:n], y_train, leakage=c.train_div.pit_leakage)
    if c.shuffle_rows:
        perm = rng.permutation(n)
        F_fit, y_fit = F_train[perm], y_train[perm]
    else:
        F_fit, y_fit = F_train, y_train
    model = make_model(seed + c.seed_offset, **c.model_overrides).fit(F_fit, y_fit)
    y_all = np.concatenate([y_train, sp.y[n:]])
    F_off = batch_features(spec, statics, X, y_all, leakage=c.train_div.pit_leakage)[n:]
    F_on = serve_requests(spec, statics, rows[:n], y_train, rows[n:], sp.y[n:], c.serve_div)
    lineage = {"data": array_hash(F_train, y_train), "code": code_hash(), "params": str(model.get_params())}
    return model, F_off, F_on, lineage


def evaluate_candidate(sp: Split, seed: int, c: Candidate, incumbent: dict) -> dict:
    model, F_off, F_on, lineage = _pipeline(sp, seed, c)
    y = sp.y[sp.n_train:]
    k = int(GATE_FRACTION * len(y))
    p_off, p_on = model.predict_proba(F_off)[:, 1], model.predict_proba(F_on)[:, 1]
    auc_gate = safe_auc(y[:k], p_off[:k])
    skew_rate = float(mismatch(F_off[:k], F_on[:k]).any(axis=1).mean())
    lineage_ok = run_check(lambda: all(bool(v) for v in lineage.values()))
    auc_ok = run_check(_auc_check, auc_gate, incumbent["auc_gate"])
    skew_ok = run_check(lambda: skew_rate <= GATE_SKEW_MAX)
    promote_auc_only, _ = decide({"lineage": lineage_ok, "auc": auc_ok})
    promote_full, reason = decide({"lineage": lineage_ok, "auc": auc_ok, "skew": skew_ok})
    auc_post = safe_auc(y[k:], p_on[k:])
    return dict(dataset=sp.ds.spec.name, seed=seed, candidate=c.name, kind=c.kind,
                auc_gate_offline=auc_gate, skew_rate_gate=skew_rate, n_gate_requests=k,
                auc_post_online=auc_post, harmful=bool(auc_post < incumbent["auc_post"] - HARM_MARGIN),
                promote_auc_only=promote_auc_only, promote_auc_skew=promote_full, block_reason=reason)


def run_gate(ds, seeds=SEEDS) -> list[dict]:
    rows = []
    for seed in seeds:
        sp = make_split(ds, seed)
        inc_model, F_off, F_on, _ = _pipeline(sp, seed, Candidate("incumbent", "benign"))
        y = sp.y[sp.n_train:]
        k = int(GATE_FRACTION * len(y))
        incumbent = dict(auc_gate=safe_auc(y[:k], inc_model.predict_proba(F_off[:k])[:, 1]),
                         auc_post=safe_auc(y[k:], inc_model.predict_proba(F_on[k:])[:, 1]))
        for c in CANDIDATES:
            rows.append(evaluate_candidate(sp, seed, c, incumbent))
    return rows


def summarise_gate(rows: list[dict]) -> dict:
    out: dict = {"per_candidate": {}}
    for c in CANDIDATES:
        rs = [r for r in rows if r["candidate"] == c.name]
        out["per_candidate"][c.name] = dict(
            kind=c.kind, n=len(rs), n_harmful=sum(r["harmful"] for r in rs),
            n_blocked_auc_only=sum(not r["promote_auc_only"] for r in rs),
            n_blocked_auc_skew=sum(not r["promote_auc_skew"] for r in rs),
            mean_skew_rate_gate=float(np.mean([r["skew_rate_gate"] for r in rs])))
    for gate in ("auc_only", "auc_skew"):
        blocked = [not r[f"promote_{gate}"] for r in rows]
        harmful = [r["harmful"] for r in rows]
        tp = sum(b and h for b, h in zip(blocked, harmful, strict=True))
        fn = sum((not b) and h for b, h in zip(blocked, harmful, strict=True))
        fp = sum(b and not h for b, h in zip(blocked, harmful, strict=True))
        tn = sum((not b) and not h for b, h in zip(blocked, harmful, strict=True))
        inj = [r for r in rows if r["kind"] == "injected"]
        ben = [r for r in rows if r["kind"] == "benign"]
        out[gate] = dict(
            harmful_blocked=tp, harmful_promoted=fn, harmless_blocked=fp, harmless_promoted=tn,
            n_total=len(rows), n_harmful=tp + fn, n_harmless=fp + tn,
            recall_harmful=tp / (tp + fn) if tp + fn else float("nan"),
            harmless_blocked_rate=fp / (fp + tn) if fp + tn else float("nan"),
            injected_blocked=sum(not r[f"promote_{gate}"] for r in inj), n_injected=len(inj),
            benign_blocked=sum(not r[f"promote_{gate}"] for r in ben), n_benign=len(ben))
    return out
