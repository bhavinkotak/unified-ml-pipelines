"""Two feature paths over the same raw records.

* Single definition (feature-store pattern): `FeatureView.compute` is the only feature logic. Offline
  backfill replays history through it; online serving calls it per request against an online store.
* Dual implementation: a vectorised pandas batch job for training and a separate per-request Python
  function for serving. Five documented divergence sources can be switched on independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DatasetSpec

TRAIN_FRACTION = 0.7
TEST_FRACTION = 0.3
ALPHA = 20.0              # pseudo-count that smooths group aggregates toward their prior
REFRESH_INTERVAL = 1000   # requests between refreshes of the stale serving snapshot
SERVING_DECIMALS = 2      # decimals kept by the serving JSON serialiser (precision source)
SKEW_ATOL = 1e-9
SKEW_RTOL = 1e-9
MISSING_GROUP = "__missing__"
SOURCES = ("missing_default", "precision", "encoding_order", "stale_window", "pit_leakage")


@dataclass(frozen=True)
class Divergence:
    """Switches for the dual implementation. All False means the dual path equals the single path."""
    missing_default: bool = False   # serving fills a missing numeric field with 0.0, training with the median
    precision: bool = False         # serving casts to float32 and rounds derived features to 2 decimals
    encoding_order: bool = False    # serving vocabulary in first-seen file order, unknown -> 0 (training: sorted, -1)
    stale_window: bool = False      # serving reads aggregates from a snapshot refreshed every REFRESH_INTERVAL
    pit_leakage: bool = False       # batch aggregates use the whole table (self and future rows), not point in time

    @classmethod
    def all(cls) -> Divergence:
        return cls(**{s: True for s in SOURCES})

    @classmethod
    def only(cls, source: str) -> Divergence:
        return cls(**{source: True})


@dataclass
class Statics:
    """Constants learned from training rows only and shared by every path."""
    medians: dict[str, float]
    vocab: dict[str, dict[str, int]]          # sorted vocabulary
    first_seen: dict[str, dict[str, int]]     # first-appearance order in the original file
    prior_rate: float

    @classmethod
    def fit(cls, spec: DatasetSpec, X: pd.DataFrame, y: np.ndarray, file_row: np.ndarray) -> Statics:
        medians = {}
        for c in spec.numeric:
            v = X[c].to_numpy(dtype=float)
            medians[c] = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
        order = np.argsort(file_row, kind="stable")
        vocab, first = {}, {}
        for c in spec.categorical:
            vals = [v for v in X[c].to_numpy()[order] if v is not None]
            vocab[c] = {v: i for i, v in enumerate(sorted(set(vals)))}
            first[c] = {v: i for i, v in enumerate(dict.fromkeys(vals))}
        return cls(medians=medians, vocab=vocab, first_seen=first, prior_rate=float(np.mean(y)))


def feature_names(spec: DatasetSpec) -> list[str]:
    return [*spec.numeric, "ratio", *spec.categorical, "grp_count", "grp_mean", "grp_pos_rate"]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _group(row: dict, spec: DatasetSpec) -> str:
    g = row[spec.group_key]
    return MISSING_GROUP if _missing(g) else g


@dataclass
class AggState:
    """Per-group running (count, sum of agg_col, positives) over events strictly before now."""
    table: dict[str, list[float]] = field(default_factory=dict)

    def get(self, g: str) -> tuple[float, float, float]:
        e = self.table.get(g)
        return (0.0, 0.0, 0.0) if e is None else (e[0], e[1], e[2])

    def update(self, g: str, value: float, label: int) -> None:
        e = self.table.setdefault(g, [0.0, 0.0, 0.0])
        e[0] += 1.0
        e[1] += value
        e[2] += float(label)

    def copy(self) -> AggState:
        return AggState({k: list(v) for k, v in self.table.items()})


def smoothed(count: float, total: float, positives: float, agg_prior: float, rate_prior: float):
    return ((total + ALPHA * agg_prior) / (count + ALPHA), (positives + ALPHA * rate_prior) / (count + ALPHA))


class FeatureView:
    """The single feature definition. Unknown or missing category -> -1; missing numeric -> training median;
    missing aggregate (unknown entity or store timeout) -> count 0, i.e. the smoothing priors."""

    def __init__(self, spec: DatasetSpec, statics: Statics):
        self.spec, self.s = spec, statics

    def numeric(self, row: dict, c: str) -> float:
        v = row[c]
        return self.s.medians[c] if _missing(v) else float(v)

    def agg_value(self, row: dict) -> float:
        return self.numeric(row, self.spec.agg_col)

    def compute(self, row: dict, agg: tuple[float, float, float] | None) -> list[float]:
        sp = self.spec
        nums = [self.numeric(row, c) for c in sp.numeric]
        a, b = self.numeric(row, sp.ratio[0]), self.numeric(row, sp.ratio[1])
        cats = [float(self.s.vocab[c].get(row[c], -1)) if not _missing(row[c]) else -1.0 for c in sp.categorical]
        count, total, pos = agg if agg is not None else (0.0, 0.0, 0.0)
        mean, rate = smoothed(count, total, pos, self.s.medians[sp.agg_col], self.s.prior_rate)
        return [*nums, a / (b + 1.0), *cats, count, mean, rate]


def replay(view: FeatureView, rows: list[dict], labels: np.ndarray, *, online_from: int | None = None,
           timeout_rate: float = 0.0, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Point-in-time replay of the single definition. Rows at index >= online_from are online requests whose
    store lookup times out with probability timeout_rate; a timeout serves the documented fallback (no aggregate)
    and marks the request degraded. Labels are applied after each prediction (zero label delay)."""
    state = AggState()
    out = np.empty((len(rows), len(feature_names(view.spec))))
    degraded = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        g = _group(row, view.spec)
        agg: tuple[float, float, float] | None = state.get(g)
        if online_from is not None and i >= online_from and timeout_rate > 0 and rng is not None \
                and rng.random() < timeout_rate:
            agg, degraded[i] = None, True
        out[i] = view.compute(row, agg)
        state.update(g, view.agg_value(row), int(labels[i]))
    return out, degraded


def batch_features(spec: DatasetSpec, s: Statics, X: pd.DataFrame, y: np.ndarray, *, leakage: bool) -> np.ndarray:
    """Dual path, training side: vectorised pandas over the table it is given (rows in arrival order)."""
    num = pd.DataFrame({c: X[c].astype(float).fillna(s.medians[c]) for c in spec.numeric}, index=X.index)
    ratio = num[spec.ratio[0]] / (num[spec.ratio[1]] + 1.0)
    cats = pd.DataFrame({c: pd.Categorical(X[c], categories=list(s.vocab[c])).codes.astype(float)
                         for c in spec.categorical}, index=X.index)
    g = X[spec.group_key].astype(object).where(X[spec.group_key].notna(), MISSING_GROUP)
    v = num[spec.agg_col]
    lab = pd.Series(y.astype(float), index=X.index)
    if leakage:
        count = g.map(g.value_counts()).astype(float)
        total = v.groupby(g).transform("sum")
        pos = lab.groupby(g).transform("sum")
    else:
        count = g.groupby(g).cumcount().astype(float)
        total = v.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
        pos = lab.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
    prior_v = s.medians[spec.agg_col]
    mean = (total + ALPHA * prior_v) / (count + ALPHA)
    rate = (pos + ALPHA * s.prior_rate) / (count + ALPHA)
    out = pd.concat([num, ratio.rename("ratio"), cats, count.rename("grp_count"), mean.rename("grp_mean"),
                     rate.rename("grp_pos_rate")], axis=1)
    return out.to_numpy(dtype=float)


def serve_requests(spec: DatasetSpec, s: Statics, history: list[dict], history_labels: np.ndarray,
                   requests: list[dict], labels: np.ndarray, div: Divergence) -> np.ndarray:
    """Dual path, serving side: per-request Python written independently of the batch job."""
    vocab = s.first_seen if div.encoding_order else s.vocab
    unknown = 0.0 if div.encoding_order else -1.0

    def num(row: dict, c: str) -> float:
        v = row[c]
        if _missing(v):
            return 0.0 if div.missing_default else s.medians[c]
        return float(np.float32(v)) if div.precision else float(v)

    state = AggState()
    for row, lab in zip(history, history_labels, strict=True):
        state.update(_group(row, spec), num(row, spec.agg_col), int(lab))
    snapshot = state.copy()
    out = np.empty((len(requests), len(feature_names(spec))))
    for i, row in enumerate(requests):
        if div.stale_window and i > 0 and i % REFRESH_INTERVAL == 0:
            snapshot = state.copy()
        g = _group(row, spec)
        count, total, pos = (snapshot if div.stale_window else state).get(g)
        nums = [num(row, c) for c in spec.numeric]
        a, b = num(row, spec.ratio[0]), num(row, spec.ratio[1])
        mean, rate = smoothed(count, total, pos, s.medians[spec.agg_col], s.prior_rate)
        if div.precision:
            ratio = round(float(np.float32(a) / (np.float32(b) + np.float32(1.0))), SERVING_DECIMALS)
            mean, rate = round(mean, SERVING_DECIMALS), round(rate, SERVING_DECIMALS)
        else:
            ratio = a / (b + 1.0)
        cats = [unknown if _missing(row[c]) else float(vocab[c].get(row[c], unknown)) for c in spec.categorical]
        out[i] = [*nums, ratio, *cats, count, mean, rate]
        state.update(g, num(row, spec.agg_col), int(labels[i]))
    return out


def mismatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise feature mismatch: |a - b| > SKEW_ATOL + SKEW_RTOL * |a|."""
    return np.abs(a - b) > SKEW_ATOL + SKEW_RTOL * np.abs(a)
