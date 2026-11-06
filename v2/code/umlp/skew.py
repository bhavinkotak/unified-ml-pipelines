"""Training-serving skew experiment: single definition vs dual implementation, ablations, breakpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import Dataset
from .features import (
    SOURCES, TRAIN_FRACTION, Divergence, FeatureView, Statics, batch_features, feature_names, mismatch, replay,
    serve_requests,
)

SEEDS = (0, 1, 2, 3, 4)
MISSING_RATE = 0.05        # simulated MCAR missingness injected into numeric cells
MISSING_RATE_HIGH = 0.2    # breakpoint: 4x the base rate
SHIFT_NOISE = 0.5          # covariate-shift ordering noise, in standard deviations of the shift column
TIMEOUT_RATES = (0.01, 0.05, 0.2)
THRESHOLD = 0.5
MAX_ITER = 150
LEARNING_RATE = 0.1
MAX_FEATURES = 0.8
MIN_EFFECT = 0.005         # preregistered minimum meaningful AUC difference
CONFIDENCE = 0.95          # two-sided t interval over seeds


@dataclass
class Split:
    ds: Dataset
    X: pd.DataFrame
    y: np.ndarray
    file_row: np.ndarray
    n_train: int
    rows: list[dict]
    statics: Statics

    @property
    def n_test(self) -> int:
        return len(self.y) - self.n_train


def make_split(ds: Dataset, seed: int, *, missing_rate: float = MISSING_RATE, order: str = "random") -> Split:
    rng = np.random.default_rng(seed)
    X = ds.X.copy()
    num = list(ds.spec.numeric)
    if missing_rate > 0:
        mask = rng.random((len(X), len(num))) < missing_rate
        X[num] = X[num].mask(mask)
    n = len(X)
    if order == "random":
        perm = rng.permutation(n)
    elif order == "shift":
        col = X[ds.spec.shift_col].astype(float)
        key = col.fillna(col.median()).to_numpy() + rng.normal(0.0, SHIFT_NOISE * float(col.std()), n)
        perm = np.argsort(key, kind="stable")
    else:
        raise ValueError(order)
    X = X.iloc[perm].reset_index(drop=True)
    y = ds.y[perm]
    n_train = int(TRAIN_FRACTION * n)
    statics = Statics.fit(ds.spec, X.iloc[:n_train], y[:n_train], perm[:n_train])
    rows = X.to_dict("records")
    for r in rows:  # pandas may hand back NaN for missing categoricals; normalise to None
        for c in ds.spec.categorical:
            if not isinstance(r[c], str):
                r[c] = None
    return Split(ds, X, y, perm, n_train, rows, statics)


def make_model(seed: int | None, **overrides) -> HistGradientBoostingClassifier:
    params = dict(max_iter=MAX_ITER, learning_rate=LEARNING_RATE, max_features=MAX_FEATURES,
                  early_stopping=False, random_state=seed)
    params.update(overrides)
    return HistGradientBoostingClassifier(**params)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC is undefined when one class is absent; report NaN rather than a number."""
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def evaluate(model, F_off: np.ndarray, F_on: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    p_off, p_on = model.predict_proba(F_off)[:, 1], model.predict_proba(F_on)[:, 1]
    mm = mismatch(F_off, F_on)
    skewed = mm.any(axis=1)
    disagree = (p_off >= THRESHOLD) != (p_on >= THRESHOLD)
    out = dict(
        n_requests=len(y), n_skewed=int(skewed.sum()), skew_rate=float(skewed.mean()),
        n_cells=int(mm.size), n_cells_mismatched=int(mm.sum()), cell_mismatch_rate=float(mm.mean()),
        n_disagree=int(disagree.sum()), disagreement_rate=float(disagree.mean()),
        mean_abs_dprob=float(np.mean(np.abs(p_off - p_on))),
        auc_offline=safe_auc(y, p_off), auc_online=safe_auc(y, p_on),
        acc_offline=float(accuracy_score(y, p_off >= THRESHOLD)),
        acc_online=float(accuracy_score(y, p_on >= THRESHOLD)),
    )
    out["auc_gap"] = out["auc_offline"] - out["auc_online"]
    out["mismatched_features"] = ";".join(n for n, k in zip(names, mm.any(axis=0), strict=True) if k)
    return out


def single_path(sp: Split, seed: int):
    view = FeatureView(sp.ds.spec, sp.statics)
    F, _ = replay(view, sp.rows, sp.y)
    model = make_model(seed).fit(F[:sp.n_train], sp.y[:sp.n_train])
    return view, F, model


def dual_path(sp: Split, seed: int, div: Divergence):
    spec, s, n = sp.ds.spec, sp.statics, sp.n_train
    F_train = batch_features(spec, s, sp.X.iloc[:n], sp.y[:n], leakage=div.pit_leakage)
    F_off = batch_features(spec, s, sp.X, sp.y, leakage=div.pit_leakage)[n:]
    F_on = serve_requests(spec, s, sp.rows[:n], sp.y[:n], sp.rows[n:], sp.y[n:], div)
    model = make_model(seed).fit(F_train, sp.y[:n])
    return F_train, F_off, F_on, model


def run_dataset(ds: Dataset, log=print, seeds=SEEDS) -> tuple[list[dict], list[dict]]:
    """Returns (rows for skew_runs.csv, rows for control checks)."""
    names = feature_names(ds.spec)
    rows, controls = [], []

    def rec(scenario: str, path: str, seed: int, sp: Split, res: dict, **extra) -> None:
        rows.append(dict(dataset=ds.spec.name, scenario=scenario, path=path, seed=seed, n_train=sp.n_train,
                         **res, **extra))

    for seed in seeds:
        for scenario, kw in (("base", {}), ("missing_high", {"missing_rate": MISSING_RATE_HIGH}),
                             ("shift", {"order": "shift"})):
            sp = make_split(ds, seed, **kw)
            n = sp.n_train
            view, F, model = single_path(sp, seed)
            y_test = sp.y[n:]
            rec(scenario, "single", seed, sp, evaluate(model, F[n:], F[n:], y_test, names))
            _, F_off, F_on, m_dual = dual_path(sp, seed, Divergence.all())
            rec(scenario, "dual_all", seed, sp, evaluate(m_dual, F_off, F_on, y_test, names))
            if scenario != "base":
                continue
            # correctness control: the dual code with every source off must reproduce the single definition
            F_tr0 = batch_features(ds.spec, sp.statics, sp.X.iloc[:n], sp.y[:n], leakage=False)
            F_on0 = serve_requests(ds.spec, sp.statics, sp.rows[:n], sp.y[:n], sp.rows[n:], sp.y[n:], Divergence())
            controls.append(dict(dataset=ds.spec.name, seed=seed,
                                 max_abs_diff_train=float(np.max(np.abs(F_tr0 - F[:n]))),
                                 max_abs_diff_serve=float(np.max(np.abs(F_on0 - F[n:]))),
                                 n_mismatched_requests=int(mismatch(F[n:], F_on0).any(axis=1).sum())))
            for src in SOURCES:
                _, F_off, F_on, m = dual_path(sp, seed, Divergence.only(src))
                rec(scenario, f"dual_{src}", seed, sp, evaluate(m, F_off, F_on, y_test, names))
            for rate in TIMEOUT_RATES:
                F_on, degraded = replay(view, sp.rows, sp.y, online_from=n, timeout_rate=rate,
                                        rng=np.random.default_rng(seed + 7919))
                rec(scenario, f"single_timeout_{rate}", seed, sp, evaluate(model, F[n:], F_on[n:], y_test, names),
                    timeout_rate=rate, n_degraded=int(degraded[n:].sum()))
        log(f"  {ds.spec.name} seed {seed} done")
    return rows, controls


def ci95(x) -> dict:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf((1 + CONFIDENCE) / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return dict(n=n, mean=m, std=sd, ci_low=m - h, ci_high=m + h)


def summarise(df: pd.DataFrame) -> dict:
    out: dict = {}
    metrics = ("skew_rate", "cell_mismatch_rate", "disagreement_rate", "mean_abs_dprob", "auc_offline",
               "auc_online", "auc_gap", "acc_offline", "acc_online")
    for (dsn, scen, path), g in df.groupby(["dataset", "scenario", "path"], sort=True):
        d = out.setdefault(dsn, {}).setdefault(scen, {}).setdefault(path, {})
        for m in metrics:
            d[m] = ci95(g[m])
        d["n_requests_per_seed"] = int(g["n_requests"].iloc[0])
        d["n_skewed_total"] = int(g["n_skewed"].sum())
        d["n_requests_total"] = int(g["n_requests"].sum())
        d["n_disagree_total"] = int(g["n_disagree"].sum())
        if "n_degraded" in g and g["n_degraded"].notna().any():
            d["n_degraded_total"] = int(g["n_degraded"].sum())
    # paired online-AUC differences vs the single path of the same scenario and seed
    deltas: dict = {}
    for (dsn, scen), g in df.groupby(["dataset", "scenario"], sort=True):
        base = g[g.path == "single"].set_index("seed")["auc_online"]
        for path, h in g.groupby("path"):
            if path == "single":
                continue
            diff = base - h.set_index("seed")["auc_online"]
            deltas.setdefault(dsn, {}).setdefault(scen, {})[path] = ci95(diff.to_numpy())
    out["delta_auc_online_single_minus_path"] = deltas
    return out


def verdict(summary: dict, min_effect: float = MIN_EFFECT) -> dict:
    per = {}
    for dsn, scen in summary["delta_auc_online_single_minus_path"].items():
        d = scen["base"]["dual_all"]
        per[dsn] = dict(mean=d["mean"], ci_low=d["ci_low"], ci_high=d["ci_high"],
                        supports=bool(d["mean"] >= min_effect and d["ci_low"] > 0),
                        refutes=bool(d["ci_high"] < min_effect))
    n_sup = sum(v["supports"] for v in per.values())
    n_ref = sum(v["refutes"] for v in per.values())
    label = "SUPPORTED" if n_sup >= 2 else "REFUTED" if n_ref >= 2 else "INCONCLUSIVE"
    return dict(verdict=label, primary_endpoint="delta_auc_online = AUC_online(single) - AUC_online(dual_all)",
                min_effect=min_effect, datasets_supporting=n_sup, datasets_refuting=n_ref, per_dataset=per)
