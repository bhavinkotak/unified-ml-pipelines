import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from umlp import gate, lineage, skew  # noqa: E402
from umlp.data import Dataset, DatasetSpec  # noqa: E402
from umlp.features import (  # noqa: E402
    REFRESH_INTERVAL, SOURCES, AggState, Divergence, FeatureView, Statics, batch_features, feature_names, mismatch,
    replay, serve_requests,
)

SPEC = DatasetSpec(name="toy", openml_id=0, target="t", positive="1", numeric=("a", "b", "c"),
                   categorical=("k", "m"), group_key="k", agg_col="a", ratio=("a", "b"), shift_col="c",
                   drop_col="m")


def toy(n: int = 600, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(10, 3, n).round(3), "b": rng.integers(0, 50, n).astype(float),
                      "c": rng.normal(0, 1, n)})
    X["k"] = pd.Series(rng.choice(["z", "x", "y", "w"], n), dtype=object)
    X["m"] = pd.Series(rng.choice(["p", "q"], n), dtype=object)
    X.loc[rng.random(n) < 0.05, "k"] = None
    logit = 0.3 * (X["a"] - 10) + 0.8 * (X["k"] == "x") - 0.5 * X["c"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    return Dataset(spec=SPEC, X=X, y=y, sha256="toy")


@pytest.fixture(scope="module")
def split():
    return skew.make_split(toy(), 0)


def test_dual_without_divergence_equals_single_definition(split):
    sp = split
    n = sp.n_train
    F, _ = replay(FeatureView(SPEC, sp.statics), sp.rows, sp.y)
    F_train = batch_features(SPEC, sp.statics, sp.X.iloc[:n], sp.y[:n], leakage=False)
    F_full = batch_features(SPEC, sp.statics, sp.X, sp.y, leakage=False)
    F_on = serve_requests(SPEC, sp.statics, sp.rows[:n], sp.y[:n], sp.rows[n:], sp.y[n:], Divergence())
    assert not mismatch(F[:n], F_train).any()
    assert not mismatch(F[n:], F_full[n:]).any()
    assert not mismatch(F[n:], F_on).any()


@pytest.mark.parametrize("source", SOURCES)
def test_each_divergence_source_creates_skew(split, source):
    sp = split
    n = sp.n_train
    div = Divergence.only(source)
    F_off = batch_features(SPEC, sp.statics, sp.X, sp.y, leakage=div.pit_leakage)[n:]
    F_on = serve_requests(SPEC, sp.statics, sp.rows[:n], sp.y[:n], sp.rows[n:], sp.y[n:], div)
    assert mismatch(F_off, F_on).any(axis=1).mean() > 0


def test_point_in_time_aggregates_exclude_current_and_future_rows():
    st = Statics(medians={"a": 0.0, "b": 0.0, "c": 0.0}, vocab={"k": {"x": 0}, "m": {}},
                 first_seen={"k": {"x": 0}, "m": {}}, prior_rate=0.5)
    rows = [{"a": 1.0, "b": 0.0, "c": 0.0, "k": "x", "m": None}, {"a": 3.0, "b": 0.0, "c": 0.0, "k": "x", "m": None}]
    F, _ = replay(FeatureView(SPEC, st), rows, np.array([1, 0]))
    names = feature_names(SPEC)
    assert F[0, names.index("grp_count")] == 0.0
    assert F[1, names.index("grp_count")] == 1.0
    assert F[1, names.index("grp_pos_rate")] == pytest.approx((1 + 20 * 0.5) / 21)


def test_unknown_and_missing_inputs_use_documented_defaults(split):
    view = FeatureView(SPEC, split.statics)
    row = {"a": None, "b": float("nan"), "c": None, "k": "never-seen", "m": None}
    f = view.compute(row, None)
    names = feature_names(SPEC)
    assert f[names.index("a")] == split.statics.medians["a"]
    assert f[names.index("k")] == -1.0 and f[names.index("m")] == -1.0
    assert f[names.index("grp_count")] == 0.0


def test_timeouts_fall_back_and_flag_requests(split):
    sp = split
    view = FeatureView(SPEC, sp.statics)
    F, _ = replay(view, sp.rows, sp.y)
    F_t, degraded = replay(view, sp.rows, sp.y, online_from=sp.n_train, timeout_rate=1.0,
                           rng=np.random.default_rng(0))
    assert degraded[sp.n_train:].all() and not degraded[:sp.n_train].any()
    assert (F_t[sp.n_train:, feature_names(SPEC).index("grp_count")] == 0).all()
    assert not mismatch(F[:sp.n_train], F_t[:sp.n_train]).any()


def test_stale_snapshot_refreshes_on_interval():
    st = AggState()
    st.update("g", 1.0, 1)
    snap = st.copy()
    st.update("g", 1.0, 1)
    assert snap.get("g")[0] == 1.0 and st.get("g")[0] == 2.0
    assert REFRESH_INTERVAL > 0


def test_pinned_rerun_is_bit_identical(split):
    sp = split
    F, _ = replay(FeatureView(SPEC, sp.statics), sp.rows, sp.y)
    n = sp.n_train
    rows = lineage.repro_experiment(F[:n], sp.y[:n], F[n:], "toy")
    pinned = [r for r in rows if r["regime"] == "pinned"]
    assert all(r["identical_model"] and r["identical_pred"] and not r["resolved_lineage_changed"] for r in pinned)
    unpinned = [r for r in rows if r["regime"] == "unpinned_seed"]
    assert all(r["resolved_lineage_changed"] for r in unpinned)
    assert not any(r["explicit_lineage_changed"] for r in unpinned)


def test_lineage_hash_is_order_sensitive_only_when_requested():
    F = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([5, 4, 3, 2, 1, 0])
    assert lineage.data_hash(F, y, order_sensitive=False) == lineage.data_hash(F[p], y[p], order_sensitive=False)
    assert lineage.data_hash(F, y, order_sensitive=True) != lineage.data_hash(F[p], y[p], order_sensitive=True)


def test_gate_fails_closed_on_unknown_outcomes():
    assert gate.decide({"auc": True, "skew": True}) == (True, "promote")
    assert gate.decide({"auc": None, "skew": True})[0] is False
    assert gate.decide({"auc": True, "skew": float("nan")})[0] is False
    assert gate.decide({"auc": False, "skew": True})[0] is False
    assert gate.run_check(lambda: 1 / 0) is None
    assert gate._auc_check(float("nan"), 0.8) is None


def test_registry_keeps_incumbent_on_missing_or_tampered_version(split):
    sp = split
    F, _ = replay(FeatureView(SPEC, sp.statics), sp.rows, sp.y)
    m1 = skew.make_model(0, max_iter=5).fit(F[:sp.n_train], sp.y[:sp.n_train])
    m2 = skew.make_model(1, max_iter=5).fit(F[:sp.n_train], sp.y[:sp.n_train])
    reg = gate.Registry()
    reg.register("v1", m1, lineage.model_hash(m1))
    assert reg.promote("v1", True) == "v1"
    assert reg.promote("v2", True) == "v1"  # absent
    reg.register("v3", m2, lineage.model_hash(m1))  # artefact does not match its lineage
    assert reg.promote("v3", True) == "v1"
    reg.register("v4", m2, lineage.model_hash(m2))
    assert reg.promote("v4", False) == "v1"  # gate blocked
    assert reg.promote("v4", True) == "v4"


def test_ci95_and_verdict_rule():
    c = skew.ci95([0.01, 0.02, 0.03, 0.02, 0.02])
    assert c["ci_low"] < c["mean"] < c["ci_high"]
    mk = lambda m, lo, hi: {"base": {"dual_all": {"mean": m, "ci_low": lo, "ci_high": hi}}}  # noqa: E731
    s = {"delta_auc_online_single_minus_path": {"a": mk(0.02, 0.01, 0.03), "b": mk(0.01, 0.002, 0.02),
                                                "c": mk(0.0, -0.01, 0.01)}}
    assert skew.verdict(s)["verdict"] == "SUPPORTED"
    s = {"delta_auc_online_single_minus_path": {"a": mk(0.0, -0.001, 0.001), "b": mk(0.0, -0.002, 0.003),
                                                "c": mk(0.02, 0.01, 0.03)}}
    assert skew.verdict(s)["verdict"] == "REFUTED"
