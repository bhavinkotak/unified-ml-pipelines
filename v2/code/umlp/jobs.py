"""Process-pool job wrappers. Each job loads its dataset from the verified cache and seeds itself."""

from __future__ import annotations

from threadpoolctl import threadpool_limits

from . import data, gate, lineage, skew

N_THREADS = 1  # OpenMP threads per worker; parallelism comes from the process pool


def init_worker() -> None:
    threadpool_limits(limits=N_THREADS)


def skew_job(name: str, seed: int):
    ds = data.load(data.SPEC_BY_NAME[name])
    return skew.run_dataset(ds, log=lambda *_: None, seeds=(seed,))


def gate_job(name: str, seed: int):
    return gate.run_gate(data.load(data.SPEC_BY_NAME[name]), seeds=(seed,))


def repro_job(name: str):
    ds = data.load(data.SPEC_BY_NAME[name])
    sp = skew.make_split(ds, skew.SEEDS[0])
    _, F, model = skew.single_path(sp, skew.SEEDS[0])
    n = sp.n_train
    return lineage.repro_experiment(F[:n], sp.y[:n], F[n:], name), (F[:n], sp.y[:n], model)
