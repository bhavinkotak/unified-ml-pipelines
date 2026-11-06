"""Single entry point: cache data, run every experiment with fixed seeds, write results/, regenerate figures.

Usage: python code/run_all.py [--quick | --tables-only]
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_BLOCKTIME", "0")  # idle OpenMP threads sleep instead of spinning (shared CPUs)

import platform  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from umlp import data, figures, gate, jobs, lineage, skew, tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_WORKERS = 4  # independent (dataset, seed) jobs; every job seeds itself, so results do not depend on this


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="credit-g only, written to a temp dir, for smoke testing")
    ap.add_argument("--tables-only", action="store_true", help="re-render tables/figures from existing results")
    args = ap.parse_args()
    results, figs = ROOT / "results", ROOT / "paper" / "figures"
    if args.tables_only:  # re-derive every summary from the raw per-run CSVs, then re-render
        def redump(name: str, obj) -> None:
            (results / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
        summary = skew.summarise(pd.read_csv(results / "skew_runs.csv"))
        redump("skew_summary.json", summary)
        redump("verdict.json", skew.verdict(summary, skew.MIN_EFFECT))
        redump("repro_summary.json", lineage.summarise_repro(pd.read_csv(results / "repro_runs.csv").to_dict("records")))
        redump("gate_summary.json", gate.summarise_gate(pd.read_csv(results / "gate_runs.csv").to_dict("records")))
        figures.make_all(results, figs, skew.TIMEOUT_RATES)
        log(f"tables injected into main.tex: {tables.inject(results, ROOT / 'paper' / 'main.tex')}")
        return 0
    if args.quick:  # smoke runs never touch the reported results
        results = Path(tempfile.mkdtemp(prefix="umlp_quick_"))
        figs = results / "figures"
        log(f"quick mode: writing to {results}")
    results.mkdir(exist_ok=True)

    def dump(name: str, obj) -> None:
        (results / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))

    t0 = time.time()
    specs = [data.SPEC_BY_NAME["credit-g"]] if args.quick else list(data.SPECS)
    datasets = [data.load(s) for s in specs]
    data.write_sources_md()
    names = [s.name for s in specs]

    ds_summary = {}
    for ds in datasets:
        sp = skew.make_split(ds, skew.SEEDS[0])
        ds_summary[ds.spec.name] = dict(
            openml_id=ds.spec.openml_id, n_rows=len(ds.y), n_numeric=len(ds.spec.numeric),
            n_categorical=len(ds.spec.categorical), n_model_features=len(skew.feature_names(ds.spec)),
            positive_rate=float(ds.y.mean()), n_train=sp.n_train, n_serving_requests=sp.n_test,
            n_groups=int(ds.X[ds.spec.group_key].nunique()), group_key=ds.spec.group_key,
            n_missing_cells_raw=int(ds.X.isna().sum().sum()), sha256=ds.sha256)
    dump("dataset_summary.json", ds_summary)

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=jobs.init_worker) as pool:
        log("[1/3] training-serving skew experiment")
        futs = [pool.submit(jobs.skew_job, n, s) for n in names for s in skew.SEEDS]
        runs, controls = [], []
        for f in futs:
            r, c = f.result()
            runs += r
            controls += c
            log(f"  skew {r[0]['dataset']} seed {r[0]['seed']} done ({time.time() - t0:.0f}s)")
        df = pd.DataFrame(runs)
        df.to_csv(results / "skew_runs.csv", index=False)
        pd.DataFrame(controls).to_csv(results / "controls.csv", index=False)
        summary = skew.summarise(df)
        dump("skew_summary.json", summary)
        prereg = json.loads((ROOT / "preregistration.json").read_text())
        assert prereg["decision_rule"]["min_effect"] == skew.MIN_EFFECT, "code and preregistration disagree"
        dump("verdict.json", skew.verdict(summary, prereg["decision_rule"]["min_effect"]))

        log("[2/3] reproducibility and lineage")
        repro_rows = []
        for n, f in zip(names, [pool.submit(jobs.repro_job, n) for n in names], strict=True):
            rows, (F, y, model) = f.result()
            repro_rows += rows
            if n == names[0]:
                lineage.write_lineage_example(results / "lineage_example.json", F, y, model)
        pd.DataFrame(repro_rows).to_csv(results / "repro_runs.csv", index=False)
        dump("repro_summary.json", lineage.summarise_repro(repro_rows))

        log("[3/3] simulated CI gate")
        gate_rows = []
        for f in [pool.submit(jobs.gate_job, n, s) for n in names for s in skew.SEEDS]:
            gate_rows += f.result()
        pd.DataFrame(gate_rows).to_csv(results / "gate_runs.csv", index=False)
        dump("gate_summary.json", gate.summarise_gate(gate_rows))

    figures.make_all(results, figs, skew.TIMEOUT_RATES)
    if not args.quick:
        log(f"tables injected into main.tex: {tables.inject(results, ROOT / 'paper' / 'main.tex')}")
    dump("run_info.json", dict(runtime_s=round(time.time() - t0, 1), python=platform.python_version(),
                               numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                               machine=platform.machine(), cpu_count=os.cpu_count(), workers=N_WORKERS,
                               threads_per_worker=jobs.N_THREADS, seeds=list(skew.SEEDS), quick=args.quick))
    log(f"done in {time.time() - t0:.0f}s; verdict: {json.loads((results / 'verdict.json').read_text())['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
