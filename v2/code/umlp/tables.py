"""Render LaTeX table bodies from results/ and splice them into paper/main.tex between marker comments."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .features import SOURCES
from .gate import CANDIDATES
from .lineage import REGIMES

DATASETS = ("adult", "bank-marketing", "credit-g")
SRC_LABEL = {"missing_default": "missing default", "precision": "precision", "encoding_order": "encoding order",
             "stale_window": "stale window", "pit_leakage": "PIT leakage", "all": "all five"}


def _f(x: float, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _ms(c: dict, d: int = 3) -> str:
    return f"{c['mean']:.{d}f}$\\pm${c['std']:.{d}f}"


def _ci(c: dict, d: int = 3) -> str:
    return f"[{c['ci_low']:.{d}f}, {c['ci_high']:.{d}f}]"


def datasets(r: Path) -> str:
    s = json.loads((r / "dataset_summary.json").read_text())
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Dataset & Rows & Num. & Cat. & Pos. rate & Train & Requests \\\\", "\\midrule"]
    for d in DATASETS:
        x = s[d]
        lines.append(f"{d} & {x['n_rows']} & {x['n_numeric']} & {x['n_categorical']} & {_f(x['positive_rate'])} & "
                     f"{x['n_train']} & {x['n_serving_requests']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main_result(r: Path) -> str:
    s = json.loads((r / "skew_summary.json").read_text())
    dl = s["delta_auc_online_single_minus_path"]
    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
             "Dataset & Path & Offline AUC & Online AUC & Gap \\\\", "\\midrule"]
    for d in DATASETS:
        for p, name in (("single", "single"), ("dual_all", "dual")):
            x = s[d]["base"][p]
            lines.append(f"{d if p == 'single' else ''} & {name} & {_ms(x['auc_offline'])} & {_ms(x['auc_online'])} & "
                         f"{_f(x['auc_gap']['mean'])} \\\\")
        c = dl[d]["base"]["dual_all"]
        lines.append(f" & $\\Delta$ & \\multicolumn{{3}}{{c}}{{{_f(c['mean'])} {_ci(c)}}} \\\\")
        if d != DATASETS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def ablation(r: Path) -> str:
    s = json.loads((r / "skew_summary.json").read_text())
    dl = s["delta_auc_online_single_minus_path"]
    lines = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "Dataset & Source & Req. skew & Cell mism. & Disagree & $\\Delta$AUC \\\\", "\\midrule"]
    for d in DATASETS:
        for i, src in enumerate([*SOURCES, "all"]):
            x = s[d]["base"][f"dual_{src}"]
            lines.append(f"{d if i == 0 else ''} & {SRC_LABEL[src]} & {_f(x['skew_rate']['mean'])} & "
                         f"{_f(x['cell_mismatch_rate']['mean'])} & {_f(x['disagreement_rate']['mean'])} & "
                         f"{_f(dl[d]['base'][f'dual_{src}']['mean'])} \\\\")
        if d != DATASETS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def breakpoints(r: Path) -> str:
    s = json.loads((r / "skew_summary.json").read_text())
    dl = s["delta_auc_online_single_minus_path"]
    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
             "Dataset & Scenario & Single & Dual & $\\Delta$ [95\\% CI] \\\\", "\\midrule"]
    for d in DATASETS:
        for i, (scen, name) in enumerate((("base", "base"), ("missing_high", "4x missing"), ("shift", "shift"))):
            a, b = s[d][scen]["single"]["auc_online"], s[d][scen]["dual_all"]["auc_online"]
            c = dl[d][scen]["dual_all"]
            lines.append(f"{d if i == 0 else ''} & {name} & {_f(a['mean'])} & {_f(b['mean'])} & "
                         f"{_f(c['mean'])} {_ci(c)} \\\\")
        if d != DATASETS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def repro(r: Path) -> str:
    s = json.loads((r / "repro_summary.json").read_text())
    names = {"pinned": "pinned", "pinned_threads": "pinned, 1 vs 2 threads", "unpinned_seed": "unpinned seed",
             "data_order": "row order varies", "implicit_default": "default change (sim.)"}
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Regime & Identical & Differ & Expl. (explicit) & Expl. (resolved) \\\\", "\\midrule"]
    for g in REGIMES:
        x = s[g]
        lines.append(f"{names[g]} & {x['n_identical_pred']}/{x['n_reruns']} & {x['n_nonidentical']} & "
                     f"{x['n_nonidentical_explained_explicit']}/{x['n_nonidentical']} & "
                     f"{x['n_nonidentical_explained_resolved']}/{x['n_nonidentical']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def gate(r: Path) -> str:
    s = json.loads((r / "gate_summary.json").read_text())["per_candidate"]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Candidate & Harmful & AUC-only & AUC+skew & Skew rate \\\\", "\\midrule"]
    for c in CANDIDATES:
        x = s[c.name]
        name = c.name.replace("_", " ").replace("pit", "PIT")
        lines.append(f"{name} & {x['n_harmful']}/{x['n']} & {x['n_blocked_auc_only']}/{x['n']} & "
                     f"{x['n_blocked_auc_skew']}/{x['n']} & {_f(x['mean_skew_rate_gate'])} \\\\")
        if c.name == "row_shuffle":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


RENDERERS = {"tab_datasets": datasets, "tab_main": main_result, "tab_ablation": ablation,
             "tab_breakpoints": breakpoints, "tab_repro": repro, "tab_gate": gate}


def inject(results: Path, tex_path: Path) -> list[str]:
    """Replace the text between '%BEGIN:<name>' and '%END:<name>' with the rendered table body."""
    if not tex_path.exists():
        return []
    tex = tex_path.read_text()
    done = []
    for name, fn in RENDERERS.items():
        pat = re.compile(rf"(%BEGIN:{name}\n).*?(%END:{name})", re.DOTALL)
        if pat.search(tex):
            body = fn(results)
            tex = pat.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), tex)
            done.append(name)
    tex_path.write_text(tex)
    return done
