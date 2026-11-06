"""Figures for the paper, generated only from files in results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .features import SOURCES  # noqa: E402

COLORS = {"adult": "#1b9e77", "bank-marketing": "#d95f02", "credit-g": "#7570b3"}
LABELS = {"missing_default": "missing\ndefault", "precision": "preci-\nsion", "encoding_order": "enc.\norder",
          "stale_window": "stale\nwindow", "pit_leakage": "PIT\nleak.", "all": "all\nfive"}


def _style() -> None:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "figure.dpi": 150})


def _err(d: dict) -> float:
    return 0.0 if not np.isfinite(d["ci_high"]) else d["ci_high"] - d["mean"]


def ablation(summary: dict, out: Path) -> None:
    datasets = [k for k in COLORS if k in summary]
    paths = [*(f"dual_{s}" for s in SOURCES), "dual_all"]
    names = [*SOURCES, "all"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1))
    x = np.arange(len(paths))
    w = 0.8 / len(datasets)
    panels = (("cell_mismatch_rate", "(a) feature-cell mismatch rate"),
              ("disagreement_rate", "(b) prediction disagreement rate"),
              ("delta", "(c) online AUC loss vs single path"))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for i, dsn in enumerate(datasets):
            if metric == "delta":
                ds = [summary["delta_auc_online_single_minus_path"][dsn]["base"][p] for p in paths]
            else:
                ds = [summary[dsn]["base"][p][metric] for p in paths]
            ax.bar(x + (i - (len(datasets) - 1) / 2) * w, [d["mean"] for d in ds], w,
                   yerr=[_err(d) for d in ds], color=COLORS[dsn], label=dsn, capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(x, [LABELS[n] for n in names], fontsize=6)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].legend(frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out)
    plt.close(fig)


def offline_online(summary: dict, out: Path) -> None:
    datasets = [k for k in COLORS if k in summary]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    bars = (("single", "auc_offline", "single, offline"), ("single", "auc_online", "single, online"),
            ("dual_all", "auc_offline", "dual, offline"), ("dual_all", "auc_online", "dual, online"))
    x = np.arange(len(datasets))
    w = 0.2
    shades = ("#4d4d4d", "#a6a6a6", "#2166ac", "#92c5de")
    for i, ((path, m, lab), col) in enumerate(zip(bars, shades, strict=True)):
        ds = [summary[d]["base"][path][m] for d in datasets]
        ax.bar(x + (i - 1.5) * w, [d["mean"] for d in ds], w, yerr=[_err(d) for d in ds], color=col, label=lab,
               capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(x, datasets)
    ax.set_ylabel("ROC AUC (held-out requests)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6,
              handlelength=1.2, columnspacing=0.8)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def timeouts(summary: dict, rates: tuple[float, ...], out: Path) -> None:
    datasets = [k for k in COLORS if k in summary]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    xs = [0.0, *rates]
    for dsn in datasets:
        base = summary[dsn]["base"]
        skew = [base["single"]["skew_rate"]["mean"], *(base[f"single_timeout_{r}"]["skew_rate"]["mean"] for r in rates)]
        dl = [0.0, *(summary["delta_auc_online_single_minus_path"][dsn]["base"][f"single_timeout_{r}"]["mean"]
                     for r in rates)]
        axes[0].plot(xs, skew, marker="o", ms=2.5, color=COLORS[dsn], label=dsn, lw=0.9)
        axes[1].plot(xs, dl, marker="o", ms=2.5, color=COLORS[dsn], lw=0.9)
    axes[0].set_title("(a) request skew rate")
    axes[1].set_title("(b) online AUC loss")
    for ax in axes:
        ax.set_xlabel("store timeout probability")
        ax.grid(lw=0.3, alpha=0.5)
    axes[0].legend(frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def make_all(results: Path, figdir: Path, rates: tuple[float, ...]) -> None:
    _style()
    figdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results / "skew_summary.json").read_text())
    ablation(summary, figdir / "fig_ablation.pdf")
    offline_online(summary, figdir / "fig_auc.pdf")
    timeouts(summary, rates, figdir / "fig_timeout.pdf")
