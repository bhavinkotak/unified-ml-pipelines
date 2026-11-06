# REVIEW: unified-ml-pipelines (v1 -> v2)

Hostile-reviewer grill of v1 (Zenodo 10.5281/zenodo.18445586), done before any v2 experiment was run.

## Round 1: Frontier (assumptions, prerequisites, baselines)

- Hidden assumption: a feature store makes offline and online features identical. True only if both paths execute one definition and the online store answers every lookup; timeouts, fallbacks, and stale snapshots reintroduce skew.
- Hidden assumption: skew always hurts accuracy. Not shown; numeric precision skew may change values but not predictions. The metric must separate "values differ" from "decisions differ".
- Hidden assumption: labels and aggregates are available at request time. v2 assumes zero label delay and states it as a threat.
- Prerequisites: v1 needs Kubernetes, Redis, S3, Airflow, Feast, enterprise data. None is reproducible. v2 uses three public OpenML tables, scikit-learn, one CPU.
- Baselines a reviewer demands: (a) dual implementation with documented divergences; (b) the same dual code with divergences switched off (must equal the single path, a correctness control); (c) per-source ablation; (d) AUC-only CI gate vs gate with skew replay.
- Unavailable baselines: Feast/Tecton/TFX Transform in a real cluster; real production incident logs; organisational outcomes (lead time, headcount). Recorded as threats.

## Round 2: Breakpoints

| Breakpoint | v2 handling |
|---|---|
| Scale x100 (entities, QPS) | Not tested (single-process simulation). Threat. |
| Missing data | Tested: MCAR injection at the base rate and at 4x the base rate. |
| Online-store / API timeout | Tested: single path with injected lookup timeouts and documented fallback defaults. |
| Unknown categories, empty groups | Tested in unit tests and covered by covariate-shift scenario. |
| Distribution shift | Tested: arrival ordered by a covariate (age) with noise. |
| Adversarial inputs | Not tested. Threat. |
| Label delay for target aggregates | Not tested (zero delay assumed). Threat. |
| Library-version change | Simulated only (implicit default change). Labelled as simulation. |

## Round 3: Defensibility

- Primary metric: online ROC AUC difference (single minus dual). Request skew rate is a proxy; it is reported but is not the primary endpoint.
- Falsifier: 95% CI of the AUC difference below 0.005 in at least 2 of 3 datasets (preregistration.json).
- Measurement bias: effect size depends on the injected divergence magnitudes chosen by the author. Mitigation: each source is fixed in advance, documented, and ablated separately. The single path zero-skew result is true by construction and is reported as a sanity check, not as evidence.
- Small-n: credit-g has 300 serving requests per seed; CIs are wide. Five seeds per dataset; t-based CI with df = 4.

## v1 claims: disposition

| v1 claim | Status | v2 action |
|---|---|---|
| "reduce model-to-production time by 70%" | No data | Removed |
| Netflix table (skew incidents -90%, MTTR -83%, cost -20%, 2x productivity, +7% accuracy) | Fabricated / unattributable | Removed |
| Bank and healthcare case studies (audit time, CFPB findings, etc.) | No data | Removed |
| "18% AUC degradation (0.94 -> 0.77)", "12 divergence points" | No data | Removed; replaced by measured offline-online AUC gaps |
| TKDE draft: 47-incident taxonomy with shares 38/24/19/12/7 | No data | Removed; divergence sources kept as qualitative categories and ablated |
| "60-75% overhead reduction", "4-5x velocity", "AUC gap 0.14 -> 0.02" | No data | Removed |
| "10-20% degradation", "MTTD 3-7 days", "40-60% of incidents", "2-3x headcount" | No data | Removed |
| "Prevents 85%+ of skew-related incidents" | No data | Reframed: measured catch counts of a simulated gate |
| "Feature stores eliminate skew" | Overclaim | Reframed: measured zero skew without timeouts; nonzero with timeouts |
| "Empirical performance data" for centralized vs federated stores | No data | Removed; architecture discussion kept qualitative |
| Refs: tech360us, Databricks/C3/Aerospike blogs, Netflix blog, TinyML book | Non-scholarly / unrelated | Removed |
| Refs: Sculley 2015, Breck 2019 (MLSys), MLflow 2018 | No DOI/arXiv record | Dropped (checker requires DOI/arXiv) |
| Ratner 2020 Snorkel cited for pipelines | Off-topic | Removed |
| Kubernetes/GitOps reference architecture, lineage design, patterns | Conceptual, valid | Kept, shortened, framed as design, not as measured |

## v1 scores (C1-C11)

| Criterion | v1 | Note |
|---|---|---|
| C1 filler/hype | Fail | "comprehensive", "strategic imperative", em-dashes throughout |
| C2 citations | Fail | blogs, no DOIs, numbers attributed to unverifiable sources |
| C3 formal model | Fail (md) / Partial (TKDE) | TKDE has skew equations; md has none |
| C4 outcome handling | Fail | no timeout/missing behaviour for store, registry, gate |
| C5 measurements | Fail | no numerators/denominators, no seeds, no CI |
| C6 consistency | Fail | md says 18% drop, TKDE says 17 pp; 70% vs 4-5x |
| C7 refs to figs/tabs | Partial | |
| C8 claims within evidence | Fail | case studies presented as measured |
| C9 numeric grounding | Fail | almost no number traceable |
| C10 threats | Fail | none |
| C11 preregistration | Fail | none |

Dimensions (1-10): Soundness 2, Novelty 3, Clarity 5, Reproducibility 1. Weighted (0.6 S + 0.4 N) = 2.4. Reject.

## v2 outcome (after the preregistered run)

- Verdict (results/verdict.json): SUPPORTED, 3 of 3 datasets. Delta online AUC: adult 0.086 [0.063, 0.109], bank-marketing 0.222 [0.171, 0.274], credit-g 0.139 [0.072, 0.205].
- Attribution: encoding order dominates; missing default second on adult; precision and stale window within noise; PIT leakage slightly negative (low-cardinality keys). Reported as is.
- Proxy check confirmed: request skew rate >= 0.96 for every source, harmless ones included.
- Breakpoints tested: store timeouts (skew equals degraded requests; AUC loss <= 0.0022), 4x missingness (delta grows on adult), age-ordered shift (delta stays positive).
- Rerun study: pinned 15/15 identical (also across 1 vs 2 threads); unpinned 0/15, explained by resolved lineage 15/15, explicit 0/15; order-sensitive hash gives 15 false alarms.
- Gate (simulation): harmful blocked 35/67 (AUC-only) vs 65/67 (AUC+skew); harmless blocked 10/98 vs 53/98.
- Deviations from plan: cell mismatch rate added as a descriptive metric; first full run stopped in the rerun stage (OpenMP stalls on a shared machine) and restarted; skew results byte-identical between the two runs.

## Residual reviewer concerns (recorded in Threats to Validity)

- Divergence magnitudes are author-chosen; effect sizes are conditional on them.
- Simulated time and simulated store; no networked feature store, no latency, no scale test, no adversarial inputs.
- One model family with ordinal category codes; one-hot models would react differently to encoding order.
- Low-cardinality entity keys, so leakage harm is understated.
- Gate harm label uses a small post-gate window on credit-g; a disagreement-based gate was not evaluated.

## v2 scores

| Criterion | v2 | Note |
|---|---|---|
| C1 filler/hype | Pass | desk check: no phrases, no hype, no em-dashes |
| C2 citations | Pass | 25 cited records verified against Crossref/arXiv; 4 URL-only software/data entries carry no numbers |
| C3 formal model | Pass | 7 labeled Methodology equations, all referenced |
| C4 outcome handling | Pass | outcome table (tab:outcomes): store, definition, gate, registry, download; implemented and unit-tested |
| C5 measurements | Pass | numerators/denominators, 5 seeds, 95% t CIs, per-source ablation |
| C6 consistency | Pass | tables generated from results/ by code; text numbers checked by desk check |
| C7 refs to figs/tabs | Pass | every label referenced |
| C8 claims within evidence | Pass | simulations labelled; generalisation limits in Threats |
| C9 numeric grounding | Pass | desk check numeric grounding passes |
| C10 threats | Pass | breakpoints not tested and unavailable baselines listed |
| C11 preregistration | Pass | preregistration.json before full run; verdict.json computed by code |

Dimensions (1-10): Soundness 7, Novelty 6, Clarity 8, Reproducibility 9.
- Weighted = 0.6 x 7 + 0.4 x 6 = 6.6, below the 7.0 accept line. The effect is real but partly by construction (author-injected divergences), and novelty is modest (a controlled measurement of a known failure mode). Honest assessment: a sound, reproducible preprint and replication-style study; not yet a top-venue acceptance.

## Final desk check

```
Desk check: .../unified-ml-pipelines/paper/main.tex  sections=['Introduction', 'Related Work', 'Methodology', 'Evaluation', 'Threats to Validity', 'Conclusion', 'Acknowledgment']  abstract_words=235  method_eqs=7  numbers_checked=364
PASS
```
All 25 DOI/arXiv references returned "verify <key>: ok". pytest: 15 passed. LaTeX: 8 pages, no undefined references, no overfull boxes.
