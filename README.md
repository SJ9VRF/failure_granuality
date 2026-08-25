# Failure Granularity Tax — Complete Submission Package

**Paper:** *The Failure Granularity Tax: How Detailed Should Agent Failures Be for Self-Evolving Training?*

This folder contains the complete blind manuscript, LaTeX source, bibliography, executable experiment code, raw and summarized results, statistical analyses, figures, reviewer simulation, reproducibility instructions, and submission checklist.

## Start here

1. [`PAPER_BLIND.md`](PAPER_BLIND.md) — complete blind paper in Markdown, including appendix.
2. [`PAPER_BLIND.tex`](PAPER_BLIND.tex) — LaTeX manuscript source for submission.
3. [`AUTO_RESULTS.md`](AUTO_RESULTS.md) — concise paper-facing results generated from the CSV outputs.
4. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — exact reproduction instructions.
5. [`REVIEWER_SIMULATION.md`](REVIEWER_SIMULATION.md) — adversarial reviewer-style critique and scores.
6. [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) — final pre-submission checklist.
7. [`run_all.sh`](run_all.sh) — top-level reproduction entry point.

---

# Manuscript and paper metadata

- [`PAPER_BLIND.md`](PAPER_BLIND.md) — full anonymized manuscript. Contains abstract, introduction, related work, formulation, theory, experimental protocol, results, limitations, discussion, and appendix.
- [`PAPER_BLIND.tex`](PAPER_BLIND.tex) — LaTeX version of the complete blind manuscript.
- [`references.bib`](references.bib) — BibTeX bibliography used by the LaTeX manuscript.
- [`AUTO_RESULTS.md`](AUTO_RESULTS.md) — automatically generated summary of headline numerical results; intended to prevent hand-copied number mismatches between tables/text and CSV outputs.
- [`MANIFEST.json`](MANIFEST.json) — package manifest and file hashes/metadata for integrity checking.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — reproduction guide, dependencies, commands, and expected outputs.
- [`REVIEWER_SIMULATION.md`](REVIEWER_SIMULATION.md) — simulated NeurIPS/ICLR-style review covering novelty, correctness, empirical strength, clarity, significance, reproducibility, and remaining limitations.
- [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) — checklist for anonymity, citations, results consistency, appendix completeness, and packaging.

---

# Experiment code

All experiment scripts are under [`code/`](code/).

- [`code/fgt8_lib.py`](code/fgt8_lib.py) — core GranularityBench-8 simulator library. Defines the 8-failure hierarchy, task generation, curriculum masks, update/transfer dynamics, evaluation metrics, support metrics, and seeded simulation utilities.
- [`code/run_fgt8_main.py`](code/run_fgt8_main.py) — runs the main heterogeneous benchmark comparison across curriculum strategies and representation/update regimes.
- [`code/run_fgt8_tax.py`](code/run_fgt8_tax.py) — runs the direct 100-seed Failure Granularity Tax causal experiment comparing Global, Capability, Family, and Failure conditioning.
- [`code/run_fgt8_phase_batch.py`](code/run_fgt8_phase_batch.py) — runs phase-diagram batches over budget, diagnostic noise, transfer, target prevalence, and granularity.
- [`code/run_fgt8_stress_axes.py`](code/run_fgt8_stress_axes.py) — reviewer stress tests for entanglement, horizon, and transition stochasticity/noise.
- [`code/run_fgt8_support_alt.py`](code/run_fgt8_support_alt.py) — alternative support metrics: effective support, coverage, and Jensen–Shannon divergence, used to verify that the main phenomenon is not an artifact of normalized entropy alone.
- [`code/fgt8_stats_main.py`](code/fgt8_stats_main.py) — main statistical tests and corrected pairwise comparisons.
- [`code/analyze_fgt8.py`](code/analyze_fgt8.py) — primary aggregation and figure-generation logic for main/tax/phase experiments.
- [`code/analyze_fgt8_extra.py`](code/analyze_fgt8_extra.py) — additional stress-test and alternative-support analyses and plots.
- [`code/generate_auto_results.py`](code/generate_auto_results.py) — generates `AUTO_RESULTS.md` directly from CSV outputs.

`code/__pycache__/` is Python runtime cache only and is **not required** for reproduction.

---

# Results: main experiment

- [`results/fgt8_main.csv`](results/fgt8_main.csv) — raw row-level output of the main experiment. This is the canonical source for the headline multi-method comparisons.
- [`results/fgt8_main_summary.csv`](results/fgt8_main_summary.csv) — aggregated main results by method / regime.
- [`results/fgt8_main_overall.csv`](results/fgt8_main_overall.csv) — compact overall method averages used in the main-results table.
- [`results/fgt8_main_auc.csv`](results/fgt8_main_auc.csv) — learning-curve area-under-curve results measuring sample efficiency.
- [`results/fgt8_main_stats.csv`](results/fgt8_main_stats.csv) — inferential statistics for main method comparisons, including corrected significance tests/effect summaries.

**What this experiment answers:** Whether failure-conditioned curricula outperform broad/uniform/difficulty/random-granularity baselines in a heterogeneous controlled setting. It also preserves the important negative result that no failure-conditioned method universally dominates.

---

# Results: direct Failure Granularity Tax experiment

- [`results/fgt8_tax_100.csv`](results/fgt8_tax_100.csv) — raw 100-seed causal experiment for Global vs Capability vs Family vs Failure conditioning.
- [`results/fgt8_tax_summary.csv`](results/fgt8_tax_summary.csv) — aggregated means and uncertainty for repair, held-out transfer, external transfer, and support.
- [`results/fgt8_tax_stats.csv`](results/fgt8_tax_stats.csv) — bootstrap/statistical contrasts for the direct tax experiment.

**What this experiment answers:** The core causal claim of the paper. Increasing diagnostic specificity increases local/targeted repair but contracts experience support and can substantially reduce broad held-out and external transfer.

---

# Results: phase diagram / optimal granularity map

The phase experiments vary **diagnostic noise** (`n00`, `n02`, `n04`) and **training budget** (`b40`, `b80`, `b160`) while sweeping the other regime variables.

Raw batch files:

- [`results/fgt8_phase_n00_b40.csv`](results/fgt8_phase_n00_b40.csv) — noise 0.0, budget 40.
- [`results/fgt8_phase_n00_b80.csv`](results/fgt8_phase_n00_b80.csv) — noise 0.0, budget 80.
- [`results/fgt8_phase_n00_b160.csv`](results/fgt8_phase_n00_b160.csv) — noise 0.0, budget 160.
- [`results/fgt8_phase_n02_b40.csv`](results/fgt8_phase_n02_b40.csv) — noise 0.2, budget 40.
- [`results/fgt8_phase_n02_b80.csv`](results/fgt8_phase_n02_b80.csv) — noise 0.2, budget 80.
- [`results/fgt8_phase_n02_b160.csv`](results/fgt8_phase_n02_b160.csv) — noise 0.2, budget 160.
- [`results/fgt8_phase_n04_b40.csv`](results/fgt8_phase_n04_b40.csv) — noise 0.4, budget 40.
- [`results/fgt8_phase_n04_b80.csv`](results/fgt8_phase_n04_b80.csv) — noise 0.4, budget 80.
- [`results/fgt8_phase_n04_b160.csv`](results/fgt8_phase_n04_b160.csv) — noise 0.4, budget 160.

Combined/derived files:

- [`results/fgt8_phase_raw.csv`](results/fgt8_phase_raw.csv) — concatenated raw phase-grid results across all batches.
- [`results/fgt8_phase_summary.csv`](results/fgt8_phase_summary.csv) — aggregated phase-grid metrics.
- [`results/fgt8_phase_winners.csv`](results/fgt8_phase_winners.csv) — best granularity for each phase-grid cell.
- [`results/fgt8_phase_win_fractions.csv`](results/fgt8_phase_win_fractions.csv) — fraction of cells won by each granularity under each budget/noise regime.

**What this experiment answers:** How the optimal granularity changes with training budget, diagnostic noise, transfer structure, and deployment distribution. The central observed pattern is a budget-dependent transition from finer to intermediate to broader conditioning.

---

# Results: reviewer stress tests

- [`results/fgt8_stress_axes.csv`](results/fgt8_stress_axes.csv) — raw stress-test results across entanglement, horizon, and transition stochasticity/noise.
- [`results/fgt8_stress_summary.csv`](results/fgt8_stress_summary.csv) — aggregated stress-test outcomes.

**What these experiments answer:** Whether the Failure Granularity Tax persists when tasks become more compositional, longer-horizon, or transitions become less deterministic.

---

# Results: alternative support metrics

- [`results/fgt8_support_alt.csv`](results/fgt8_support_alt.csv) — raw per-run alternative support measures.
- [`results/fgt8_support_alt_summary.csv`](results/fgt8_support_alt_summary.csv) — summary of normalized entropy, effective support, coverage, and JS divergence.
- [`results/fgt8_support_alt_correlations.csv`](results/fgt8_support_alt_correlations.csv) — correlations between alternative support measures and broad/transfer performance.

**What this experiment answers:** Whether the support-collapse finding depends on a single entropy definition. It does not: effective support, coverage, and Jensen–Shannon support divergence show the same qualitative relationship.

---

# Results: power analysis

- [`results/fgt8_power_analysis.csv`](results/fgt8_power_analysis.csv) — empirical/statistical power calculations for detectable effect sizes and seed counts.

**What this file is for:** Justifies the use of large seed counts and helps distinguish truly small effects from underpowered null results.

---

# Figures

All paper figures are under [`figures/`](figures/).

- [`figures/fig_h8_tax.png`](figures/fig_h8_tax.png) — **core figure**: local repair vs support/generalization as granularity becomes finer.
- [`figures/fig_h8_phase.png`](figures/fig_h8_phase.png) — **phase diagram**: which granularity is optimal across regime variables.
- [`figures/fig_h8_main.png`](figures/fig_h8_main.png) — headline main-experiment comparison across methods.
- [`figures/fig_h8_local_global.png`](figures/fig_h8_local_global.png) — local targeted improvement versus broad/global generalization trade-off.
- [`figures/fig_h8_entangle.png`](figures/fig_h8_entangle.png) — robustness to failure/task entanglement.
- [`figures/fig_h8_horizon.png`](figures/fig_h8_horizon.png) — effect of task horizon on the preferred granularity and transfer.
- [`figures/fig_h8_transition_noise.png`](figures/fig_h8_transition_noise.png) — robustness to stochastic/noisy environment transitions.
- [`figures/fig_h8_support_alt.png`](figures/fig_h8_support_alt.png) — alternative experience-support metrics and their relationship to generalization.

---

# Reproduction

From the package root:

```bash
bash run_all.sh
```

For details and expected outputs, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

The experiments are designed to run without external model checkpoints, inference APIs, proprietary datasets, or GPU-only dependencies. The manuscript scope is therefore a **controlled causal study of failure-conditioned self-evolving agent learning**, not a claim of frontier-LLM benchmark SOTA.

---

# Suggested reading order

If you want to understand the project quickly:

1. `PAPER_BLIND.md`
2. `figures/fig_h8_tax.png`
3. `figures/fig_h8_phase.png`
4. `AUTO_RESULTS.md`
5. `results/fgt8_tax_summary.csv`
6. `results/fgt8_phase_win_fractions.csv`
7. `REVIEWER_SIMULATION.md`
8. `REPRODUCIBILITY.md`

If you want to audit every result:

1. `results/fgt8_main.csv`
2. `results/fgt8_tax_100.csv`
3. `results/fgt8_phase_raw.csv`
4. `results/fgt8_stress_axes.csv`
5. `results/fgt8_support_alt.csv`
6. `code/fgt8_lib.py`
7. corresponding runner scripts
8. analysis/statistics scripts

---

# Scope and scientific caveat

The final package intentionally does **not** claim Qwen/Llama/GRPO or frontier-LLM benchmark improvements. All principal claims in the paper are supported by the included controlled CPU-executable experiments. External validation on a real LLM/tool-use benchmark remains future validation rather than fabricated evidence.
