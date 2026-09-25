# `results/` layout

Three trees, one rule each. The rule exists because the paper's numbers must stay
reproducible: a new experiment never writes where a published one already lives.

```
results/
  full/            frozen reference of the paper: never written by a new run
  runs/<tag>/      sandbox of a new run (tag = channel/date, chosen by the runner)
  archive/         superseded or orphan artifacts, kept for provenance
  RESULTS_GENERATION.md
```

## What is a frozen reference

These files back the numbers and the figures of the submitted paper. Do not
regenerate them in place, do not move them, do not delete them:

| Path | Role |
|---|---|
| `full/ber_vs_snr/<scenario>/<arch>/{metrics.csv,best_model.keras}` | communication and sensing benchmark; its four `k3_doppler_full` checkpoints are the frozen receivers used by frequency agility, channel generalization and latency |
| `full/ber_vs_snr/operating_region_table.csv` | pooled operating-region table |
| `full/jamming/<arch>/{best_model.keras,metrics.csv}` | checkpoints of the jamming grid |
| `full/jamming/<arch>/jamming/{jamming_results_*.csv,jamming_realizations_*.csv,baseline_metrics.csv,metrics.csv}` | Monte Carlo jamming grid, mean/std/min/max plus per-realization BER |
| `full/jamming_interpretability/<arch>/{conditions,conditions_realizations,per_snr}.csv` | layer-retention probe (Fig. `layers`) and the clean-trained side of Fig. `jamaware` |
| `full/jamming_interpretability/jamming_aware_training/qkv/**` | consolidated jamming-aware QKV receiver, three passes with provenance in `run_metadata.json` |
| `full/frequency_agility/<arch>/{frequency_agility_vs_jsr.csv,frequency_agility_vs_dwell.csv,frequency_agility_realizations.csv,run_metadata.json}` | frequency-agility sweep, `hold_mode: per_slot` |
| `full/classical_receivers/**`, `full/final_report/**` | equations-only baselines; SWaP-C, latency and aggregated report |
| `full/diagnostics/jamming_artifact/**` | CW single-realization artefact versus Monte Carlo |
| `full/logs/RUN_INFO.txt` | provenance manifest of the 2026-09-24 Colab run. It is not produced by the code: never edit or delete it |

## What lives elsewhere

Figures are **not** here. They are vector PDFs in `figures/`, produced from the
CSVs by the scripts in `scripts/figures/`, and the paper reads them through
`latex/paper.tex` (`\graphicspath{{../figures/}}`). No PNG preview is written
anywhere.

## Adding a run

1. Pick a run tag, e.g. `v2_canaleB`.
2. Run with `--output-dir results/runs/<tag>`; the runner refuses to write inside
   `results/full`.
3. The run writes its own `logs/` (config snapshot), `RUN_INFO_<date>.txt` and
   `timings.json`. New manifests are separate files: `full/logs/RUN_INFO.txt` is
   historical and stays as it is.
4. Promote to `full/` only after reviewing the CSV diff, and record what was
   replaced in the run's `RUN_INFO_<date>.txt`.
