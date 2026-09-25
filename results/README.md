# `results/` layout

One tree, one rule: the results of an experiment live in
`results/<mode>/<experiment>/` and the previous version of the same experiment is
never destroyed, it is archived.

```
results/
  full/                results of the `full` mode: the numbers the paper cites
  fast/                results of the `fast` smoke mode, when it is run
  archive/             previous versions and orphan artifacts, kept for provenance
  README.md            this file
  STATUS.md            which subtree is final, which one a run will supersede
  RESULTS_GENERATION.md  notes on the archived Colab run
```

## What is a reference

`full/ber_vs_snr/<scenario>/<arch>/best_model.keras` and
`full/jamming/<arch>/best_model.keras` are the frozen receivers: the validation
experiments load them, and nothing in the tree regenerates them in place. The
same holds for the CSV files the figures and tables are rendered from.

| Path | Role |
|---|---|
| `full/ber_vs_snr/**` | benchmark, `operating_region_table.csv`, the 12 frozen checkpoints |
| `full/jamming/<arch>/{best_model.keras,metrics.csv}` and `jamming/**` | jamming checkpoints and the Monte Carlo grid (mean/std/min/max plus per-realization BER) |
| `full/jamming_interpretability/**` | layer-retention probe and the consolidated jamming-aware receiver |
| `full/frequency_agility/**` | frequency-agility sweep |
| `full/classical_receivers/**`, `full/final_report/**` | equations-only baselines; SWaP-C, latency |
| `full/diagnostics/jamming_artifact/**` | artefact investigation |
| `full/logs/RUN_INFO.txt` | provenance manifest of the 2026-09-24 Colab run. Not produced by the code: never edit or delete it |

Figures are **not** here: they are vector PDFs in `figures/`, produced from the
CSVs by `scripts/figures/` and read by `latex/paper.tex`
(`\graphicspath{{../figures/}}`). No PNG preview is written anywhere.

## Re-running an experiment

A run writes into `results/<mode>/<experiment>/`. If that directory already
contains results, the runner needs an explicit decision, so a cell can never
clobber the paper by accident:

| Flag | Effect |
|---|---|
| *(none)* | refuse to write, and say which flags are available |
| `--supersede` | move the previous directory to `results/archive/<experiment>_<timestamp>/`, then write the new results. Nothing is lost |
| `--resume` | keep what is there and skip the experiment |
| `--force` | overwrite in place, no archive: only for throwaway experiments |

`--dry-run` prints the same decision (channel, propagation diagnostics, checkpoint
resolution, target directory, whether it is populated) and writes nothing.

Every run also writes `<experiment>/logs/config_used.yaml` and, through
`scripts/tools/write_run_info.py`, a `RUN_INFO_<date>.txt` plus `timings.json`
inside its own directory. The historical `full/logs/RUN_INFO.txt` stays as it is.

