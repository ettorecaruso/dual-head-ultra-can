# `scripts/` layout

Every script is an entry point run from the repository root; figures and tables
are never hard-coded, they are rendered from the CSVs in `results/`.

```
scripts/
  make_all.sh          regenerate every figure and table in dependency order
  style/               shared publication look (palette, axes, save())
  figures/             one script per figure of the paper -> figures/<stem>.pdf
  tables/              one script per LaTeX table -> results/.../<table>.tex
  diagnostics/         investigations of a specific artefact (not deliverables)
  tools/               long-running helpers: Monte Carlo reruns, latency, retraining
  data/                dataset generation wrappers
```

## Conventions

* `REPO` is `Path(__file__).resolve().parents[2]` in every entry script, because
  the scripts live one level below `scripts/`.
* The shared style is imported as `from style import figure_style as fs`; the
  package root (`scripts/`) is put on `sys.path` by the two lines above the
  import.
* `style/figure_style.save(fig, stem)` is the only writer for figures: it writes
  `figures/<stem>.pdf` and mirrors it into `results/figures/` when that directory
  exists. No PNG raster copy is produced.
* `scripts/data/*.sh` resolve the repository root from their own location, so they
  work from any working directory.

## Entry points

| Script | Output |
|---|---|
| `figures/make_fig_architecture.py` | `figures/architecture.pdf` |
| `figures/make_fig_ber_full_linear.py` | `figures/ber_full_linear.pdf` and the blind-receiver table of `results/full/ber_vs_snr/` |
| `figures/make_fig_ber_region_zoom.py` | `figures/ber_region_zoom*.pdf` |
| `figures/make_fig_sensing_delay.py` | `figures/sensing_delay_single.pdf` |
| `figures/make_fig_layer_retention.py` | `figures/jamming_layer_retention.pdf` |
| `figures/make_fig_jamming_aware_control.py` | `figures/jamming_aware_control.pdf` |
| `figures/make_fig_frequency_agility.py` | `figures/frequency_agility.pdf`, `figures/frequency_agility_gain.pdf` |
| `figures/make_fig_frequency_agility_reaction.py` | `figures/frequency_agility_reaction.pdf` |
| `figures/make_fig_channel_generalization.py` | `figures/channel_generalization.pdf` |
| `tables/make_table_operating_region.py` | `results/full/ber_vs_snr/operating_region_table.csv` |
| `tables/make_table_channel_generalization.py` | `generalization_table.{csv,tex}` next to the run in the argument |
| `diagnostics/diagnose_jamming_mc.py` | `results/full/diagnostics/jamming_artifact/**` |
| `tools/rerun_jamming_mc.py` | `results/full/jamming/<arch>/jamming/jamming_*.csv` (evaluation only, no retraining) |
| `tools/latency_bench.py` | `results/full/final_report/latency_results.json` |
| `tools/run_jam_aware_qkv.py` | retraining of the jamming-aware QKV receiver |
| `tools/write_run_info.py` | `RUN_INFO_<date>.txt` + `timings.json` inside a run directory |
| `tools/stage_reference_checkpoints.py` | symlinks of the frozen trees into a sandbox run root |
| `data/generate_dataset.sh`, `data/generate_all_datasets.sh` | `data/raw/<hash>/` |

Three producers accept a run directory as an optional argument, so the same script
serves the current results and an archived copy:
`tables/make_table_channel_generalization.py`,
`figures/make_fig_channel_generalization.py` and
`figures/make_fig_frequency_agility_reaction.py`.

Two scripts write inside `results/full`, which is the frozen tree of the paper:
`tables/make_table_operating_region.py` and `figures/make_fig_ber_full_linear.py`.
Run them deliberately, and only when their inputs are known to be current.
