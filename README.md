# Dual-Head Ultra-CAN: ISAC Receiver for the Internet of Drones

Companion code for the IEEE IoT-J paper **"Dual-Head Ultra-CAN: a dual-head
integrated sensing and communications receiver for chaos-based drone links"**.

This repository reproduces every figure and table of the paper evaluation:
communication BER curves (full range and operating region), delay (ranging)
estimation, classical-receiver equations-only sweeps, jamming robustness,
layer-wise interpretability, jamming-aware training, and the final SWaP-C and
latency report.

## Layout

```text
configs/            base_config.yaml + experiments.yaml (single source of truth)
src/data/           dataset generation and data loading
src/models/         receivers: Ultra-CAN, Ultra-CAN-QKV, LSTM-OFDM-DCSK, MC-DLCSK
src/training/       multi-task losses and the Trainer
src/evaluation/     BER/sensing/jamming evaluation and the SWaP-C table
src/experiments/    runner.py (single entry point), pipeline helpers, final report
scripts/            figure/table producers and benchmarks
notebooks/          Colab notebooks (setup, benchmark, jamming, final report)
results/full/       generated outputs (models ignored by git, curated CSVs kept)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the experiments

Experiments are named after the action they perform:

| Name | Paper section | Main outputs |
|---|---|---|
| `ber_vs_snr` | Communication & sensing benchmark | `metrics.csv` per (scenario, architecture) |
| `classical_receivers` | Classical receivers (no figures) | `ber_vs_k.csv`, `ber_vs_doppler.csv` |
| `jamming` | Jamming robustness | BER-vs-JSR tables per architecture |
| `jamming_interpretability` | Interpretability probe | `conditions.csv`, `per_snr.csv` |
| `final_report` | SWaP-C + conclusions | `weight_table.tex`, `final_report.md` |

```bash
# Full paper run (slow; GPU recommended)
python src/experiments/runner.py --experiments all --mode full

# A single experiment
python src/experiments/runner.py --experiments ber_vs_snr --mode full

# Smoke run on CPU
python src/experiments/runner.py --experiments ber_vs_snr --mode fast

# Reuse the already-generated datasets
python src/experiments/runner.py --experiments all --mode full --no-regen
```

## Reproduce the figures and tables

Run these after the corresponding experiments (see also the notebooks):

```bash
python scripts/make_fig_architecture.py             # architecture.pdf
python scripts/make_fig_ber_full_linear.py          # ber_full_linear.pdf + blind receiver tables
python scripts/make_fig_ber_region_zoom.py          # ber_region_zoom.pdf
python scripts/make_fig_sensing_delay.py            # sensing_delay_single.pdf
python scripts/make_table_operating_region.py       # operating-region table (pooled BER, min SNR @1e-4)
python scripts/make_fig_layer_retention.py          # jamming_layer_retention.pdf
python scripts/run_jam_aware_qkv.py                 # jamming-aware training (long)
python scripts/make_fig_jamming_aware_control.py    # jamming_aware_control.pdf
python scripts/latency_bench.py                     # single-burst latency (CPU)
```

All figure scripts resolve their inputs under `results/` and write PDFs into
`figures/`; they never rely on absolute paths.

## Operating region

The paper reports operating-region metrics for **SNR >= 5 dB** (pooled BER and
minimum SNR to reach BER <= 1e-4). The table script and the final report use
this convention.

## Notebooks

`notebooks/` contains the Colab reproduction flow:
`00_Setup_and_Datasets`, `01_BER_and_Sensing_Benchmark`,
`02_Jamming_and_Interpretability`, `03_Final_Report`. Each notebook issues
short commands and regenerates the corresponding figures/tables.

## Tests

```bash
python -m pytest tests -v
```
