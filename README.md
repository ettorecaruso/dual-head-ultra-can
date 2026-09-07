# Dual-Head Ultra-CAN

Code, models, results and notebooks for the paper

**Dual-Head Ultra-CAN: A Lightweight Edge AI Architecture for Integrated Sensing and Chaos-Based Communications in Internet of Drones** (IEEE Internet of Things Journal, 2026).

The work extends the Compact Attention-Augmented Neural Decoder (Ultra-CAN) introduced in *Compact Attention-Augmented Neural Decoder for Chaos-Based Wireless Communications*, IEEE Communications Letters, 2026.

## Repository layout

```
configs/       link, training, evaluation and jamming parameters
src/           chaos maps, aerial channel, datasets, models, losses, training and evaluation
scripts/       experiment CLI and figure generation
notebooks/     end-to-end Colab-ready notebooks
results/       committed curves, tables and the paper figures
```

## Requirements

- Python 3.10+
- TensorFlow 2.13+ (Keras), NumPy, Matplotlib, pandas, PyYAML

```
pip install -r requirements.txt
```

## Quickstart

Build the four receivers and verify their footprint:

```
python scripts/run_experiments.py build
```

Generate the SWaP-C table:

```
python scripts/run_experiments.py swapc
```

Short CPU smoke run (one epoch on tiny sets):

```
python scripts/run_experiments.py smoke --model conv1d --samples 2000 --epochs 1
```

Run the full self-check (architecture counts, committed tables and curves, jam-aware numbers, one-epoch smoke training):

```
python scripts/self_check.py
```

Regenerate the figures from the committed curves:

```
python scripts/make_figures.py
```

The committed figures in `results/figures/` are the exact figures of the paper. `make_figures.py` reproduces them from the committed curves with the repository plot style; commit the regenerated PDFs only if you want to replace the canonical set.

## Models

| Receiver | Parameters | Float32 memory |
| --- | ---: | ---: |
| Ultra-CAN (Conv1D) | 26,597 | 103.9 kB |
| Ultra-CAN (QKV) | 43,172 | 168.6 kB |
| LSTM-OFDM-DCSK | 45,988 | 179.6 kB |
| MC-DLCSK | 44,756 | 174.8 kB |
| Classical (DCSK) | 0 | 0.0 kB |

## Reproducibility

Each architecture is trained with a fixed deterministic seed (conv1d=5, qkv=67, lstm=63, mc_dlsk=80). Testing follows the "at least 100 bit errors" criterion per SNR point with a cap of 2,000,000 symbols per SNR point. Reported results correspond to a single run per architecture; no averaging over multiple seeds is performed. The exact experimental logs behind every reported number are committed under `results/curves/`.

## Paper mapping

- Table I (link parameters): `configs/experiments.yaml`.
- Table II (pooled BER) and the operating-region figure: notebook `04`, `results/curves/ber_curves.csv`, `results/figures/operating_region_ber.pdf`.
- Delay estimation figure: notebook `05`, `results/figures/sensing_delay_single.pdf`.
- Jamming retention and jamming-aware figures: notebook `06`, `results/figures/activation_retention.pdf`, `results/figures/jamming_aware_training.pdf`.
- Table III (SWaP-C): notebook `07`, `results/tables/table_swapc.csv`.

## Citation

If you use this repository in your research, please cite the paper (see `CITATION.cff`).
