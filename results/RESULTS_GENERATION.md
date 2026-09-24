# Results of the Dual-Head Ultra-CAN ISAC evaluation — manifest

Curated outputs of every experiment reported in the paper, in the same layout the
runner produces (`results/full/<experiment>/...`), plus the run provenance.
This directory is intended as a **drop-in replacement** for
`dual-head-ultra-can/results/`.

Every number quoted in the paper is read from a CSV in this tree; the figures are
regenerated from these CSVs by the scripts in `scripts/` (no value is hard-coded
in the plotting code).

## 1. Content

| experiment | files | produced by | recorded metadata |
|---|---|---|---|
| `ber_vs_snr` — 3 scenarios x 5 receivers (Conv1D, QKV, LSTM-OFDM-DCSK, MC-DLCSK, blind statistical), 14 SNR points | 16 csv + 12 `.keras` | `runner.py --experiments ber_vs_snr --mode full` | `n_errors`/`n_symbols` per point in each `metrics.csv`; `operating_region_table.csv` |
| `classical_receivers` — equations-only reference | 2 csv | `runner.py --experiments classical_receivers --mode full` | deterministic (`ber_vs_k.csv`, `ber_vs_doppler.csv`) |
| `jamming` — 4 receivers x 3 jammers x 11 JSR x **5 realizations** | 36 csv + 24 pdf + 4 `.keras` | `runner.py --experiments jamming --mode full`, re-evaluated with `scripts/rerun_jamming_mc.py` | `jamming_results_*.csv` = mean/std/min/max over realizations; `jamming_realizations_*.csv` = per-realization BER |
| `jamming_interpretability` — 4 receivers x 19 conditions x 5 realizations, layer-wise activation probes | 16 csv + 5 json + 1 yaml + 1 `.keras` + 1 md | `runner.py --experiments jamming_interpretability --mode full`, then `scripts/run_jam_aware_qkv.py` (see `jamming_aware_training/README.md` for the consolidation note) | `run_metadata.json` per receiver, `logs/config_used.yaml` |
| `frequency_agility` — 4 receivers x 4 jammer models x 6 JSR x 5 realizations + dwell sweep (5 hop rates) | 13 csv + 4 json + 1 yaml + 1 pdf | `runner.py --experiments frequency_agility --mode full --no-regen` | `run_metadata.json` per receiver (`channel_process`, `hop`), `logs/config_used.yaml` |
| `final_report` — SWaP-C table, latency, aggregated report | `weight_table.tex`, `latency_results.json`, `final_report.md` | `runner.py --experiments final_report --mode full` + `scripts/latency_bench.py` | latency measured on the reference laptop (AMD Ryzen 5 PRO 4650U, float32, single thread, TF 2.21; median of five trials) |
| `diagnostics/jamming_artifact` — CW single-realization artifact vs Monte Carlo | 2 csv + 1 pdf | `scripts/diagnose_jamming_mc.py` | `f_cw`, per-realization BER |
| `logs/` — provenance of the 2026-09-24 frequency-agility run | `runner.log`, `RUN_INFO.txt`, `CHECKPOINTS_INFO.txt` | the runner itself | git commit `12e9017`, TensorFlow 2.20.0, Tesla T4, sha256 of the four checkpoints |

Pointer to the paper: `ber_vs_snr` -> communication benchmark and sensing
(Table `tab:metrics`, `tab:blind_ber`, Figs. `ber_full`, `zoom`, `sensing`);
`jamming` + `jamming_interpretability` -> jamming robustness and layer retention
(Fig. `layers`, `jamaware`); `frequency_agility` -> frequency-agility section;
`final_report` -> SWaP-C and latency (`tab:swapc`, `tab:latency`).

## 2. Headline numbers of the frequency-agility run (validated)

Configuration: 8 channels, slot 10 us, 100 k hop/s, 20 000 bursts per condition,
5 realizations, SNR 21 dB, K = 3 echoes, all four receivers.

| jammer model | covered fraction (`fh_on`) | BER, hopping | BER, fixed carrier | gain |
|---|---|---|---|---|
| `barrage` (matched control) | 1.000 | 0.44-0.50 | 0.44-0.50 | 0 dB (as expected) |
| `fixed_partial` (25 %) | 0.250 | 0.11-0.13 | 0.44-0.50 | **+6.0 dB** |
| `sweep` (duty cycle 1/8) | 0.122 | 0.05-0.06 | 0.44-0.50 | **+9.1 dB** |
| `follower` (reaction 2 bursts = 20 us) | 0.000 | 1e-4 - 9e-4 | 0.44-0.50 | jammer never engages |

`barrage` is the matched control: the mask is identical in both modalities, and
the per-realization BER is **exactly equal** (`max|off-on| = 0`) for all four
receivers, i.e. the two modalities share the same per-slot channel process and
the same transmission. Dwell sweep (`follower`, JSR +6 dB): dwell 1 -> 0.000
(BER ~1e-4), dwell 2 -> 0.000, dwell 4 -> 0.500 (BER 0.20-0.23), dwell 8 -> 0.750
(BER 0.30-0.34), dwell 16 -> 0.875 (BER 0.35-0.40): the protection comes from
hopping faster than the jammer reaction time.

## 3. Provenance

| directories | run date | environment | code |
|---|---|---|---|
| `ber_vs_snr`, `classical_receivers`, `final_report` | 2026-09-16 | not recorded by that run | commit not recorded |
| `diagnostics/jamming_artifact`, `jamming`, `jamming_interpretability` | 2026-09-23 | Colab GPU, TensorFlow 2.20 | commit `aa516a6` (the run that added the 5 jammer realizations and the per-realization CSVs) |
| `frequency_agility`, `logs/` | 2026-09-24 | Colab Tesla T4, TensorFlow 2.20.0 | commit `12e9017`, see `logs/RUN_INFO.txt` |

The four `best_model.keras` of `ber_vs_snr/k3_doppler_full/` are the frozen
receivers used by `frequency_agility`, `channel_generalization` and the latency
benchmark; their sha256 fingerprints are in `logs/CHECKPOINTS_INFO.txt`:

| receiver | parameters | file |
|---|---|---|
| Ultra-CAN (Conv1D) | 26 597 | `ber_vs_snr/k3_doppler_full/conv1d/best_model.keras` |
| Ultra-CAN-QKV | 43 172 | `ber_vs_snr/k3_doppler_full/qkv/best_model.keras` |
| LSTM-OFDM-DCSK | 45 988 | `ber_vs_snr/k3_doppler_full/lstm/best_model.keras` |
| MC-DLCSK | 44 756 | `ber_vs_snr/k3_doppler_full/mc_dlsk/best_model.keras` |

The 4 checkpoints of `jamming/<arch>/` are the models the jamming grid was
evaluated on, and `jamming_interpretability/jamming_aware_training/qkv/` is the
jamming-aware QKV receiver of Fig. `jamaware` (17 `.keras` files in total).

## 4. Notes and known limitations

1. **SNR convention of the jamming curves.** `jamming_results_<jammer>.csv` is the
   BER averaged over the *whole* SNR grid (-5 ... 21 dB), because the jamming
   evaluation averages `evaluate_model` over all SNR points. The per-SNR breakdown
   is available in `jamming_interpretability/<arch>/per_snr.csv` (6 SNR points).
   If a single BER-vs-JSR curve is reported, the SNR convention must be stated.
2. **Follower jammer.** With `dwell_bursts = 1` and `follower_reaction_bursts = 2`
   the follower never engages (`jammed_fraction = 0`): the flat BER is the absence
   of jamming, not a coding gain. The meaningful presentation is the dwell sweep
   (`frequency_agility_vs_dwell.csv`).
3. **Interpretability schemas differ slightly between revisions.** For `conv1d`
   and `qkv`, `conditions.csv`/`per_snr.csv` carry the realization summary columns
   (`ber_std`, `realization`, `n_realizations`); for `lstm` and `mc_dlsk` the files
   were produced by an earlier revision of the probe and end at
   `arch,jammer,jsr_db`. The layer-retention columns (`*_cos`, `*_energy`) are
   present for all four receivers.
4. **BER stopping rule.** Every `ber_vs_snr` point accumulates until 100 bit
   errors (cap 2 x 10^6 symbols), so `n_symbols` differs from point to point;
   per-point binomial uncertainty is not reported.
5. **`channel_generalization` is intentionally absent** from this tree: it is being
   regenerated with `data.echoes: [3]` so that the nominal row and the mismatched
   rows share the same K (today the nominal row is K = 3 while the variants
   alternate K = 1/3, which mixes two variables in one table).
6. **Excluded on purpose:** `results/newRes/` (the default output directory of
   `make_fig_ber_full_linear.py`, a duplicate of `ber_vs_snr`) and the
   Colab-measured `latency_results.json` (the latency in `final_report/` is the
   reference-laptop measurement quoted by the paper).
7. **`jamming_aware_training/` ships a single, complete directory (`qkv/`).** The
   jamming-aware QKV probe is long, so it was executed in three partial passes
   (CW + clean, partial band, barrage); their outputs were consolidated into
   `qkv/` and the intermediate pass directories are **not part of the released
   results**. No value was altered in the consolidation: the consolidated table
   is bit-identical to the passes for every shared (jammer, JSR) condition
   (`max|delta| = 0`). This is recorded in `qkv/run_metadata.json`
   (`consolidated_chunks`, `provenance`). The per-jammer separation lives inside
   the table, through the `jammer` column (one row per (jammer, JSR) x 5
   realizations), which is what Fig. `jamaware` needs.

## 5. How to regenerate

```bash
python src/experiments/runner.py --experiments ber_vs_snr --mode full
python src/experiments/runner.py --experiments classical_receivers --mode full
python src/experiments/runner.py --experiments jamming --mode full
python scripts/rerun_jamming_mc.py                          # per-realization jamming grid
python src/experiments/runner.py --experiments jamming_interpretability --mode full
python scripts/run_jam_aware_qkv.py                          # jamming-aware QKV training
python src/experiments/runner.py --experiments frequency_agility --mode full --no-regen
python scripts/make_fig_ber_full_linear.py ; python scripts/make_fig_ber_region_zoom.py
python scripts/make_fig_sensing_delay.py ; python scripts/make_table_operating_region.py
python scripts/make_fig_layer_retention.py ; python scripts/make_fig_jamming_aware_control.py
python scripts/make_fig_frequency_agility.py ; python scripts/latency_bench.py
python src/experiments/runner.py --experiments final_report --mode full
```

## 6. Outside this folder

`results/full/frequency_agility/frequency_agility.pdf` is the figure of the
frequency-agility experiment: it belongs to the repository's `figures/`
directory (`dual-head-ultra-can/figures/frequency_agility.pdf`). The remaining
PDFs in `figures/` were also regenerated on 2026-09-24 from these same CSVs;
they differ from the tracked versions at byte level but not in content.

