# Dual-Head Ultra-CAN ISAC - Results report

> **Stale, superseded.** This report predates the 5-realization re-run of the
> jamming experiments and the consolidation of the jamming-aware probe. Its
> barrage row (BER at JSR $-2$ dB) does **not** match the released data: it quotes
> $0.0221$, while `jamming_interpretability/.../comparison_clean_vs_jamaware.csv`
> and `jamming/<arch>/jamming/jamming_results_*.csv` give $0.0220$ and $0.0423$
> respectively under their own protocols. The paper quotes the
> `jamming_interpretability` files only. Do not copy numbers from this report;
> read the CSVs each section names.

This report aggregates the outputs of the runner experiments and mirrors the evaluation flow of the paper.

## Communication benchmark (operating region, SNR >= 5 dB)

| scenario | architecture | pooled BER | min SNR @1e-4 (dB) | best BER |
|---|---|---|---|---|
| k1_doppler_full | Ultra-CAN (Conv1D) | 1.4752e-04 | 11.0 | 7.1429e-05 |
| k1_doppler_full | Ultra-CAN (QKV) | 1.1349e-04 | 11.0 | 4.8000e-05 |
| k1_doppler_full | LSTM-OFDM-DCSK | 2.0995e-04 | n/a | 1.4714e-04 |
| k1_doppler_full | MC-DLCSK | 2.7850e-04 | n/a | 1.8500e-04 |
| k3_doppler_full | Ultra-CAN (Conv1D) | 2.9388e-04 | n/a | 1.5286e-04 |
| k3_doppler_full | Ultra-CAN (QKV) | 1.6450e-04 | 17.0 | 8.5385e-05 |
| k3_doppler_full | LSTM-OFDM-DCSK | 2.9789e-04 | n/a | 1.8500e-04 |
| k3_doppler_full | MC-DLCSK | 8.4778e-04 | n/a | 6.3500e-04 |
| k3_doppler_limited | Ultra-CAN (Conv1D) | 2.8197e-04 | n/a | 1.5000e-04 |
| k3_doppler_limited | Ultra-CAN (QKV) | 1.7230e-04 | 19.0 | 9.3636e-05 |
| k3_doppler_limited | LSTM-OFDM-DCSK | 3.5387e-04 | n/a | 2.0400e-04 |
| k3_doppler_limited | MC-DLCSK | 3.5219e-04 | n/a | 2.2600e-04 |

## Sensing: delay estimation (single-echo scenario)

| architecture | corr(tau) @top SNR | RMSE tau (samples) |
|---|---|---|
| Ultra-CAN (Conv1D) | 0.928 | 3.56 |
| Ultra-CAN (QKV) | 0.919 | 3.75 |
| LSTM-OFDM-DCSK | 0.922 | 3.71 |
| MC-DLCSK | 0.921 | 3.70 |

## Blind statistical reference receiver

| scenario | mean BER (SNR >= 5 dB) |
|---|---|
| k1_doppler_full | 0.1884 |
| k3_doppler_full | 0.2311 |
| k3_doppler_limited | 0.2313 |

## Jamming robustness and interpretability

| architecture | clean BER | BER @JSR=-2 dB (barrage) |
|---|---|---|
| Ultra-CAN (Conv1D) | 0.0021 | 0.0226 |
| Ultra-CAN (QKV) | 0.0021 | 0.0221 |
| LSTM-OFDM-DCSK | 0.0016 | 0.0173 |
| MC-DLCSK | 0.0014 | 0.0174 |

## Jamming-aware training control (QKV receiver)

| jammer | clean-trained BER @JSR=-2 dB | jamming-aware BER @JSR=-2 dB |
|---|---|---|
| cw | 0.3346 | 0.0135 |
| barrage | 0.0221 | 0.0236 |
| partial_band | 0.2513 | 0.0237 |

## Memory footprint (SWaP-C)

| architecture | parameters | footprint (KiB, float32) |
|---|---|---|
| conv1d | 26597 | 103.9 |
| qkv | 43172 | 168.6 |
| lstm | 45988 | 179.6 |
| mc_dlsk | 44756 | 174.8 |
