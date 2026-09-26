# Dual-Head Ultra-CAN ISAC - Results report

This report aggregates the outputs of the runner experiments and mirrors the evaluation flow of the paper.

## Communication benchmark (operating region, SNR >= 5 dB)

| scenario | architecture | pooled BER | min SNR @1e-4 (dB) | best BER |
|---|---|---|---|---|
| k1_doppler_full | Ultra-CAN (Conv1D) | 0.5600 | n/a | 0.5400 |
| k1_doppler_full | Ultra-CAN (QKV) | 0.5000 | n/a | 0.5000 |

## Sensing: delay estimation (single-echo scenario)

| architecture | corr(tau) @top SNR | RMSE tau (samples) |
|---|---|---|
| Ultra-CAN (Conv1D) | 0.445 | 2.32 |
| Ultra-CAN (QKV) | 0.453 | 2.02 |

## Blind statistical reference receiver

| scenario | mean BER (SNR >= 5 dB) |
|---|---|
| k1_doppler_full | 0.2200 |

## Jamming robustness and interpretability

| architecture | clean BER | BER @JSR=-2 dB (barrage) |
|---|---|---|
| Ultra-CAN (Conv1D) | 0.5000 | n/a |
| Ultra-CAN (QKV) | 0.5033 | n/a |

## Jamming-aware training control (QKV receiver)

| jammer | clean-trained BER @JSR=-2 dB | jamming-aware BER @JSR=-2 dB |
|---|---|---|

## Memory footprint (SWaP-C)

| architecture | parameters | footprint (KiB, float32) |
|---|---|---|
| conv1d | 26629 | 104.0 |
| qkv | 43204 | 168.8 |
| lstm | 45988 | 179.6 |
| mc_dlsk | 44756 | 174.8 |
