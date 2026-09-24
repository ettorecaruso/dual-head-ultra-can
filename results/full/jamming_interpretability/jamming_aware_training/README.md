# `jamming_aware_training/` — the jamming-aware QKV probe

`qkv/` is the single, **complete** result of the jamming-aware QKV probe used in
the paper: 19 conditions (clean, CW x 6 JSR, barrage x 6 JSR, partial band x 6
JSR), each averaged over 5 jammer realizations, plus the per-realization detail
(`conditions_realizations.csv`), the per-SNR breakdown (`per_snr.csv`), the
comparison against the clean-trained receiver
(`comparison_clean_vs_jamaware.csv`) and the retrained model itself
(`best_model.keras`). It is the file consumed by
`scripts/make_fig_jamming_aware_control.py` and by the "Jamming-aware training"
paragraph of the paper.

**Consolidation note.** The probe is long, so it was executed in three partial
passes (CW + clean, partial band, barrage) whose outputs were then consolidated
into this single directory. The intermediate pass directories are **not part of
the released results**, and no value was altered in the consolidation: the
consolidated table is bit-identical to the passes for every shared
(jammer, JSR) condition (verified: `max|delta| = 0`). The consolidation is
recorded in `qkv/run_metadata.json` (`consolidated_chunks`, `provenance`).

The per-jammer separation lives **inside** the table, through the `jammer`
column: one row per (jammer, JSR) x 5 realizations. Do not split it into one
directory per jammer — the paper's Fig. `jamaware` needs the three curves in the
same table.

