# `results/archive/` — superseded and orphan artifacts

Nothing here is consumed by the paper, by the figure/table scripts or by the
tests. The files are kept only so that the historical tree stays auditable; they
can be deleted at any time without affecting any result.

## `legacy_plots_2026_09/`

Intermediate plots of the 2026-09 jamming and frequency-agility runs, moved out
of `results/full/` because the paper reads **CSVs** and renders its figures from
`figures/*.pdf`:

| Artifact | Why it is not a deliverable |
|---|---|
| `jamming/<arch>/jamming/plots/ber_vs_jsr_{cw,barrage,partial_band,overlay,<arch>_vs_clean}.pdf` | per-receiver diagnostics written by the runner's jamming loop; `ber_vs_jsr_<arch>_vs_clean.pdf` is duplicated both here and one directory above |
| `jamming/<arch>/jamming/ber_vs_jsr_<arch>_vs_clean.pdf` | same content as the copy inside `plots/` |
| `frequency_agility/frequency_agility.pdf` | stray copy; the figure of the paper is `figures/frequency_agility.pdf`, produced by the figure script |
| `frequency_agility_gain.csv` | orphan output: no script writes or reads it. It is listed in `results/full/logs/RUN_INFO.txt` with its sha256, so it is archived instead of deleted, which keeps the historical manifest satisfiable |

The CSVs that back those plots remain in `results/full/jamming/<arch>/jamming/`
and `results/full/frequency_agility/<arch>/`, so the plots can be regenerated at
any time with `scripts/figures/` and `scripts/tools/`.
