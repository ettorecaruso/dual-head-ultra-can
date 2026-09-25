# `results/archive/` — superseded and orphan artifacts

Nothing here is consumed by the paper, by the figure/table scripts or by the
tests. The files are kept only so that the historical tree stays auditable; they
can be deleted at any time without affecting any result.

## `frequency_agility_gain.csv`

Orphan output of the first frequency-agility run: no script writes or reads it.
It is listed in `results/full/logs/RUN_INFO.txt` with its sha256, so it is archived
instead of deleted, which keeps the historical manifest satisfiable.

## Removed on the cleanup pass

| Artifact | Why it is not a deliverable |
|---|---|
| `legacy_plots_2026_09/jamming/<arch>/jamming/plots/*.pdf` and the duplicated `ber_vs_jsr_<arch>_vs_clean.pdf` | per-receiver plots written by the runner's jamming loop; the paper reads `figures/*.pdf` and the CSVs that back them stay in `results/full/jamming/<arch>/jamming/` |
| `legacy_plots_2026_09/frequency_agility/frequency_agility.pdf` | stray copy; the figure of the paper is `figures/frequency_agility.pdf`, produced by `scripts/figures/make_fig_frequency_agility.py` |

Delete them again at will: everything except the CSV above is a pure rendering of
the CSVs that remain under `results/full/`.

