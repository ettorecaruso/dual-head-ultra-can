# Status of `results/` before the six-cell campaign

Three classes. There is a single `results` tree: a re-run writes into
`results/<mode>/<experiment>/`, and the previous version of that directory is moved
to `results/archive/<experiment>_<timestamp>/` by `--supersede`. Nothing here is
overwritten in place unless `--force` is passed on purpose, so the only thing that
changes is *which* copy the paper cites.

## 1. Final — do not touch, do not regenerate

Numbers, checkpoints and provenance that the submitted paper cites, none of which
any cell of the campaign rewrites:

| Path | Why it is final |
|---|---|
| `full/ber_vs_snr/**` (28 files) | benchmark metrics, the three `operating_region_table.csv` inputs and the **12 frozen checkpoints** the validation experiments load |
| `full/jamming/**` (40 files) | the checkpoints of the jamming grid and the Monte Carlo grid (mean/std/min/max + per-realization). Cell 6 only *extends* it with `rerun_jamming_mc.py`, which writes into its own safe subfolder |
| `full/classical_receivers/**`, `full/final_report/**` | equations-only baselines, SWaP-C, latency |
| `full/diagnostics/jamming_artifact/**` | the CW single-realization vs Monte Carlo diagnostic (its script regenerates it deterministically) |
| `full/logs/**` | the historical provenance manifest of the 2026-09-24 run. Not produced by the code: never edit or delete |
| `archive/frequency_agility_gain.csv` | orphan output listed in `full/logs/RUN_INFO.txt` with its sha256; kept so the historical manifest stays satisfiable |

## 2. Superseded-pending — keep until the replacement is validated

These are the **only** published copy of their section, so they are not deleted:
the new campaign writes to a sandbox and the promotion happens in the merge cell
(7), after a CSV-level diff.

| Path | Superseded by | Why it is still needed now |
|---|---|---|
| `full/frequency_agility/**` (284 KB) | cell 4: the same experiment on the 3GPP channel with `hold_mode = per_hop`, the blind-jammer modality and the CI columns | it is the `per_slot` + channel-A baseline the paper's figures are built from, and the `per_slot` vs `per_hop` comparison needs it |
| `full/jamming_interpretability/<arch>/**` (1000 KB incl. the retrained receiver) | cell 5: the same probe with `n_realizations = 30` and CI columns | it is the source of Fig. `layers` and of the clean-trained side of Fig. `jamaware`; note the four architectures come from two vintages (conv1d/qkv 2026-09-23, lstm/mc_dlsk 2026-09-16), which cell 5 aligns |
| `full/jamming_interpretability/jamming_aware_training/qkv/**` | cell 6, only if the optional second pass is run | it is the consolidated three-pass result with its provenance; the brief marks it as untouchable |

Promotion rule: run the cell with `--supersede`. The runner moves the previous
directory to `archive/<experiment>_<timestamp>/` before writing the new results,
so the comparison copy and the current copy both survive and the diff can be
reviewed afterwards with `scripts/tools/write_run_info.py`.

## 3. Removed during the cleanup

* the 25 intermediate PDFs that the runner writes inside `jamming/<arch>/jamming/`
  and the stale `frequency_agility/frequency_agility.pdf` copy: not consumed by
  any script, the paper reads `figures/*.pdf` and the CSVs stay here, so they are
  regenerable at any time (recorded in `archive/README.md`);
* the two smoke sandboxes used to verify the rewritten experiments: partial
  (`--mode fast`, incomplete grids) and therefore misleading. They no longer
  exist, and the campaign writes into `results/<mode>/<experiment>/` directly.

## Naming inside the archive

Directories created by `--supersede` are named `<experiment>_<timestamp>`, e.g.
`frequency_agility_2026-09-26T101533`. They hold a complete previous version of
that experiment, so they are the natural baseline for the `per_slot` vs `per_hop`
comparison once the new run lands.
