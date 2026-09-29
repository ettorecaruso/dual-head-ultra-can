#!/usr/bin/env bash
# Regenerate every figure and table of the paper from the CSVs in results/full.
#
#   PYTHON=/path/to/python ./scripts/make_all.sh [run_root]
#
# ``run_root`` is optional and only affects the two producers that read a run
# directory: it defaults to results/full and can point at an archived copy, e.g.
# results/archive/frequency_agility_2026-09-26T101533.
#
# Figures are vector PDFs written to figures/ (the paper reads them through
# \graphicspath{{../figures/}} in latex/paper.tex); tables are written next to the
# results they summarise. No PNG preview is produced anywhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${1:-}"

cd "${REPO_ROOT}"

echo "== tables =="
"${PYTHON}" scripts/tables/make_table_operating_region.py
if [ -n "${RUN_ROOT}" ]; then
    "${PYTHON}" scripts/tables/make_table_channel_generalization.py "${RUN_ROOT}/channel_generalization"
    "${PYTHON}" scripts/figures/make_fig_channel_generalization.py "${RUN_ROOT}/channel_generalization"
else
    "${PYTHON}" scripts/tables/make_table_channel_generalization.py
    "${PYTHON}" scripts/figures/make_fig_channel_generalization.py
fi

echo "== figures =="
"${PYTHON}" scripts/figures/make_fig_architecture.py
"${PYTHON}" scripts/figures/make_fig_ber_full_linear.py
# The blind reference is generated inside results/newRes; copy it into the run
# tree so every other consumer (tables, zoom figures, the merged tree shipped to
# the paper repo) finds it there instead of looking for a curve that is not in
# the tree it reads.
for SCENARIO in k1_doppler_full k3_doppler_full k3_doppler_limited; do
    SRC="${REPO_ROOT}/results/newRes/${SCENARIO}/blind_stat/metrics.csv"
    if [ -f "${SRC}" ]; then
        mkdir -p "${REPO_ROOT}/results/full/ber_vs_snr/${SCENARIO}/blind_stat"
        cp "${SRC}" "${REPO_ROOT}/results/full/ber_vs_snr/${SCENARIO}/blind_stat/metrics.csv"
    fi
done
"${PYTHON}" scripts/figures/make_fig_ber_region_zoom.py
"${PYTHON}" scripts/figures/make_fig_sensing_delay.py
"${PYTHON}" scripts/figures/make_fig_layer_retention.py
"${PYTHON}" scripts/figures/make_fig_sensing_under_jamming.py
"${PYTHON}" scripts/figures/make_fig_jamming_aware_control.py
"${PYTHON}" scripts/figures/make_fig_frequency_agility.py
"${PYTHON}" scripts/figures/make_fig_frequency_agility_reaction.py

echo "== cross-channel and jamming synthesis =="
"${PYTHON}" scripts/figures/make_fig_channel_summary.py
"${PYTHON}" scripts/figures/make_fig_sensing_observability.py
"${PYTHON}" scripts/figures/make_fig_retraining_control.py
"${PYTHON}" scripts/figures/make_fig_peer_estimation.py
"${PYTHON}" scripts/figures/make_fig_peer_estimation_price.py
"${PYTHON}" scripts/figures/make_fig_agility_law.py
"${PYTHON}" scripts/figures/make_fig_jamaware_margin.py
"${PYTHON}" scripts/figures/make_fig_jamming_tone_selectivity.py
"${PYTHON}" scripts/figures/make_fig_lpi_detectability.py

echo "== done: figures/ and the result tables are up to date =="

