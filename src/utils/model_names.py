"""Canonical architecture names, user-facing aliases and display labels.

The codebase uses short internal keys for the four DL receivers
(``conv1d``, ``qkv``, ``lstm``, ``mc_dlsk``).  Those keys are part of the
on-disk contract: they appear in ``base_config.yaml`` (``model.seeds.<key>``),
in every result folder (``results/full/ber_vs_snr/<scenario>/<key>/``) and in
the tables/figures.  The *canonical scientific name* of the fourth one is
**MC-DLCSK**, so ``mc_dlsk`` is only a legacy abbreviation.

This module keeps the two concepts apart:

  - ``canonical_model_name``: maps any spelling used by a human or a CLI
    (``mc_dlcsk``, ``MC-DLCSK``, ``mc-dlsk``, ``mc_dlsk``, ``ultra-can``, ...)
    to the internal key used by configs/result paths/seeds dict, so that
    ``--model mc_dlcsk`` behaves exactly like the historical ``mc_dlsk``.
  - ``display_model_name``: maps an internal key to the label printed in
    tables and figures (``MC-DLCSK``, ``Ultra-CAN (QKV)``, ...).

Both functions are pure and dependency-free (imported by pipeline, baselines
and the runner).
"""

from __future__ import annotations

from typing import Dict, Tuple

#: Internal keys, in the canonical (figure/table) order.
CANONICAL_MODELS: Tuple[str, ...] = ("conv1d", "qkv", "lstm", "mc_dlsk")

#: ``normalised spelling -> internal key`` (normalisation: lower case, ``-``
#: and spaces replaced by ``_``).
_ALIASES: Dict[str, str] = {
    # Ultra-CAN (Conv1D)
    "conv1d": "conv1d",
    "conv_1d": "conv1d",
    "ultra_can": "conv1d",
    "ultra_can_conv1d": "conv1d",
    "dual_head_ultra_can": "conv1d",
    # Ultra-CAN (QKV)
    "qkv": "qkv",
    "qkv_attention": "qkv",
    "ultra_can_qkv": "qkv",
    # LSTM-OFDM-DCSK baseline
    "lstm": "lstm",
    "lstm_baseline": "lstm",
    "lstm_ofdm_dcsk": "lstm",
    # MC-DLCSK baseline: canonical spelling is ``mc_dlcsk`` (MC-DLCSK); the
    # historical abbreviation ``mc_dlsk`` is the internal key.
    "mc_dlsk": "mc_dlsk",
    "mc_dlcsk": "mc_dlsk",
    "mc_dlcs": "mc_dlsk",
    "mc_dlsk_baseline": "mc_dlsk",
    "mc_dlcsk_baseline": "mc_dlsk",
}

#: Labels used in markdown tables / figures / CSV columns.
_DISPLAY: Dict[str, str] = {
    "conv1d": "Ultra-CAN (Conv1D)",
    "qkv": "Ultra-CAN (QKV)",
    "lstm": "LSTM-OFDM-DCSK",
    "mc_dlsk": "MC-DLCSK",
    "blind_stat": "Blind statistical",
    "dcsk_correlator": "DCSK correlator",
}


def _normalise(name: str) -> str:
    """Lower-case and unify separators (``MC-DLCSK`` -> ``mc_dlcsk``)."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_model_name(name: str) -> str:
    """Return the internal key for any accepted architecture spelling.

    Args:
        name: architecture name, e.g. ``"mc_dlcsk"``, ``"MC-DLCSK"``,
            ``"mc_dlsk"``, ``"qkv"``.

    Returns:
        The internal key (one of :data:`CANONICAL_MODELS`).

    Raises:
        TypeError: if ``name`` is not a string.
        ValueError: if the name is not a known architecture.
    """
    if not isinstance(name, str):
        raise TypeError(f"model name must be a string, got: {type(name).__name__}")
    key = _normalise(name)
    if key in _ALIASES:
        return _ALIASES[key]
    if key in CANONICAL_MODELS:
        return key
    raise ValueError(
        f"unknown model name: {name!r} (accepted: {sorted(_ALIASES)})"
    )


def display_model_name(name: str) -> str:
    """Return the human-readable label for an architecture (never raises)."""
    try:
        return _DISPLAY.get(canonical_model_name(name), str(name))
    except (TypeError, ValueError):
        return _DISPLAY.get(str(name), str(name))


def canonical_or_none(name: object) -> str:
    """Best-effort canonicalisation: ``None``/unknown names are returned as-is.

    Used where a graceful fallback matters more than a hard failure (e.g. when
    normalising an optional CLI value that is validated later on).
    """
    if not isinstance(name, str):
        return name  # type: ignore[return-value]
    try:
        return canonical_model_name(name)
    except ValueError:
        return name
