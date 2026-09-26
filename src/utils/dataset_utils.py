"""Dataset utilities: hashed dataset directory derived from the parameters.

The hashed directory includes alpha_min/alpha_max because the .npz content
depends on the echo amplitudes: without them in the hash, a change in the
paper parameters would silently reuse stale data. Bump _DATASET_VERSION when
the generator semantics change so --no-regen runs never reuse old files.

Version 4 adds the waveform parameters (map_type, map_param, sequence_length)
and the direct-path distribution (rician_kappa_db, doppler_direct_max) to the
digest: the revision moved the operating point to the Ulam parameter mu = 4.0,
whose impulse-like ACF changes every sample, so the mu = 3.9 tree must never
be reused. It also appends a digest of the cooperative peers when they are
enabled, because their echoes change the samples while the scenario name does
not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

_DATASET_VERSION = 4


def get_dataset_dir(config: Dict[str, Any]) -> Path:
    
    data = config["data"]
    echoes = tuple(sorted(data["echoes"]))
    max_delay = data["max_delay"]
    max_doppler = data["max_doppler"]
    snr_range = tuple(data["snr_range"])
    snr_step = data["snr_step"]
    feature_mode = data.get("feature_mode", "real")
    alpha_min = data.get("alpha_min", 1e-6)
    alpha_max = data.get("alpha_max", 0.01)
    coupling = bool(data.get("alpha_tau_coupling", False))
    floor = data.get("alpha_floor", 1e-3)
    map_type = data.get("map_type", "logistic")
    map_param = data.get("map_param")
    sequence_length = data.get("sequence_length")
    kappa_db = data.get("rician_kappa_db")
    doppler_direct = data.get("doppler_direct_max")
    params_str = (
        f"v{_DATASET_VERSION}_e{echoes}_d{max_delay}_D{max_doppler}_S{snr_range}"
        f"_s{snr_step}_f{feature_mode}_a{alpha_min}_{alpha_max}"
        f"_C{coupling}_F{floor}_m{map_type}{map_param}_n{sequence_length}"
        f"_k{kappa_db}_p{doppler_direct}{_channel_suffix(config)}"
        f"{_peers_suffix(config)}"
    )
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    raw_dir = Path(data.get("raw_dir", "data/raw"))
    return raw_dir / params_hash


def _channel_suffix(config: Dict[str, Any]) -> str:
    """Channel digest appended to the dataset hash, empty for the nominal channel.

    The data depend on the propagation model, so a different channel must land in
    a different directory; the nominal channel keeps the historical digest so the
    archived ``data/raw`` tree stays valid.
    """
    from src.data.channel_models import (
        DEFAULT_CHANNEL_MODEL,
        fingerprint_of,
        resolved_channel_section,
    )
    from src.utils.config_loader import DEFAULT_CHANNEL_VARIANT, load_channels

    nominal = load_channels()[DEFAULT_CHANNEL_VARIANT]
    nominal.setdefault("model", DEFAULT_CHANNEL_MODEL)
    section = resolved_channel_section(config)
    digest = fingerprint_of(section)
    if digest == fingerprint_of(nominal):
        return ""
    return f"_c{digest}"


def _peers_suffix(config: Dict[str, Any]) -> str:
    """Peer digest appended to the dataset hash when the peers are enabled.

    The cooperative echoes are extra taps of the same channel, so two scenarios
    that differ only in the ``peers`` section would otherwise share a directory
    and the peer-aware receiver would be trained on data without any peer.
    """
    from src.data.channel_models import fingerprint_of

    peers = config.get("peers")
    if not isinstance(peers, dict) or not bool(peers.get("enable", False)):
        return ""
    return f"_P{fingerprint_of({str(key): value for key, value in peers.items()})}"
