#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Biosensor-as-Graph representation for GNN surrogate training.

A biosensor array is represented as a directed acyclic graph:
  - Nodes: one per detection channel (SOST, CTX, P1NP)
  - Edges: signal combination (all pairs, undirected; weighted by w_ctx, w_p1np, w_scl)
  - Global: shared parameters (sensitivity, response_time, noise)

Node features (per channel i):
  [log_kd_i, weight_i, log_conc_healthy_i, log_conc_disease_i,
   occ_healthy_i, occ_disease_i, occ_ratio_i]

Edge features (channel i -> channel j):
  [weight_product, kd_ratio]

Global graph features:
  [log_sensitivity, log_response_time, noise_enc, scenario_enc, biosensor_type_enc]

Target outputs per graph:
  [detection_rate, fnr, ttd] for a given scenario

This representation allows a GNN to:
  1. Reason about each channel's contribution independently (node features)
  2. Model cross-channel correlations (edge features)
  3. Handle variable-topology biosensors (remove channels by setting weight=0)
  4. Generalize across scenarios via the global scenario_enc node
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

CHANNEL_NAMES = ["SOST", "CTX", "P1NP"]
N_CHANNELS = 3

# Nominal sensor concentrations (nM) per scenario
_NOMINAL_CONCS = {
    "healthy":  [0.375, 0.200, 0.350],   # [SOST, CTX, P1NP]
    "pmo_mild": [0.5625, 0.300, 0.385],
    "pmo":      [0.875,  0.500, 0.525],
    "ckd_mbd":  [1.125,  0.500, 0.625],
}

# Scenario encoding (integer)
SCENARIO_ENC = {"healthy": 0, "pmo_mild": 1, "pmo": 2, "ckd_mbd": 3}
NOISE_ENC    = {"low": 0, "medium": 1, "realistic": 2, "high": 3}
BTYPE_ENC    = {"direct_binding": 0, "amplifying": 1, "array": 2}

NODE_FEATURE_DIM = 7    # per channel: [log_kd, weight, log_c_h, log_c_d, occ_h, occ_d, occ_ratio]
EDGE_FEATURE_DIM = 2    # per edge: [w_product, kd_ratio]
GLOBAL_FEATURE_DIM = 5  # [log_sens, log_rt, noise_enc, scenario_enc, btype_enc]
OUTPUT_DIM = 3          # [DR, FNR, TTD]


@dataclass
class BiosensorGraph:
    """
    Graph representation of one biosensor configuration + scenario.

    Attributes:
        node_features:   (N_CHANNELS, NODE_FEATURE_DIM)
        edge_index:      (2, E) integer array — pairs (src, dst)
        edge_features:   (E, EDGE_FEATURE_DIM)
        global_features: (GLOBAL_FEATURE_DIM,)
        targets:         (OUTPUT_DIM,) or None
        config:          original config dict (optional, for debugging)
        scenario:        scenario name
    """
    node_features:   np.ndarray
    edge_index:      np.ndarray
    edge_features:   np.ndarray
    global_features: np.ndarray
    targets:         Optional[np.ndarray] = None
    config:          Optional[Dict] = None
    scenario:        str = "pmo"

    def to_flat_vector(self) -> np.ndarray:
        """Flatten all graph features into a single 1D vector for sklearn."""
        return np.concatenate([
            self.node_features.flatten(),
            self.edge_features.flatten(),
            self.global_features,
        ], axis=0)

    @property
    def n_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def flat_dim(self) -> int:
        return N_CHANNELS * NODE_FEATURE_DIM + (N_CHANNELS * (N_CHANNELS - 1)) * EDGE_FEATURE_DIM + GLOBAL_FEATURE_DIM


def _occ(conc: float, kd: float) -> float:
    """Langmuir occupancy."""
    return float(conc / (kd + conc + 1e-12))


def config_to_graph(
    config: Dict,
    scenario: str,
    targets: Optional[Tuple[float, float, float]] = None,
) -> BiosensorGraph:
    """
    Convert a biosensor config dict + scenario to a BiosensorGraph.

    Args:
        config:   biosensor parameter dict (same format as surrogate API)
        scenario: one of "healthy", "pmo_mild", "pmo", "ckd_mbd"
        targets:  optional (DR, FNR, TTD) ground-truth tuple for training

    Returns:
        BiosensorGraph ready for GNN or flat-vector ML.
    """
    def _safe_float(v, default):
        f = float(v) if v is not None else default
        return f if not np.isnan(f) else default

    kd_nm      = _safe_float(config.get("kd_nm"),        1.0)
    kd_ctx     = _safe_float(config.get("kd_ctx_nm"),    kd_nm)
    kd_p1np    = _safe_float(config.get("kd_p1np_nm"),   kd_nm)
    w_ctx      = _safe_float(config.get("w_ctx"),        0.10)
    w_p1np     = _safe_float(config.get("w_p1np"),       0.10)
    w_scl      = max(0.0, 1.0 - w_ctx - w_p1np)
    sensitivity  = _safe_float(config.get("sensitivity"), 1.0)
    response_time = _safe_float(config.get("response_time_s"), 600.0)
    noise_preset  = config.get("noise_preset", "realistic")
    biosensor_type = config.get("biosensor_type", "array")

    kds = [kd_nm, kd_ctx, kd_p1np]
    weights = [w_scl, w_ctx, w_p1np]

    concs_h = _NOMINAL_CONCS["healthy"]
    concs_d = _NOMINAL_CONCS.get(scenario, concs_h)

    # --- Node features ---
    node_feats = np.zeros((N_CHANNELS, NODE_FEATURE_DIM), dtype=np.float32)
    for i, (kd, w, c_h, c_d) in enumerate(zip(kds, weights, concs_h, concs_d)):
        occ_h = _occ(c_h, kd)
        occ_d = _occ(c_d, kd)
        occ_ratio = occ_d / max(occ_h, 1e-9)
        node_feats[i] = [
            np.log10(max(kd, 1e-3)),      # log_kd
            w,                             # channel weight
            np.log10(max(c_h, 1e-4)),      # log_conc_healthy
            np.log10(max(c_d, 1e-4)),      # log_conc_disease
            occ_h,                         # occupancy at healthy
            occ_d,                         # occupancy at disease
            occ_ratio,                     # discrimination ratio
        ]

    # --- Edges: all directed pairs (i->j and j->i for undirected) ---
    src, dst = [], []
    for i in range(N_CHANNELS):
        for j in range(N_CHANNELS):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = np.array([src, dst], dtype=np.int32)   # (2, E)

    # Edge features
    n_edges = len(src)
    edge_feats = np.zeros((n_edges, EDGE_FEATURE_DIM), dtype=np.float32)
    for e, (i, j) in enumerate(zip(src, dst)):
        kd_i, kd_j = kds[i], kds[j]
        w_i, w_j   = weights[i], weights[j]
        edge_feats[e] = [
            w_i * w_j,                             # weight product (signal co-activation)
            np.log10(max(kd_i, 1e-3)) - np.log10(max(kd_j, 1e-3)),  # log kd ratio
        ]

    # --- Global features ---
    global_feats = np.array([
        np.log10(max(sensitivity, 1e-3)),
        np.log10(max(response_time, 1.0)),
        float(NOISE_ENC.get(noise_preset, 2)),
        float(SCENARIO_ENC.get(scenario, 2)),
        float(BTYPE_ENC.get(biosensor_type, 2)),
    ], dtype=np.float32)

    # --- Targets ---
    target_arr = None
    if targets is not None:
        dr, fnr, ttd = targets
        target_arr = np.array([
            float(dr),
            float(fnr),
            float(np.log1p(min(ttd, 9000.0)) / np.log1p(9000.0)),  # normalize TTD to [0,1]
        ], dtype=np.float32)

    return BiosensorGraph(
        node_features=node_feats,
        edge_index=edge_index,
        edge_features=edge_feats,
        global_features=global_feats,
        targets=target_arr,
        config=config,
        scenario=scenario,
    )


def dataset_from_csv(
    csv_path: str,
    scenarios: Optional[List[str]] = None,
) -> List[BiosensorGraph]:
    """
    Build a list of BiosensorGraph objects from a surrogate training CSV.

    The CSV must contain columns matching the surrogate training format
    (output of dataset/generator.py master_index.csv).

    Args:
        csv_path: path to master_index.csv
        scenarios: which scenario labels to include (default: all 4)

    Returns:
        List of BiosensorGraph with targets set.
    """
    import pandas as pd

    if scenarios is None:
        scenarios = ["healthy", "pmo_mild", "pmo", "ckd_mbd"]

    df = pd.read_csv(csv_path)
    graphs = []

    # Column name mapping — support both v16 style (kd, response_time) and
    # legacy style (kd_nm, response_time_s, kd_ctx_nm, kd_p1np_nm).
    def _col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_kd   = _col(df, "kd_nm", "kd_scl", "kd")
    col_rt   = _col(df, "response_time_s", "response_time")
    col_ctx  = _col(df, "kd_ctx_nm", "kd_ctx")
    col_p1np = _col(df, "kd_p1np_nm", "kd_p1np")

    missing = [name for name, col in [
        ("kd_nm/kd", col_kd), ("response_time_s/response_time", col_rt),
        ("biosensor_type", _col(df, "biosensor_type")),
        ("noise_preset", _col(df, "noise_preset")),
        ("scenario", _col(df, "scenario")),
        ("detection_rate", _col(df, "detection_rate")),
        ("false_negative_rate", _col(df, "false_negative_rate")),
        ("time_to_detection", _col(df, "time_to_detection")),
    ] if col is None]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for _, row in df.iterrows():
        sc = row.get("scenario", "pmo")
        if sc not in scenarios:
            continue

        rt_raw = row[col_rt] if col_rt else float("nan")
        rt_val = float(rt_raw) if not pd.isna(rt_raw) else 600.0  # MOBO default
        config = {
            "kd_nm": row[col_kd],
            "sensitivity": row["sensitivity"],
            "response_time_s": rt_val,
            "biosensor_type": row["biosensor_type"],
            "noise_preset": row["noise_preset"],
        }
        if col_ctx and not pd.isna(row.get(col_ctx, float("nan"))):
            config["kd_ctx_nm"] = row[col_ctx]
        if col_p1np and not pd.isna(row.get(col_p1np, float("nan"))):
            config["kd_p1np_nm"] = row[col_p1np]
        for key in ("w_ctx", "w_p1np"):
            if key in df.columns and not pd.isna(row.get(key, float("nan"))):
                config[key] = row[key]

        targets = (
            float(row["detection_rate"]),
            float(row["false_negative_rate"]),
            float(row["time_to_detection"]),
        )
        g = config_to_graph(config, str(sc), targets=targets)
        graphs.append(g)

    return graphs
