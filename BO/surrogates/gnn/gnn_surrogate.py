#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GNN-based biosensor surrogate model.

Architecture:
  - Message-passing layers (3 rounds) over the biosensor graph
  - Global readout (mean aggregation of node embeddings + global features)
  - MLP head per output (DR, FNR, TTD)

Two implementations:
  1. NumpyMPNN  - pure numpy, always available, supports inference only
                  (trained weights can be loaded from .npz file)
  2. TorchMPNN  - PyTorch, required for training
                  (import guard: only imported if torch is available)

The GNNSurrogate wrapper auto-detects torch availability and uses TorchMPNN
for training and NumpyMPNN for inference.

Design rationale for message passing:
  - Round 1: aggregate neighbor node features (cross-channel interaction)
  - Round 2: aggregate with edge features weighted by edge importance
  - Round 3: global pooling -> graph-level embedding
  - This allows the GNN to learn that CTX and SOST together predict better
    than either alone (synergy captured by edge messages)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .graph_biosensor import (
        BiosensorGraph, config_to_graph,
        N_CHANNELS, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, OUTPUT_DIM,
    )
except ImportError:
    from graph_biosensor import (
        BiosensorGraph, config_to_graph,
        N_CHANNELS, NODE_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, OUTPUT_DIM,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numpy message-passing network (inference only)
# ---------------------------------------------------------------------------

class NumpyMPNN:
    """
    Pure-numpy message-passing neural network for biosensor graph inference.

    Implements 3-layer MPNN with:
      - Linear message functions (W_e @ [h_i, h_j, e_ij] + b_e)
      - Mean aggregation
      - ReLU activations
      - Final MLP head per output (DR, FNR, TTD)

    Weights can be loaded from a .npz file produced by TorchMPNN.save().
    Untrained weights produce random predictions — train with TorchMPNN first.
    """

    HIDDEN_DIM = 64
    N_LAYERS = 3

    def __init__(self, weights_path: Optional[str] = None):
        self._weights: Optional[Dict[str, np.ndarray]] = None
        if weights_path:
            self.load(weights_path)
        else:
            logger.warning("NumpyMPNN: no weights loaded — call load() or train with TorchMPNN first")
            self._init_random_weights()

    def _init_random_weights(self):
        """Initialize random weights for shape verification."""
        rng = np.random.RandomState(0)
        H = self.HIDDEN_DIM
        in_msg_dim = NODE_FEATURE_DIM + NODE_FEATURE_DIM + EDGE_FEATURE_DIM

        self._weights = {}
        for layer in range(self.N_LAYERS):
            in_dim = NODE_FEATURE_DIM if layer == 0 else H
            msg_in = in_dim + in_dim + EDGE_FEATURE_DIM
            self._weights[f"W_msg_{layer}"] = rng.randn(H, msg_in).astype(np.float32) * 0.1
            self._weights[f"b_msg_{layer}"] = np.zeros(H, dtype=np.float32)
            self._weights[f"W_update_{layer}"] = rng.randn(H, in_dim + H).astype(np.float32) * 0.1
            self._weights[f"b_update_{layer}"] = np.zeros(H, dtype=np.float32)

        # Global readout: concat(mean_pool(H), global_feats) -> H
        self._weights["W_readout"] = rng.randn(H, H + GLOBAL_FEATURE_DIM).astype(np.float32) * 0.1
        self._weights["b_readout"] = np.zeros(H, dtype=np.float32)

        # Output heads (one per output: DR, FNR, TTD)
        for i in range(OUTPUT_DIM):
            self._weights[f"W_out_{i}_0"] = rng.randn(H // 2, H).astype(np.float32) * 0.1
            self._weights[f"b_out_{i}_0"] = np.zeros(H // 2, dtype=np.float32)
            self._weights[f"W_out_{i}_1"] = rng.randn(1, H // 2).astype(np.float32) * 0.1
            self._weights[f"b_out_{i}_1"] = np.zeros(1, dtype=np.float32)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self._weights = {k: data[k] for k in data.files}
        logger.info(f"NumpyMPNN loaded weights from {path} ({len(self._weights)} arrays)")

    def save(self, path: str):
        if self._weights is None:
            raise RuntimeError("No weights to save")
        np.savez(path, **self._weights)
        logger.info(f"NumpyMPNN saved weights to {path}")

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _message_pass(self, H: np.ndarray, edge_index: np.ndarray, edge_feats: np.ndarray, layer: int) -> np.ndarray:
        """One round of message passing. H: (N, h_dim)."""
        W_msg = self._weights[f"W_msg_{layer}"]
        b_msg = self._weights[f"b_msg_{layer}"]
        W_upd = self._weights[f"W_update_{layer}"]
        b_upd = self._weights[f"b_update_{layer}"]

        src, dst = edge_index[0], edge_index[1]
        N = H.shape[0]
        agg = np.zeros((N, W_msg.shape[0]), dtype=np.float32)

        for e in range(edge_index.shape[1]):
            i, j = src[e], dst[e]
            msg_in = np.concatenate([H[i], H[j], edge_feats[e]], axis=0)
            msg = self._relu(W_msg @ msg_in + b_msg)
            agg[j] += msg   # accumulate at destination node

        # Normalize by degree
        degree = np.bincount(dst, minlength=N).astype(np.float32)
        degree = np.where(degree > 0, degree, 1.0)
        agg = agg / degree[:, None]

        # Update nodes: output dimension is W_upd.shape[0] (HIDDEN_DIM)
        new_H = np.zeros((N, W_upd.shape[0]), dtype=np.float32)
        for i in range(N):
            upd_in = np.concatenate([H[i], agg[i]], axis=0)
            new_H_i = self._relu(W_upd @ upd_in + b_upd)
            new_H[i] = new_H_i

        return new_H.astype(np.float32)

    def forward(self, graph: BiosensorGraph) -> np.ndarray:
        """
        Forward pass through the MPNN.

        Returns:
            (OUTPUT_DIM,) array: [DR, FNR, TTD_normalized]
        """
        if self._weights is None:
            raise RuntimeError("No weights loaded")

        H = graph.node_features.copy()   # (N_CHANNELS, NODE_FEATURE_DIM)

        # Expand to HIDDEN_DIM if first layer
        for layer in range(self.N_LAYERS):
            H = self._message_pass(H, graph.edge_index, graph.edge_features, layer)

        # Global pooling: mean of node embeddings
        h_pool = H.mean(axis=0)   # (H,)

        # Concat global features
        h_global = np.concatenate([h_pool, graph.global_features], axis=0)

        # Readout
        W_r = self._weights["W_readout"]
        b_r = self._weights["b_readout"]
        h_read = self._relu(W_r @ h_global + b_r)

        # Per-output heads
        outputs = []
        for i in range(OUTPUT_DIM):
            h = self._relu(self._weights[f"W_out_{i}_0"] @ h_read + self._weights[f"b_out_{i}_0"])
            y = (self._weights[f"W_out_{i}_1"] @ h + self._weights[f"b_out_{i}_1"])[0]
            outputs.append(float(y))

        return np.array(outputs, dtype=np.float32)

    def predict(self, config: dict, scenario: str) -> Tuple[float, float, float]:
        """
        Predict (DR, FNR, TTD) for a config + scenario.

        TTD is returned in seconds [0, 9000].
        """
        graph = config_to_graph(config, scenario)
        y = self.forward(graph)
        dr  = float(np.clip(self._sigmoid(y[0]), 0.0, 1.0))
        fnr = float(np.clip(self._sigmoid(y[1]), 0.0, 1.0))
        ttd_norm = float(np.clip(self._sigmoid(y[2]), 0.0, 1.0))
        ttd = float(np.expm1(ttd_norm * np.log1p(9000.0)))   # denormalize
        return dr, fnr, ttd


# ---------------------------------------------------------------------------
# GNNSurrogate wrapper (auto-detects torch)
# ---------------------------------------------------------------------------

class GNNSurrogate:
    """
    High-level surrogate interface backed by NumpyMPNN (inference).
    For training, use train_gnn.py which requires torch.

    API mirrors SurrogateLoaderV3.predict() for drop-in compatibility.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self._mpnn = NumpyMPNN(weights_path=weights_path)
        self._weights_path = weights_path

    def predict(
        self,
        kd_nm: float,
        sensitivity: float,
        response_time: float,
        biosensor_type: str,
        noise_preset: str,
        scenario: str,
        kd_ctx: float = 0.0,
        kd_p1np: float = 0.0,
        w_ctx: float = 0.0,
        w_p1np: float = 0.0,
    ) -> Tuple[float, float, float]:
        config = {
            "kd_nm": kd_nm,
            "sensitivity": sensitivity,
            "response_time_s": response_time,
            "biosensor_type": biosensor_type,
            "noise_preset": noise_preset,
            "kd_ctx_nm": kd_ctx,
            "kd_p1np_nm": kd_p1np,
            "w_ctx": w_ctx,
            "w_p1np": w_p1np,
        }
        return self._mpnn.predict(config, scenario)

    def load(self, path: str):
        self._mpnn.load(path)
        self._weights_path = path

    @property
    def is_trained(self) -> bool:
        return self._weights_path is not None


# ---------------------------------------------------------------------------
# Torch-based MPNN (training only — imported lazily)
# ---------------------------------------------------------------------------

def build_torch_mpnn(hidden_dim: int = 64, n_layers: int = 3):
    """
    Build a PyTorch MPNN for training.

    Requires: pip install torch

    Architecture exactly mirrors NumpyMPNN so weights can be exported via
    save_numpy() after training:
      - Layer 0: takes NODE_FEATURE_DIM-dimensional node features directly (no input_proj)
      - Layers 1+: take H-dimensional hidden features
      - Readout: concat(mean_pool, global_feats) -> H
      - Per-output heads: H -> H//2 -> 1

    Returns a (model, optimizer) tuple ready for training.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for GNN training.\n"
            "Install: pip install torch\n"
            "Inference only is available via NumpyMPNN without torch."
        )

    H = hidden_dim

    class TorchMPNNLayer(nn.Module):
        def __init__(self, in_dim, h_dim, edge_dim):
            super().__init__()
            self.msg_fn  = nn.Linear(in_dim * 2 + edge_dim, h_dim)
            self.upd_fn  = nn.Linear(in_dim + h_dim, h_dim)
            self.act     = nn.ReLU()

        def forward(self, H, edge_index, edge_feats):
            src, dst = edge_index[0], edge_index[1]
            N = H.shape[0]
            agg = torch.zeros(N, self.msg_fn.out_features, device=H.device)

            msg_in = torch.cat([H[src], H[dst], edge_feats], dim=-1)
            msgs   = self.act(self.msg_fn(msg_in))

            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msgs), msgs)

            degree = torch.zeros(N, device=H.device).scatter_add_(
                0, dst, torch.ones(len(dst), device=H.device)
            ).clamp(min=1.0)
            agg = agg / degree.unsqueeze(-1)

            upd_in = torch.cat([H, agg], dim=-1)
            return self.act(self.upd_fn(upd_in))

    class TorchMPNN(nn.Module):
        def __init__(self):
            super().__init__()
            # No input_proj — layer 0 operates on raw NODE_FEATURE_DIM features,
            # matching NumpyMPNN's weight layout exactly so save_numpy() works.
            self.layers = nn.ModuleList()
            for i in range(n_layers):
                in_d = NODE_FEATURE_DIM if i == 0 else H
                self.layers.append(TorchMPNNLayer(in_d, H, EDGE_FEATURE_DIM))

            self.readout = nn.Sequential(
                nn.Linear(H + GLOBAL_FEATURE_DIM, H),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(H, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
                for _ in range(OUTPUT_DIM)
            ])

        def forward(self, graph: BiosensorGraph):
            H_t = torch.tensor(graph.node_features, dtype=torch.float32)
            e_idx = torch.tensor(graph.edge_index, dtype=torch.long)
            e_feat = torch.tensor(graph.edge_features, dtype=torch.float32)
            g_feat = torch.tensor(graph.global_features, dtype=torch.float32)

            for layer in self.layers:
                H_t = layer(H_t, e_idx, e_feat)

            h_pool = H_t.mean(dim=0)
            h_global = torch.cat([h_pool, g_feat], dim=0)
            h_read = self.readout(h_global)

            return torch.stack([head(h_read).squeeze(-1) for head in self.heads], dim=0)

        def save_numpy(self, path: str):
            """
            Export weights to .npz format readable by NumpyMPNN.

            Layer weight shapes (H=hidden_dim):
              W_msg_0:    (H, NODE_FEATURE_DIM*2 + EDGE_FEATURE_DIM)
              W_update_0: (H, NODE_FEATURE_DIM + H)
              W_msg_k:    (H, H*2 + EDGE_FEATURE_DIM)   for k >= 1
              W_update_k: (H, H*2)                       for k >= 1
              W_readout:  (H, H + GLOBAL_FEATURE_DIM)
              W_out_i_0:  (H//2, H)
              W_out_i_1:  (1, H//2)
            """
            weights = {}
            for layer_idx, layer in enumerate(self.layers):
                weights[f"W_msg_{layer_idx}"]    = layer.msg_fn.weight.detach().numpy()
                weights[f"b_msg_{layer_idx}"]    = layer.msg_fn.bias.detach().numpy()
                weights[f"W_update_{layer_idx}"] = layer.upd_fn.weight.detach().numpy()
                weights[f"b_update_{layer_idx}"] = layer.upd_fn.bias.detach().numpy()
            weights["W_readout"] = self.readout[0].weight.detach().numpy()
            weights["b_readout"] = self.readout[0].bias.detach().numpy()
            for i, head in enumerate(self.heads):
                weights[f"W_out_{i}_0"] = head[0].weight.detach().numpy()
                weights[f"b_out_{i}_0"] = head[0].bias.detach().numpy()
                weights[f"W_out_{i}_1"] = head[2].weight.detach().numpy()
                weights[f"b_out_{i}_1"] = head[2].bias.detach().numpy()
            import numpy as _np
            _np.savez(path, **weights)
            logger.info(f"TorchMPNN weights exported to {path} ({len(weights)} arrays)")

    model = TorchMPNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, optimizer
