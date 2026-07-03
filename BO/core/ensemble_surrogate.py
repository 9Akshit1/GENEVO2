#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble surrogate: GBM (SurrogateLoaderV3) + GNN (NumpyMPNN).

Predictions are weighted by each model's held-out rank-rho:
  - GBM baseline: rank-rho = 0.517
  - GNN: rank-rho from training_history.json (falls back to 0.517 if not available)

Weights are normalized so they sum to 1.0.

API: drop-in replacement for SurrogateLoaderV3.predict().

Usage:
    from core.ensemble_surrogate import EnsembleSurrogate
    ens = EnsembleSurrogate(surrogate_dir="BO/bo_results", gnn_weights="BO/surrogates/gnn/weights/gnn_best.npz")
    dr, fnr, ttd = ens.predict(kd_nm=1.0, sensitivity=3.0, ...)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_GBM_RANK_RHO = 0.517   # established baseline


class EnsembleSurrogate:
    """
    Weighted ensemble of GBM (SurrogateLoaderV3) and GNN (NumpyMPNN) surrogates.

    If GNN weights are not available the ensemble transparently falls back to
    GBM-only predictions (weight_gnn=0).
    """

    def __init__(
        self,
        surrogate_dir: str,
        gnn_weights: Optional[str] = None,
        gnn_history: Optional[str] = None,
        gbm_rho: float = _GBM_RANK_RHO,
    ):
        from .surrogate_loader import SurrogateLoaderV3
        self._gbm = SurrogateLoaderV3(surrogate_dir)
        self._gnn = None
        self._gnn_rho = 0.0

        # Load GNN if weights provided
        if gnn_weights and Path(gnn_weights).exists():
            try:
                import sys, os
                sys.path.insert(0, str(Path(__file__).parent.parent / "surrogates" / "gnn"))
                from gnn_surrogate import NumpyMPNN
                self._gnn = NumpyMPNN(weights_path=gnn_weights)

                # Infer GNN rank-rho from training history
                if gnn_history and Path(gnn_history).exists():
                    with open(gnn_history) as f:
                        hist = json.load(f)
                    self._gnn_rho = float(hist.get("val_dr_rank_rho") or 0.0)
                else:
                    # Try default location
                    default_hist = Path(gnn_weights).parent / "training_history.json"
                    if default_hist.exists():
                        with open(default_hist) as f:
                            hist = json.load(f)
                        self._gnn_rho = float(hist.get("val_dr_rank_rho") or 0.0)
                    else:
                        self._gnn_rho = gbm_rho   # assume parity if history missing

                logger.info(f"EnsembleSurrogate: GNN loaded (rank-rho={self._gnn_rho:.3f})")
            except Exception as e:
                logger.warning(f"EnsembleSurrogate: GNN load failed ({e}), using GBM only")
                self._gnn = None
                self._gnn_rho = 0.0
        else:
            logger.info("EnsembleSurrogate: GNN weights not found — GBM only")

        self._gbm_rho = gbm_rho
        self._update_weights()

    def _update_weights(self):
        total = self._gbm_rho + self._gnn_rho
        if total <= 0 or self._gnn is None:
            self._w_gbm = 1.0
            self._w_gnn = 0.0
        else:
            self._w_gbm = self._gbm_rho / total
            self._w_gnn = self._gnn_rho / total
        logger.info(
            f"EnsembleSurrogate weights: GBM={self._w_gbm:.3f} "
            f"(rho={self._gbm_rho:.3f}), GNN={self._w_gnn:.3f} (rho={self._gnn_rho:.3f})"
        )

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
        """Return (DR, FNR, TTD) as weighted ensemble of GBM and GNN."""
        dr_gbm, fnr_gbm, ttd_gbm = self._gbm.predict(
            kd_nm=kd_nm, sensitivity=sensitivity, response_time=response_time,
            biosensor_type=biosensor_type, noise_preset=noise_preset, scenario=scenario,
            kd_ctx=kd_ctx, kd_p1np=kd_p1np, w_ctx=w_ctx, w_p1np=w_p1np,
        )

        if self._gnn is None or self._w_gnn == 0.0:
            return dr_gbm, fnr_gbm, ttd_gbm

        try:
            config = {
                "kd_nm": kd_nm, "sensitivity": sensitivity,
                "response_time_s": response_time, "biosensor_type": biosensor_type,
                "noise_preset": noise_preset, "kd_ctx_nm": kd_ctx,
                "kd_p1np_nm": kd_p1np, "w_ctx": w_ctx, "w_p1np": w_p1np,
            }
            dr_gnn, fnr_gnn, ttd_gnn = self._gnn.predict(config, scenario)

            dr  = self._w_gbm * dr_gbm  + self._w_gnn * dr_gnn
            fnr = self._w_gbm * fnr_gbm + self._w_gnn * fnr_gnn
            ttd = self._w_gbm * ttd_gbm + self._w_gnn * ttd_gnn
            return float(dr), float(fnr), float(ttd)
        except Exception as e:
            logger.warning(f"EnsembleSurrogate GNN inference failed ({e}), falling back to GBM")
            return dr_gbm, fnr_gbm, ttd_gbm

    @property
    def using_gnn(self) -> bool:
        return self._gnn is not None and self._w_gnn > 0

    def summary(self) -> str:
        if self.using_gnn:
            return (f"EnsembleSurrogate: GBM(rho={self._gbm_rho:.3f}, w={self._w_gbm:.2f}) + "
                    f"GNN(rho={self._gnn_rho:.3f}, w={self._w_gnn:.2f})")
        return f"EnsembleSurrogate: GBM only (rho={self._gbm_rho:.3f}) — GNN not loaded"
