#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drug interaction module for GENEVO2 therapeutic objective.

Real PMO/CKD patients are typically on multiple concurrent medications.
Ignoring drug-drug interactions overestimates romosozumab efficacy in
real-world patients. This module provides:

  1. DRUG_INTERACTIONS dict: per-drug efficacy multipliers
  2. apply_drug_interactions(): modifies romosozumab efficacy for a patient
  3. sample_patient_medications(): randomly assigns medications to a virtual
     patient based on published co-prescription rates

Literature basis:
  - Alendronate co-administration: Langdahl 2017 ARCH (concurrent bisphosphonate
    reduces anabolic response by ~20%, exact mechanism unclear).
  - Hormone therapy (HRT): Seifert-Klauss 2012; additive anabolic effect +10%.
  - Thiazide diuretics: raise serum calcium, risk of hypercalcemia with
    SOST-inhibition; indirect efficacy modifier via calcium homeostasis.
  - NSAIDs (CKD patients): worsen prostaglandin-mediated renal perfusion,
    increase eGFR decline risk. Not a direct efficacy modifier but a safety flag.
  - Corticosteroids: glucocorticoid-induced osteoporosis partially opposes
    romosozumab anabolic effect (~15% reduction in net BMD gain, Reid 2020).
  - Dialysis (CKD5D): Uncertain interaction — romosozumab not yet approved in
    this population. Efficacy assumed reduced 30%, high uncertainty.

Usage:
    from BO.evaluation.drug_interactions import apply_drug_interactions, sample_patient_medications

    medications = ["alendronate", "thiazide"]
    efficacy = apply_drug_interactions(base_efficacy=1.0, medications=medications)
    # efficacy ≈ 0.76 (0.80 × 0.95)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-drug efficacy multipliers on romosozumab net BMD gain
# ---------------------------------------------------------------------------
# Values are ESTIMATES from limited clinical evidence.
# Format: drug_name → (efficacy_multiplier, safety_flag, literature_note)
_INTERACTION_TABLE: Dict[str, Tuple[float, str, str]] = {
    "alendronate": (
        0.80,
        "",
        "Langdahl 2017 ARCH: concurrent bisphosphonate -20% anabolic response",
    ),
    "denosumab": (
        0.85,
        "",
        "Bone 2023 consensus: prior anti-resorptive partially blunts anabolic window",
    ),
    "hormone_therapy": (
        1.10,
        "",
        "Seifert-Klauss 2012: additive anabolic effect +10% BMD gain",
    ),
    "thiazide": (
        0.95,
        "hypercalcemia",
        "Thiazides raise serum Ca; indirect modifier via Ca homeostasis",
    ),
    "nsaid": (
        1.00,
        "egfr_risk",
        "NSAIDs: no direct BMD effect but increases eGFR decline risk in CKD",
    ),
    "corticosteroid": (
        0.85,
        "",
        "Reid 2020: glucocorticoid-induced OP partially opposes anabolic effect -15%",
    ),
    "dialysis": (
        0.70,
        "egfr_risk",
        "CKD Stage 5D dialysis: romosozumab not approved, -30% efficacy assumed (high uncertainty)",
    ),
    "calcium_supplement": (
        1.02,
        "",
        "Marginal positive effect: adequate Ca/Vit-D improves mineralization substrate",
    ),
    "vitamin_d": (
        1.02,
        "",
        "Marginal positive effect: adequate Vit-D improves Ca absorption",
    ),
}

# Co-prescription prevalence in PMO/CKD patients (approximate population rates)
_COPRESCRIPTION_RATES: Dict[str, float] = {
    "alendronate":        0.35,   # common in PMO who switch to romosozumab
    "denosumab":          0.10,   # prior biologic therapy
    "hormone_therapy":    0.15,   # declining but still used
    "thiazide":           0.20,   # hypertension comorbidity
    "nsaid":              0.15,   # musculoskeletal pain
    "corticosteroid":     0.10,   # inflammatory comorbidities
    "dialysis":           0.05,   # CKD Stage 5D subgroup
    "calcium_supplement": 0.70,   # standard of care with osteoporosis
    "vitamin_d":          0.65,   # standard of care
}

KNOWN_DRUGS = list(_INTERACTION_TABLE.keys())


def apply_drug_interactions(
    base_efficacy: float,
    medications: List[str],
    *,
    verbose: bool = False,
) -> Tuple[float, List[str]]:
    """
    Compute effective romosozumab efficacy after drug interactions.

    Parameters
    ----------
    base_efficacy : float
        Baseline efficacy fraction (1.0 = full romosozumab effect).
    medications : list of str
        Drug names from KNOWN_DRUGS. Unknown drugs are silently ignored.
    verbose : bool
        Log each interaction applied.

    Returns
    -------
    effective_efficacy : float
        Modified efficacy (product of all interaction multipliers × base).
    safety_flags : list of str
        Active safety concerns (e.g., "hypercalcemia", "egfr_risk").
    """
    efficacy = base_efficacy
    safety_flags: List[str] = []

    for drug in medications:
        drug = drug.lower().strip()
        if drug not in _INTERACTION_TABLE:
            logger.debug("Unknown drug '%s' — no interaction applied.", drug)
            continue
        multiplier, flag, note = _INTERACTION_TABLE[drug]
        efficacy *= multiplier
        if flag:
            safety_flags.append(flag)
        if verbose:
            logger.info("Drug interaction: %-20s mult=%.2f  %s", drug, multiplier, note)

    # Clip to sensible bounds (can't be negative or > 2×)
    efficacy = float(np.clip(efficacy, 0.0, 2.0))
    # Deduplicate flags
    safety_flags = list(set(safety_flags))
    return efficacy, safety_flags


def sample_patient_medications(
    rng: np.random.RandomState | None = None,
    scenario: str = "pmo",
) -> List[str]:
    """
    Randomly sample a patient's medication list based on co-prescription rates.

    Parameters
    ----------
    rng : RandomState, optional
        For reproducibility.
    scenario : str
        'pmo' or 'ckd_mbd' — slight adjustments to rates for CKD patients.

    Returns
    -------
    list of str
        Drug names the patient is currently taking.
    """
    if rng is None:
        rng = np.random.RandomState()

    rates = _COPRESCRIPTION_RATES.copy()
    if scenario == "ckd_mbd":
        # CKD patients: higher NSAID use, dialysis possibility, less HRT
        rates["nsaid"]          = 0.25
        rates["dialysis"]       = 0.15
        rates["hormone_therapy"] = 0.05

    medications = []
    for drug, rate in rates.items():
        if rng.rand() < rate:
            medications.append(drug)

    return medications


def interaction_summary(medications: List[str]) -> str:
    """Return a human-readable summary of interactions for a patient's drug list."""
    if not medications:
        return "No co-medications."
    eff, flags = apply_drug_interactions(1.0, medications)
    lines = [f"Medications: {', '.join(medications)}"]
    lines.append(f"Net efficacy multiplier: {eff:.2f}x")
    if flags:
        lines.append(f"Safety flags: {', '.join(flags)}")
    return "\n".join(lines)
