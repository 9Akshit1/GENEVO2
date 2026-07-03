#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GENEVO2 Unified Dataset Generation Entry Point

Generates simulation datasets for surrogate training.

Current gold standard: data_v18
  - 1500 configs x 4 scenarios = 6000 simulation rows
  - 3-channel array biosensor (SOST + CTX + P1NP)
  - Correlated biomarker sampling (Cholesky factorization)
  - Noise preset: realistic (13.6 dB SNR target)
  - Seed: 18

Usage:
  # Generate standard data_v18 (gold standard, ~20 min)
  python simulation/run_dataset.py

  # Custom size/output
    python simulation/run_dataset.py --n-configs 500 --out data_v18_small
    python simulation/run_dataset.py --n-configs 500 --out-dir data_v18_small

  # Large dataset for retraining
    python simulation/run_dataset.py --n-configs 3000 --seed 99 --out data_v19
    python simulation/run_dataset.py --n-configs 3000 --seed 99 --out-dir data_v19

  # Regenerate patient cohort (n=1000 synthetic patients)
  python simulation/run_dataset.py --mode cohort
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def run_dataset_generation(args) -> int:
    """Generate simulation data using dataset.generator.DatasetGenerator."""
    import logging
    import numpy as np
    import pandas as pd

    from simulation.dataset.generator import DatasetGenerator
    from simulation.models.biosensors import generate_random_array_config

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("run_dataset")

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("ODE model not found: %s", model_path)
        return 1

    gen = DatasetGenerator(
        antimony_model_path=str(model_path),
        output_dir=args.out,
        seed=args.seed,
        sigma_measurement=0.0,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scenario distribution matching data_v18 spec
    scenario_dist = {
        "healthy":  int(args.n_configs * 0.20),
        "pmo_mild": int(args.n_configs * 0.25),
        "pmo":      int(args.n_configs * 0.25),
        "ckd_mbd":  int(args.n_configs * 0.30),
    }
    # Ensure exact total
    total_assigned = sum(scenario_dist.values())
    diff = args.n_configs - total_assigned
    scenario_dist["ckd_mbd"] += diff

    logger.info("=" * 70)
    logger.info("GENEVO2 Dataset Generation")
    logger.info("  n_configs: %d  seed: %d  out: %s", args.n_configs, args.seed, out_dir)
    logger.info("  Scenario distribution: %s", scenario_dist)
    logger.info("=" * 70)

    records = []
    config_count = 0

    for scenario, n_cfg in scenario_dist.items():
        logger.info("  Generating %d configs for scenario: %s", n_cfg, scenario)
        for _ in range(n_cfg):
            cfg = generate_random_array_config(seed=None)
            cfg["target_scenario"] = scenario
            cfg["noise_preset"] = "realistic"

            record = gen.generate_single_simulation_instrumented(
                scenario_name=scenario,
                biosensor_config=cfg,
                noise_preset="realistic",
                duration=3600.0,
                num_points=361,
                apply_variability=True,
                instrument=True,
            )

            if record is not None:
                records.append(record)

            config_count += 1
            if config_count % 100 == 0:
                logger.info("    %d/%d configs done", config_count, args.n_configs)

    rows = []
    for r in records:
        bc = r.get("biosensor_config", {})
        m = r.get("measurement", {})
        rows.append({
            "run_id": r["run_id"],
            "timestamp": r["timestamp"],
            "scenario": r["scenario"],
            "biosensor_type": bc.get("circuit_type", "array"),
            "noise_preset": r["noise_preset"],
            "topology": "3ch",
            "correlated_sampling": True,
            "kd": float(bc.get("kd_scl", bc.get("kd", float("nan")))),
            "sensitivity": float(bc.get("sensitivity", float("nan"))),
            "threshold": float(bc.get("threshold", float("nan"))),
            "response_time": float(
                bc.get("response_time", float("nan"))
                if bc.get("circuit_type") == "amplifying"
                else float("nan")
            ),
            "kd_scl": float(bc.get("kd_scl", float("nan"))),
            "kd_ctx": float(bc.get("kd_ctx", float("nan"))),
            "kd_p1np": float(bc.get("kd_p1np", float("nan"))),
            "w_scl": float(bc.get("w_scl", float("nan"))),
            "w_ctx": float(bc.get("w_ctx", float("nan"))),
            "w_p1np": float(bc.get("w_p1np", float("nan"))),
            "snr_db": float(m.get("snr_db", 0.0)),
            "n_detections": int(m.get("n_detections", 0)),
            "detection_rate": float(m.get("detection_rate", 0.0)),
            "time_to_detection": float(m.get("time_to_detection", 0.0)),
            "false_negative_rate": float(m.get("false_negative_rate", 0.0)),
            "sclerostin_mean": float(r.get("sclerostin_mean", float("nan"))),
            "sclerostin_std": float(r.get("sclerostin_std", float("nan"))),
            "ctx_mean": float(r.get("ctx_mean", float("nan"))),
            "p1np_mean": float(r.get("p1np_mean", float("nan"))),
            "metadata_file": r["metadata_file"],
            "timeseries_file": r["timeseries_file"],
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "master_index.csv"
    df.to_csv(csv_path, index=False)
    logger.info("\n[OK] Generated %d rows -> %s", len(df), csv_path)
    return 0


def run_cohort_generation(args) -> int:
    """Generate synthetic patient cohort (n=1000)."""
    import subprocess
    cmd = [sys.executable, str(_ROOT / "data" / "synthetic_patient_cohort.py")]
    return subprocess.run(cmd).returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description="GENEVO2 dataset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["simulate", "cohort"], default="simulate",
                   help="Generate simulation data (simulate) or patient cohort (cohort)")
    p.add_argument("--n-configs", type=int, default=1500,
                   help="Number of biosensor configs (default: 1500 = data_v18 standard)")
    p.add_argument("--seed", type=int, default=18,
                   help="Random seed (default: 18 = data_v18 seed)")
    p.add_argument("--model", default="simulation/models/bone_environment.ant",
                   help="Path to Antimony ODE model (default: simulation/models/bone_environment.ant)")
    p.add_argument("--out", "--out-dir", dest="out", default="data_v18",
                   help="Output directory (default: data_v18)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    if args.mode == "cohort":
        return run_cohort_generation(args)
    else:
        return run_dataset_generation(args)


if __name__ == "__main__":
    sys.exit(main())
