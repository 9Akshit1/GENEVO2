#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate data_v19: corrected 3-channel dataset.

KEY DIFFERENCES FROM data_v18:
  1. response_time fixed at 600 s (was NaN in v18 → surrogate PI=0).
     All array biosensors operate with a 600-second response window;
     this was always the clinical design constraint, but was never recorded.
  2. Correlated biomarker sampling retained (SOST-CTX r=0.50, CTX-P1NP r=0.45).
  3. Surrogate trained on v19 will learn a non-degenerate log_response_time
     feature, eliminating the PI=0 artefact.

DATASET SPEC:
  - 1500 configs × 4 scenarios = 6000 simulation rows
  - All 3-channel (SOST + CTX + P1NP) via generate_random_array_config()
  - Scenarios: healthy(20%), pmo_mild(25%), pmo(25%), ckd_mbd(30%)
  - Noise preset: realistic (13.6 dB SNR target)
  - Seed: 19

USAGE:
  python simulation/dataset/run_generation_v19.py
  python simulation/dataset/run_generation_v19.py --n-configs 500 --out data_v19_small
  python simulation/dataset/run_generation_v19.py --n-configs 2000 --seed 42 --out data_v19_large
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation.dataset.generator import DatasetGenerator
from simulation.models.biosensors import generate_random_array_config

RESPONSE_TIME_S = 600.0  # fixed — not a search parameter for array biosensors


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate data_v19: corrected 3-channel dataset (response_time fixed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", "--output-dir", dest="output_dir", default="data_v19",
        help="Output directory (default: data_v19)",
    )
    parser.add_argument(
        "--n-configs", dest="n_configs", type=int, default=1500,
        help="Number of biosensor configs to generate (default: 1500)",
    )
    parser.add_argument(
        "--seed", type=int, default=19,
        help="Random seed (default: 19)",
    )
    parser.add_argument(
        "--model", default="simulation/models/bone_environment.ant",
        help="Path to Antimony ODE model (default: simulation/models/bone_environment.ant)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not Path(args.model).exists():
        logger.error("ODE model not found: %s", args.model)
        logger.error("Run from the project root directory.")
        return 1

    np.random.seed(args.seed)

    scenario_dist = {
        "healthy":  0.20,
        "pmo_mild": 0.25,
        "pmo":      0.25,
        "ckd_mbd":  0.30,
    }
    scenarios = list(scenario_dist.keys())
    probs     = list(scenario_dist.values())

    logger.info("=" * 70)
    logger.info("GENEVO2 data_v19 -- corrected dataset (response_time=600s)")
    logger.info("=" * 70)
    logger.info("  Output          : %s", args.output_dir)
    logger.info("  Configs         : %d", args.n_configs)
    logger.info("  Scenarios       : %s", scenario_dist)
    logger.info("  Response time   : %.0f s (FIXED, not NaN)", RESPONSE_TIME_S)
    logger.info("  Correlations    : ENABLED (SOST-CTX r=0.50, CTX-P1NP r=0.45)")
    logger.info("  Seed            : %d", args.seed)
    logger.info("  Expected rows   : ~%d (x4 scenarios per config)", args.n_configs)
    logger.info("")

    gen = DatasetGenerator(
        antimony_model_path=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        sigma_measurement=0.0,
    )

    records = []
    for i in range(args.n_configs):
        cfg  = generate_random_array_config(seed=None)
        scen = np.random.choice(scenarios, p=probs)

        record = gen.generate_single_simulation_instrumented(
            scenario_name=scen,
            biosensor_config=cfg,
            noise_preset="realistic",
            duration=3600.0,
            num_points=361,
            apply_variability=True,
            instrument=True,
        )

        if record is not None:
            records.append(record)

        if (i + 1) % 100 == 0:
            logger.info("  Progress: %d/%d configs done (%d succeeded)",
                        i + 1, args.n_configs, len(records))

    logger.info("  Generation complete: %d/%d succeeded", len(records), args.n_configs)

    # Build master_index.csv
    rows = []
    for r in records:
        bc = r.get("biosensor_config", {})
        m  = r.get("measurement", {})
        rows.append({
            "run_id":              r["run_id"],
            "timestamp":           r["timestamp"],
            "scenario":            r["scenario"],
            "biosensor_type":      bc.get("circuit_type", "array"),
            "noise_preset":        r["noise_preset"],
            "topology":            "3ch",
            "correlated_sampling": True,
            "kd":                  float(bc.get("kd_scl", bc.get("kd", float("nan")))),
            "sensitivity":         float(bc.get("sensitivity", float("nan"))),
            "threshold":           float(bc.get("threshold", float("nan"))),
            "response_time":       RESPONSE_TIME_S,  # v19 FIX: was NaN in v18
            "kd_scl":              float(bc.get("kd_scl", float("nan"))),
            "kd_ctx":              float(bc.get("kd_ctx", float("nan"))),
            "kd_p1np":             float(bc.get("kd_p1np", float("nan"))),
            "w_scl":               float(bc.get("w_scl", float("nan"))),
            "w_ctx":               float(bc.get("w_ctx", 0.0)),
            "w_p1np":              float(bc.get("w_p1np", float("nan"))),
            "snr_db":              float(m.get("snr_db", 0.0)),
            "n_detections":        int(m.get("n_detections", 0)),
            "detection_rate":      float(m.get("detection_rate", 0.0)),
            "time_to_detection":   float(m.get("time_to_detection", 9000.0)),
            "false_negative_rate": float(m.get("false_negative_rate", 1.0)),
            "sclerostin_mean":     r.get("sclerostin_mean", float("nan")),
            "sclerostin_std":      r.get("sclerostin_std", float("nan")),
            "ctx_mean":            r.get("ctx_mean", float("nan")),
            "p1np_mean":           r.get("p1np_mean", float("nan")),
            "metadata_file":       r["metadata_file"],
            "timeseries_file":     r["timeseries_file"],
        })

    df = pd.DataFrame(rows)
    out_path = Path(args.output_dir) / "master_index.csv"
    df.to_csv(out_path, index=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE -- data_v19 generated")
    logger.info("=" * 70)
    logger.info("  Total rows       : %d", len(df))
    logger.info("  Scenario counts  :")
    for scen, count in df["scenario"].value_counts().items():
        logger.info("    %-12s : %d", scen, count)
    logger.info("  DR mean (pmo)    : %.3f",
                df[df["scenario"] == "pmo"]["detection_rate"].mean())
    logger.info("  DR mean (ckd)    : %.3f",
                df[df["scenario"] == "ckd_mbd"]["detection_rate"].mean())
    logger.info("  response_time    : %.0f s (all rows, fixed)",
                df["response_time"].mean())
    logger.info("  Output           : %s", out_path.resolve())
    logger.info("")
    logger.info("Next steps:")
    logger.info("  python BO/core/build_surrogates_v3.py --data-dir data_v19")
    logger.info("  python BO/bo_main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
