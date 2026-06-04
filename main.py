"""
Main execution script for bone biosensor simulation dataset generation.

This script orchestrates the complete data generation pipeline:
1. Creates Antimony model file (v2.0 — corrected)
2. Initialises dataset generator
3. Runs N simulations with realistic parameters
4. Generates master index
5. (Optional) Runs validation

Features:
- UTF-8 encoding support (Windows compatible)
- Comprehensive logging
- Command-line argument parsing
- Error handling and recovery

CORRECTIONS IN v2.0 (applied to the embedded Antimony model):
  - Removed `at` events that permanently corrupted k_prod parameters
  - Fixed diffusion rate laws (25× amplification, not 250×)
  - PTH now inhibits sclerostin (biologically correct direction)
  - K_RANKL_OC corrected 50→0.3 nM; K_OPG_inhib corrected 100→5.0 nM
  - Estrogen/PTH declared as boundary species ($)
  - All k_prod values recalibrated for OC_ss=4.0, OB_ss=2.0
  - Sensor ICs set to diffusion-equilibrium values (25×/30×)
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import codecs

# CRITICAL: Set UTF-8 encoding for Python I/O (Windows compatibility)
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from utils.logging_config import setup_logging
from dataset.generator import DatasetGenerator

logger = None  # Initialised in main()


# ============================================================================
# ANTIMONY MODEL — embedded here so that running main.py always writes the
# CORRECT model file, regardless of what is already on disk.
# The model file at models/bone_environment.ant must match this string exactly.
# ============================================================================
_ANTIMONY_MODEL = """// Bone Microenvironment Model - Version 2.0
// Sclerostin-focused dynamic model for biosensor simulation
//
// FIXES from v2.0:
//   1. Replaced invalid boundary-species syntax:
//        OLD (invalid in Antimony):
//          Estrogen is boundary;
//          PTH is boundary;
//        NEW (correct):
//          species $Estrogen in bone;
//          species $PTH in bone;
//
//      The old syntax caused the parser error:
//      "syntax error, unexpected '', expecting '.' or 'is'"
//      because Antimony does NOT support standalone:
//          X is boundary;
//      declarations.
//
//   2. Replaced '^' exponentiation with pow(x,y)
//      Antimony parser is much more reliable with pow()
//      especially in multiline kinetic laws.
//
//   3. Slight formatting cleanup for multiline reactions
//      to avoid parser ambiguity.
//
// CORRECTIONS from v1.0:
//   1. Estrogen and PTH declared as boundary species ($) — they have no
//      production/degradation reactions; this is the correct SBML semantics
//   2. Removed all `at` events — those one-shot events permanently modify
//      k_prod parameters and are NOT reset by roadrunner.reset(), corrupting
//      every subsequent simulation run
//   3. Fixed diffusion rate laws — old formula gave 250x amplification;
//      correct formula gives 25x (Sclerostin) and 30x (RANKL/OPG)
//   4. PTH now mildly INHIBITS sclerostin (biologically correct;
//      K_pth_Scl=300 so effect is 5percent healthy, 32% CKD-MBD)
//   5. K_RANKL_OC corrected 50 to 0.3 nM (RANKL range is 0.5-0.9 nM)
//   6. K_OPG_inhib corrected 100 to 5.0 nM (OPG range is 4-5.5 nM)
//   7. Initial conditions set to near-equilibrium (OB=2.0, OC=4.0, OC_oc=0.3)
//   8. Sensor ICs set to diffusion-equilibrium values (25x bone for Sclerostin)
//
// Diffusion equilibrium derivation (CORRECTED):
//   Forward rate_law = k_diff * C_bone * bone    (amount/time)
//   Back    rate_law = k_back * C_sensor * sensor_chamber  (amount/time)
//
//   d(C_sensor)/dt = k_diff*C_bone*bone/sensor_chamber - k_back*C_sensor
//   At equilibrium: C_sensor/C_bone = k_diff*bone / (k_back*sensor_chamber)
//     Sclerostin: 0.05*1.0 / (0.02*0.1) = 25x  [OK]
//     RANKL/OPG:  0.03*1.0 / (0.01*0.1) = 30x  [OK]
//
// Calibration: all k_prod values calibrated so that the system is at
// near-equilibrium from t=0 when env_configs apply their overrides.
// (OB_ss=2.0, OC_ss=4.0; K_scl_OB=5.0 >> C_bone, so no active Wnt feedback)
//
// References:
//   Baron & Kneissel (2013) Nat Rev Endocrinol 9:531
//   Boyce & Xing (2008) J Bone Miner Res 23:1838
//   Ardawi et al. (2011) JCEM 96:3867
//   Loots et al. (2005) JBC 280:26
//   Hofbauer et al. (1999) Endocrinology 140:4367
//
// Units: time=seconds, concentrations=nM

model BoneEnvironment()

  // --------------------------------------------------------------------------------------------
  // COMPARTMENTS
  // --------------------------------------------------------------------------------------------

  compartment bone = 1.0;
  compartment sensor_chamber = 0.1;

  // --------------------------------------------------------------------------------------------
  // SPECIES — bone compartment
  // --------------------------------------------------------------------------------------------

  species Sclerostin_bone in bone;
  species RANKL_bone      in bone;
  species OPG_bone        in bone;

  species Osteocytes      in bone;
  species Osteoblasts     in bone;
  species Osteoclasts     in bone;

  species MineralIon      in bone;

  // Boundary species (externally controlled endocrine signals)
  species $Estrogen in bone;
  species $PTH in bone;

  // --------------------------------------------------------------------------------------------
  // SPECIES — sensor compartment
  // --------------------------------------------------------------------------------------------

  species Sclerostin_sensor in sensor_chamber;
  species RANKL_sensor      in sensor_chamber;
  species OPG_sensor        in sensor_chamber;

  // --------------------------------------------------------------------------------------------
  // KINETIC PARAMETERS
  // --------------------------------------------------------------------------------------------

  // Sclerostin kinetics
  // Healthy calibration:
  //   OC_ss=4.0
  //   E_factor=0.200
  //   P_factor=0.945
  // Result:
  //   k_prod = 1.389e-6

  k_prod_Scl = 0.000001389;
  k_deg_Scl  = 0.00007;

  // RANKL kinetics
  // Healthy calibration:
  //   OB_ss=2.0
  //   RANKL_factor=1.165

  k_prod_RANKL = 0.00000429;
  k_deg_RANKL  = 0.00002;

  // OPG kinetics
  // Healthy calibration:
  //   OB_ss=2.0
  //   OPG_factor=1.058

  k_prod_OPG = 0.0000472;
  k_deg_OPG  = 0.00002;

  // Cellular turnover

  k_prod_Osteocyte  = 0.001;
  k_death_Osteocyte = 0.0005;

  k_diff_Osteoblast = 0.01;
  k_apop_Osteoblast = 0.005;

  k_diff_Osteoclast = 0.008;
  k_apop_Osteoclast = 0.01;

  // Mineral dynamics

  k_prod_Mineral = 0.087;
  k_loss_Mineral = 0.025;

  // Diffusion parameters

  k_diff_Scl = 0.05;
  k_back_Scl = 0.02;

  k_diff_RANKL = 0.03;
  k_back_RANKL = 0.01;

  k_diff_OPG = 0.03;
  k_back_OPG = 0.01;

  // --------------------------------------------------------------------------------------------
  // REGULATORY PARAMETERS
  // --------------------------------------------------------------------------------------------

  // Estrogen  to  Sclerostin inhibition

  K_est_Scl = 0.5;
  n_est_Scl = 2.0;

  // PTH  to  Sclerostin mild inhibition

  K_pth_Scl = 300.0;
  n_pth_Scl = 1.5;

  // Estrogen  to  RANKL inhibition / OPG stimulation

  K_est_RANKL = 0.3;
  n_est_RANKL = 2.0;

  // PTH  to  OPG inhibition

  K_pth_OPG = 80.0;
  n_pth_OPG = 1.0;

  // Osteoclast regulation

  K_RANKL_OC  = 0.3;
  n_RANKL     = 2.0;

  K_OPG_inhib = 5.0;

  // Sclerostin  to  Osteoblast inhibition

  K_scl_OB = 5.0;

  // --------------------------------------------------------------------------------------------
  // INITIAL CONDITIONS
  // --------------------------------------------------------------------------------------------

  Sclerostin_bone   = 0.015;
  Sclerostin_sensor = 0.375;

  RANKL_bone        = 0.5;
  RANKL_sensor      = 15.0;

  OPG_bone          = 5.0;
  OPG_sensor        = 150.0;

  Osteocytes  = 4.0;
  Osteoblasts = 2.0;
  Osteoclasts = 0.3;

  Estrogen = 1.0;
  PTH      = 45.0;

  MineralIon = 2.5;

  // --------------------------------------------------------------------------------------------
  // REACTIONS
  // --------------------------------------------------------------------------------------------

  // Sclerostin dynamics

  R_prod_Scl: => Sclerostin_bone; k_prod_Scl * Osteocytes * (1.0 / (1.0 + pow(Estrogen / K_est_Scl, n_est_Scl))) * (1.0 / (1.0 + pow(PTH / K_pth_Scl, n_pth_Scl)));

  R_deg_Scl: Sclerostin_bone => ; k_deg_Scl * Sclerostin_bone;

  // RANKL dynamics

  R_prod_RANKL: => RANKL_bone; k_prod_RANKL * Osteoblasts * (1.0 + 2.0 / (1.0 + pow(Estrogen / K_est_RANKL, n_est_RANKL)));

  R_deg_RANKL: RANKL_bone => ; k_deg_RANKL * RANKL_bone;

  // OPG dynamics

  R_prod_OPG: => OPG_bone; k_prod_OPG * Osteoblasts * (0.5 + 1.5 * (Estrogen / (Estrogen + K_est_RANKL))) * (1.0 / (1.0 + pow(PTH / K_pth_OPG, n_pth_OPG)));

  R_deg_OPG: OPG_bone => ; k_deg_OPG * OPG_bone;

  // Osteocyte dynamics

  R_prod_Osteocyte: => Osteocytes; k_prod_Osteocyte * Osteoblasts;

  R_death_Osteocyte: Osteocytes => ; k_death_Osteocyte * Osteocytes;

  // Osteoblast dynamics

  R_diff_Osteoblast: => Osteoblasts; k_diff_Osteoblast * (1.0 / (1.0 + Sclerostin_bone / K_scl_OB));

  R_apop_Osteoblast: Osteoblasts => ; k_apop_Osteoblast * Osteoblasts;

  // Osteoclast dynamics

  R_diff_Osteoclast: => Osteoclasts; k_diff_Osteoclast * (pow(RANKL_bone, n_RANKL) / (pow(K_RANKL_OC, n_RANKL) + pow(RANKL_bone, n_RANKL))) * (1.0 / (1.0 + OPG_bone / K_OPG_inhib));

  R_apop_Osteoclast: Osteoclasts => ; k_apop_Osteoclast * Osteoclasts;

  // Mineral dynamics

  R_prod_Mineral: => MineralIon; k_prod_Mineral * (1.0 + PTH / 100.0);

  R_loss_Mineral: MineralIon => ; k_loss_Mineral * MineralIon * Osteoblasts;

  // Diffusion dynamics

  R_diff_Scl: Sclerostin_bone => Sclerostin_sensor; k_diff_Scl * Sclerostin_bone * bone;

  R_back_Scl: Sclerostin_sensor => Sclerostin_bone; k_back_Scl * Sclerostin_sensor * sensor_chamber;

  R_diff_RANKL: RANKL_bone => RANKL_sensor; k_diff_RANKL * RANKL_bone * bone;

  R_back_RANKL: RANKL_sensor => RANKL_bone; k_back_RANKL * RANKL_sensor * sensor_chamber;

  R_diff_OPG: OPG_bone => OPG_sensor; k_diff_OPG * OPG_bone * bone;

  R_back_OPG: OPG_sensor => OPG_bone; k_back_OPG * OPG_sensor * sensor_chamber;

end
"""


def create_antimony_model() -> str:
    """
    Write the corrected Antimony model to models/bone_environment.ant.

    IMPORTANT: This function always (over)writes the file so that any
    stale v1.0 model on disk is replaced with the corrected v2.0 model.

    References:
      Baron & Kneissel (2013) Nat Rev Endocrinol 9:531
      Boyce & Xing (2008) J Bone Miner Res 23:1838
      Ardawi et al. (2011) JCEM 96:3867
      Loots et al. (2005) JBC 280:26
      Hofbauer et al. (1999) Endocrinology 140:4367

    Returns:
        Path to the written .ant file.
    """
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'bone_environment.ant'

    try:
        with codecs.open(model_path, "w", encoding="utf-8") as f:
            f.write(_ANTIMONY_MODEL)
        return str(model_path)
    except IOError as e:
        if logger:
            logger.error(f"Cannot write Antimony model: {e}")
        raise


def main():
    """Main execution function."""
    global logger

    parser = argparse.ArgumentParser(
        description='Generate bone biosensor simulation dataset'
    )
    parser.add_argument('--n_simulations', type=int,   default=1000,
                        help='Number of simulations to generate (default: 1000)')
    parser.add_argument('--output_dir',    type=str,   default='data',
                        help='Output directory for dataset (default: data)')
    parser.add_argument('--duration',      type=float, default=3600.0,
                        help='Simulation duration in seconds (default: 3600)')
    parser.add_argument('--num_points',    type=int,   default=361,
                        help='Number of time points (default: 361)')
    parser.add_argument('--log_level',     type=str,   default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    logger = setup_logging(log_level=args.log_level)

    logger.info("=" * 80)
    logger.info("Bone Biosensor Simulation Framework v2.0")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Simulations : {args.n_simulations}")
    logger.info(f"  Duration    : {args.duration}s")
    logger.info(f"  Points      : {args.num_points}")
    logger.info(f"  Output dir  : {args.output_dir}")
    logger.info(f"  Seed        : {args.seed}")
    logger.info("=" * 80)

    try:
        # ── Step 1: Write corrected Antimony model ─────────────────────────
        logger.info("[STEP 1] Writing Antimony model (v2.0)...")
        model_path = create_antimony_model()
        logger.info(f"  OK: {model_path}")

        # ── Step 2: Initialise dataset generator ───────────────────────────
        logger.info("[STEP 2] Initialising dataset generator...")
        generator = DatasetGenerator(
            antimony_model_path=model_path,
            output_dir=args.output_dir,
            seed=args.seed
        )
        logger.info("  OK: Generator ready")

        # ── Step 3: Generate dataset ────────────────────────────────────────
        logger.info("[STEP 3] Starting dataset generation...")
        master_df = generator.generate_dataset(
            n_simulations=args.n_simulations,
            duration=args.duration,
            num_points=args.num_points
        )

        # ── Summary ────────────────────────────────────────────────────────
        logger.info("=" * 80)
        logger.info("Dataset Generation Complete!")
        logger.info("=" * 80)
        logger.info(f"Total simulations: {len(master_df)}")

        for col_label, col_name in [
            ("Scenario distribution",     "scenario"),
            ("Biosensor type",            "biosensor_type"),
            ("Noise preset",              "noise_preset"),
        ]:
            if col_name in master_df.columns:
                logger.info(f"\n{col_label}:")
                logger.info(master_df[col_name].value_counts().to_string())

        for stat_label, col_name in [
            ("SNR (dB)",              "snr_db"),
            ("Time-to-Detection (s)", "time_to_detection"),
            ("False Negative Rate",   "false_negative_rate"),
            ("Sclerostin mean (nM)",  "sclerostin_mean"),
            ("Sclerostin std (nM)",   "sclerostin_std"),
        ]:
            if col_name in master_df.columns:
                logger.info(f"\n{stat_label}:")
                logger.info(master_df[col_name].describe().to_string())

        logger.info("=" * 80)
        logger.info("SUMMARY:")
        logger.info(f"  Simulations : {len(master_df)}")
        logger.info(f"  Scenarios   : {master_df['scenario'].nunique()} types")
        logger.info(f"  Biosensors  : {master_df['biosensor_type'].nunique()} types")
        logger.info(f"  Noise levels: {master_df['noise_preset'].nunique()} types")
        logger.info(f"  Output      : {args.output_dir}/master_index.csv")
        logger.info("=" * 80)
        logger.info("\nNEXT STEPS:")
        logger.info(f"  Validate : python sim_validator.py --input_dir {args.output_dir}")
        logger.info(f"  Train RL : python rl_model.py --input_dir {args.output_dir}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())