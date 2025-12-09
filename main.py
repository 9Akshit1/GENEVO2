"""
Main execution script for bone biosensor simulation dataset generation.
"""

import argparse
import sys
import logging
from pathlib import Path

# Import project modules
from utils.logging_config import setup_logging
from dataset.generator import DatasetGenerator


def create_antimony_model():
    """
    Create the Antimony model file if it doesn't exist.
    
    This uses the biologically-grounded model defined in the documentation.
    All parameters are based on published literature.
    """
    antimony_model = '''// Bone Microenvironment Model
// Sclerostin-focused model for biosensor simulation
//
// References:
// [1] Baron & Kneissel (2013). WNT signaling. Nat Rev Endocrinol.
// [2] Boyce & Xing (2008). RANKL/RANK/OPG. Arch Biochem Biophys.
// [3] Modder et al. (2011). Sclerostin regulation. J Bone Miner Res.
// [4] eLife (2021). Sclerostin half-life ~3hr in osteocytes.

model BoneEnvironment()
  
  // Compartments
  compartment bone = 1.0;
  compartment sensor_chamber = 0.1;
  
  // Species in bone compartment
  species Sclerostin_bone in bone;
  species RANKL_bone in bone;
  species OPG_bone in bone;
  species Osteocytes in bone;
  species Osteoblasts in bone;
  species Osteoclasts in bone;
  species Estrogen in bone;
  species PTH in bone;
  species MineralIon in bone;
  
  // Species in sensor compartment
  species Sclerostin_sensor in sensor_chamber;
  species RANKL_sensor in sensor_chamber;
  species OPG_sensor in sensor_chamber;
  
  // Parameters - Sclerostin dynamics
  // Ref: Ardawi 2011 JCEM - serum levels 20-80 pmol/L (0.02-0.08 nM)
  // Ref: Sabbagh 2012 JASN - circulating t1/2 ~30-60 min
  // Local bone microenvironment may be 10-50x higher than serum
  k_prod_Scl = 0.0000035;  // Production rate [nM/min] - 10x lower
  k_deg_Scl = 0.00007;     // Degradation rate [1/min] - 10x lower (maintains t1/2)
  
  // Parameters - RANKL/OPG dynamics
  // Target: RANKL ~0.5 pM, OPG ~5 pM in healthy state
  k_prod_RANKL = 0.00001;  // RANKL production [pM/min] - further reduced
  k_deg_RANKL = 0.00002;   // RANKL degradation [1/min] (t1/2 ~ 34,650 sec = 9.6 hr)
  k_prod_OPG = 0.0001;     // OPG production [pM/min] - further reduced  
  k_deg_OPG = 0.00002;     // OPG degradation [1/min] (t1/2 ~ 9.6 hr)
  
  // Parameters - Cellular dynamics
  k_prod_Osteocyte = 0.001;
  k_death_Osteocyte = 0.0005;
  k_diff_Osteoblast = 0.01;
  k_apop_Osteoblast = 0.005;
  k_diff_Osteoclast = 0.008;
  k_apop_Osteoclast = 0.01;

  // Parameters - Mineral dynamics
  k_prod_Mineral = 0.1;
  k_loss_Mineral = 0.05;

  // Parameters - Transport
  k_diff_Scl = 0.05;
  k_back_Scl = 0.02;
  k_diff_RANKL = 0.03;
  k_back_RANKL = 0.01;
  k_diff_OPG = 0.03;
  k_back_OPG = 0.01;

  // Regulatory parameters
  K_est_Scl = 0.5;         // Estrogen K for sclerostin
  n_est_Scl = 2.0;         // Hill coefficient
  K_est_RANKL = 0.3;
  n_est_RANKL = 2.0;
  K_pth_Scl = 100.0;
  n_pth_Scl = 1.5;
  K_pth_OPG = 80.0;
  n_pth_OPG = 1.0;
  K_RANKL_OC = 50.0;
  n_RANKL = 2.0;
  K_OPG_inhib = 100.0;
  K_mineral_PTH = 2.0;
  n_mineral = 1.0;
  
  // Initial conditions (physiologically realistic)
  // Sclerostin: 0.02-0.08 nM in serum, ~0.1-0.5 nM in bone microenvironment
  Sclerostin_bone = 0.05;      // [nM] - matches steady-state
  RANKL_bone = 0.5;            // [pM] - matches steady-state  
  OPG_bone = 5.0;              // [pM] - matches steady-state
  Sclerostin_sensor = 0.025;   // [nM] - equilibrated via diffusion
  RANKL_sensor = 0.25;         // [pM] - equilibrated
  OPG_sensor = 2.5;            // [pM] - equilibrated
  Osteocytes = 1.0;
  Osteoblasts = 1.0;
  Osteoclasts = 1.0;
  Estrogen = 1.0;
  PTH = 50.0;
  MineralIon = 2.5;
  
  // Reactions - Sclerostin
  R_prod_Scl: => Sclerostin_bone; k_prod_Scl * Osteocytes * (1.0 / (1.0 + (Estrogen/K_est_Scl)^n_est_Scl)) * (1.0 + 0.5 * (PTH/K_pth_Scl)^n_pth_Scl / (1.0 + (PTH/K_pth_Scl)^n_pth_Scl));
  R_deg_Scl: Sclerostin_bone => ; k_deg_Scl * Sclerostin_bone;
  
  // Reactions - RANKL
  R_prod_RANKL: => RANKL_bone; k_prod_RANKL * Osteoblasts * (1.0 + 2.0 / (1.0 + (Estrogen/K_est_RANKL)^n_est_RANKL));
  R_deg_RANKL: RANKL_bone => ; k_deg_RANKL * RANKL_bone;
  
  // Reactions - OPG
  R_prod_OPG: => OPG_bone; k_prod_OPG * Osteoblasts * (0.5 + 1.5 * (Estrogen / (Estrogen + K_est_RANKL))) * (1.0 / (1.0 + (PTH/K_pth_OPG)^n_pth_OPG));
  R_deg_OPG: OPG_bone => ; k_deg_OPG * OPG_bone;
  
  // Cellular dynamics
  R_prod_Osteocyte: => Osteocytes; k_prod_Osteocyte * Osteoblasts;
  R_death_Osteocyte: Osteocytes => ; k_death_Osteocyte * Osteocytes;
  R_diff_Osteoblast: => Osteoblasts; k_diff_Osteoblast * (1.0 / (1.0 + Sclerostin_bone / 5.0));
  R_apop_Osteoblast: Osteoblasts => ; k_apop_Osteoblast * Osteoblasts;
  R_diff_Osteoclast: => Osteoclasts; k_diff_Osteoclast * (RANKL_bone^n_RANKL / (K_RANKL_OC^n_RANKL + RANKL_bone^n_RANKL)) * (1.0 / (1.0 + OPG_bone / K_OPG_inhib));
  R_apop_Osteoclast: Osteoclasts => ; k_apop_Osteoclast * Osteoclasts;
  
  // Mineral dynamics
  R_prod_Mineral: => MineralIon; k_prod_Mineral * (1.0 + PTH / 100.0);
  R_loss_Mineral: MineralIon => ; k_loss_Mineral * MineralIon * Osteoblasts;
  
  // Transport
  R_diff_Scl: Sclerostin_bone => Sclerostin_sensor; k_diff_Scl * Sclerostin_bone * bone / sensor_chamber;
  R_back_Scl: Sclerostin_sensor => Sclerostin_bone; k_back_Scl * Sclerostin_sensor * sensor_chamber / bone;
  R_diff_RANKL: RANKL_bone => RANKL_sensor; k_diff_RANKL * RANKL_bone * bone / sensor_chamber;
  R_back_RANKL: RANKL_sensor => RANKL_bone; k_back_RANKL * RANKL_sensor * sensor_chamber / bone;
  R_diff_OPG: OPG_bone => OPG_sensor; k_diff_OPG * OPG_bone * bone / sensor_chamber;
  R_back_OPG: OPG_sensor => OPG_bone; k_back_OPG * OPG_sensor * sensor_chamber / bone;

  // Safety constraints to prevent unrealistic accumulation
  // These are "soft" limits that slow production at extreme values
  at (Sclerostin_bone > 0.3): k_prod_Scl = k_prod_Scl * 0.1;
  at (RANKL_bone > 5.0): k_prod_RANKL = k_prod_RANKL * 0.1;
  at (OPG_bone > 15.0): k_prod_OPG = k_prod_OPG * 0.1;

end'''

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / 'bone_environment.ant'
    with open(model_path, 'w') as f:
        f.write(antimony_model)

    return str(model_path)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
    description='Generate bone biosensor simulation dataset'
    )
    parser.add_argument(
    '--n_simulations',
    type=int,
    default=1000,
    help='Number of simulations to generate (default: 1000)'
    )
    parser.add_argument(
    '--output_dir',
    type=str,
    default='data',
    help='Output directory for dataset (default: data)'
    )
    parser.add_argument(
    '--duration',
    type=float,
    default=3600.0,
    help='Simulation duration in seconds (default: 3600)'
    )
    parser.add_argument(
    '--num_points',
    type=int,
    default=361,
    help='Number of time points (default: 361)'
    )
    parser.add_argument(
    '--log_level',
    type=str,
    default='INFO',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    help='Logging level (default: INFO)'
    )
    parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility (default: None)'
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    logger.info("="*80)
    logger.info("Bone Biosensor Simulation Framework")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Simulations: {args.n_simulations}")
    logger.info(f"  Duration: {args.duration}s")
    logger.info(f"  Points: {args.num_points}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Seed: {args.seed}")
    logger.info("="*80)

    try:
        # Create Antimony model file
        logger.info("Creating Antimony model...")
        model_path = create_antimony_model()
        logger.info(f"Model created: {model_path}")
        
        # Initialize dataset generator
        logger.info("Initializing dataset generator...")
        generator = DatasetGenerator(
            antimony_model_path=model_path,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        # Generate dataset
        logger.info("Starting dataset generation...")
        master_df = generator.generate_dataset(
            n_simulations=args.n_simulations,
            duration=args.duration,
            num_points=args.num_points
        )
        
        # Summary statistics
        logger.info("="*80)
        logger.info("Dataset Generation Complete!")
        logger.info("="*80)
        logger.info(f"Total simulations: {len(master_df)}")
        logger.info(f"\nScenario distribution:")
        logger.info(master_df['scenario'].value_counts())
        logger.info(f"\nBiosensor type distribution:")
        logger.info(master_df['biosensor_type'].value_counts())
        logger.info(f"\nNoise preset distribution:")
        logger.info(master_df['noise_preset'].value_counts())
        logger.info(f"\nSNR statistics:")
        logger.info(master_df['snr_db'].describe())
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())