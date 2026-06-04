"""
Priority 2: Frozen Parameters Experiment

Tests the hypothesis that biosensor parameter heterogeneity is the culprit.

Runs a limited dataset with IDENTICAL biosensor parameters across all simulations.
Only biological variability changes.

If d recovers → parameter randomness is dominant cause.
If d stays low → mechanism is elsewhere.
"""

import numpy as np
import pandas as pd
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrozenParameterExperiment:
    """
    Run simulations with identical biosensor parameters.
    
    Tests: Does parameter heterogeneity cause variance inflation?
    """
    
    def __init__(self, antimony_model_path: str, output_dir: str = "frozen_params_experiment"):
        self.antimony_model_path = antimony_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Import simulator
        try:
            from simulation.simulator import BoneEnvironmentSimulator
            from models.biosensors import create_biosensor
            from models.noise import get_noise_model
            from models.environment_configs import get_config
            from models.instrumentation_utilities import BiologicalRealization, create_instrumented_run
        except ImportError:
            logger.error("Failed to import simulation modules")
            raise
        
        self.simulator = BoneEnvironmentSimulator(antimony_model_path)
        self.create_biosensor = create_biosensor
        self.get_noise_model = get_noise_model
        self.get_config = get_config
        self.BiologicalRealization = BiologicalRealization
        self.create_instrumented_run = create_instrumented_run
        
        logger.info(f"Initialized frozen parameters experiment, output to {self.output_dir}")
    
    def run_with_frozen_config(self, 
                              scenario: str,
                              frozen_biosensor_config: dict,
                              n_simulations: int = 100):
        """
        Run simulations with identical biosensor parameters.
        
        Only biological variability changes (environment parameters).
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FROZEN PARAMETERS EXPERIMENT: {scenario.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Circuit: {frozen_biosensor_config.get('circuit_type')}")
        logger.info(f"Kd: {frozen_biosensor_config.get('kd'):.4f}")
        logger.info(f"Sensitivity: {frozen_biosensor_config.get('sensitivity'):.4f}")
        logger.info(f"N simulations: {n_simulations}\n")
        
        results = []
        
        for i in range(n_simulations):
            try:
                run_id = str(uuid.uuid4())
                
                # Generate DIFFERENT biological parameters (ONLY variability)
                env_config = self.get_config(scenario)
                params = env_config.apply_variability(0.15)
                
                # Run ODE
                self.simulator.reset()
                self.simulator.set_parameters(params)
                self.simulator.equilibrate_to_new_params(600.0)
                time, species_data = self.simulator.simulate(3600.0, 361)
                
                time = np.array(time, dtype=np.float64)
                species_data_clean = {}
                for key, value in species_data.items():
                    if type(value).__name__ == 'NamedArray':
                        species_data_clean[key] = np.array(value, dtype=np.float64)
                    else:
                        species_data_clean[key] = value
                species_data = species_data_clean
                
                # Get sclerostin
                sclerostin = species_data.get('Sclerostin_sensor',
                                             species_data.get('Sclerostin_bone'))
                
                # Apply FROZEN biosensor (identical across all sims)
                biosensor = self.create_biosensor(frozen_biosensor_config)
                
                # Measure
                engine = self.biosensor_engine_class(biosensor, noise_model=None)
                
                # Direct measurement instead
                time_test = np.array([1800.0])
                output = biosensor.measure(sclerostin[180:182], time_test)  # midpoint
                
                results.append({
                    'scenario': scenario,
                    'sclerostin_mean': float(np.mean(sclerostin)),
                    'sclerostin_std': float(np.std(sclerostin)),
                    'biosensor_output': float(output[0]) if len(output) > 0 else 0,
                })
                
                if (i + 1) % max(1, n_simulations // 5) == 0:
                    logger.info(f"Progress: {i+1}/{n_simulations}")
            
            except Exception as e:
                logger.warning(f"Simulation {i} failed: {e}")
                continue
        
        # Analyze results
        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTS - FROZEN PARAMETERS")
        logger.info(f"{'='*80}\n")
        
        if results:
            df = pd.DataFrame(results)
            
            scl_mean = df['sclerostin_mean'].mean()
            scl_std = df['sclerostin_mean'].std()
            
            out_mean = df['biosensor_output'].mean()
            out_std = df['biosensor_output'].std()
            
            logger.info(f"Sclerostin concentration:")
            logger.info(f"  Mean: {scl_mean:.6f} nM")
            logger.info(f"  Std:  {scl_std:.6f} nM (CV: {scl_std/scl_mean:.3f})\n")
            
            logger.info(f"Biosensor output (frozen params):")
            logger.info(f"  Mean: {out_mean:.6f}")
            logger.info(f"  Std:  {out_std:.6f} (CV: {out_std/out_mean if out_mean > 0 else 0:.3f})\n")
            
            # Compare variance ratios
            scl_cv = scl_std / scl_mean if scl_mean > 0 else 0
            out_cv = out_std / out_mean if out_mean > 0 else 0
            
            logger.info(f"Coefficient of variation:")
            logger.info(f"  Input (sclerostin):  {scl_cv:.4f}")
            logger.info(f"  Output (biosensor):  {out_cv:.4f}")
            logger.info(f"  Ratio (out/in):      {out_cv/scl_cv if scl_cv > 0 else 0:.2f}×\n")
            
            if out_cv > scl_cv * 2:
                logger.info("⚠️  Biosensor AMPLIFIES variance by >2×")
                logger.info("    Even with frozen parameters, output variance is huge")
                logger.info("    → Indicates saturation or nonlinearity\n")
            elif out_cv > scl_cv:
                logger.info("⚠️  Biosensor amplifies variance >1×")
                logger.info("    Consistent with transfer function effects\n")
            else:
                logger.info("✓ Biosensor preserves or reduces variance")
                logger.info("  Amplification is not from circuit dynamics\n")
        
        return df if results else None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python frozen_parameters_experiment.py <antimony_model_path> [output_dir]")
        sys.exit(1)
    
    antimony_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "frozen_params_experiment"
    
    logger.info("Frozen Parameters Experiment")
    logger.info("Tests: Does biosensor parameter heterogeneity cause variance inflation?\n")
    
    try:
        exp = FrozenParameterExperiment(antimony_path, output_dir)
        
        # Use a representative frozen config
        frozen_config = {
            'circuit_type': 'direct_binding',
            'kd': 1.0,  # Fixed at this value
            'sensitivity': 2.0,  # Fixed at this value
        }
        
        logger.info("Using FROZEN biosensor configuration:")
        logger.info(f"  Circuit: {frozen_config['circuit_type']}")
        logger.info(f"  Kd: {frozen_config['kd']}")
        logger.info(f"  Sensitivity: {frozen_config['sensitivity']}\n")
        
        # Run for each disease state
        for scenario in ['healthy', 'pmo', 'ckd_mbd']:
            df = exp.run_with_frozen_config(scenario, frozen_config, n_simulations=100)
        
        logger.info(f"\n{'='*80}")
        logger.info("INTERPRETATION")
        logger.info(f"{'='*80}\n")
        logger.info("If variance amplification occurred even with frozen parameters:")
        logger.info("  → Root cause is saturation/nonlinearity in transfer function\n")
        logger.info("If variance remained low with frozen parameters:")
        logger.info("  → Parameter heterogeneity IS the primary culprit\n")
        logger.info("Next: Compare these results to full-randomness dataset")
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)