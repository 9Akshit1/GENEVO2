"""
Dataset generation orchestrator.
"""

import numpy as np
import pandas as pd
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
import logging

# Import project modules
from models.environment_configs import get_config, list_scenarios
from models.biosensors import generate_random_biosensor_config, create_biosensor
from models.noise import get_noise_model
from simulation.simulator import BoneEnvironmentSimulator
from simulation.biosensor_engine import BiosensorEngine
from utils.validators import (validate_simulation_output, 
                              validate_biosensor_config,
                              validate_parameters)

logger = logging.getLogger(__name__)

def cleanup_for_json(data):
    """Recursively converts NumPy types in a dictionary or list to Python types."""
    if isinstance(data, dict):
        return {k: cleanup_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [cleanup_for_json(e) for e in data]
    elif hasattr(data, 'dtype') and 'float' in str(data.dtype):
        # Convert numpy floats/NamedArray to standard Python float
        return float(data)
    elif hasattr(data, 'tolist'):
        # Convert numpy arrays/NamedArray to standard Python lists
        return data.tolist()
    else:
        # Return as is if not a known problematic type
        return data

class DatasetGenerator:
    """
    Orchestrates generation of simulation dataset.
    """
    
    def __init__(self,
                 antimony_model_path: str,
                 output_dir: str = "data",
                 seed: Optional[int] = None):
        """
        Initialize dataset generator.
        
        Args:
            antimony_model_path: Path to Antimony bone environment model
            output_dir: Output directory for dataset
            seed: Random seed for reproducibility
        """
        self.antimony_model_path = antimony_model_path
        self.output_dir = output_dir
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create output directories
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        self.timeseries_dir = os.path.join(output_dir, 'timeseries')
        
        for directory in [self.output_dir, self.metadata_dir, self.timeseries_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize simulator
        self.simulator = BoneEnvironmentSimulator(antimony_model_path)
        
        # Master index for all simulations
        self.master_index = []
        
        logger.info(f"Initialized DatasetGenerator, output to: {output_dir}")
    
    def generate_single_simulation(self,
                                  scenario_name: str,
                                  biosensor_config: Dict,
                                  noise_preset: str = 'medium',
                                  duration: float = 3600.0,
                                  num_points: int = 361,
                                  apply_variability: bool = True) -> Dict:
        """
        Generate a single simulation run.
        
        Args:
            scenario_name: 'healthy', 'pmo', or 'ckd_mbd'
            biosensor_config: Biosensor configuration dictionary
            noise_preset: 'low', 'medium', or 'high'
            duration: Simulation duration (seconds)
            num_points: Number of time points
            apply_variability: Apply parameter variability for diversity
        
        Returns:
            Dictionary with simulation results and metadata
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.debug(f"Starting simulation {run_id[:8]}... ({scenario_name})")
        
        try:
            # Get environment configuration
            env_config = get_config(scenario_name)
            
            # Apply variability if requested
            if apply_variability:
                params = env_config.apply_variability(variability=0.15)
            else:
                params = env_config.get_params()
            
            # Set parameters in simulator
            self.simulator.reset()
            self.simulator.set_parameters(params)
            
            # Run simulation
            time, species_data = self.simulator.simulate(
                duration=duration,
                num_points=num_points,
                reset=False
            )
            
            # Validate simulation output
            validate_simulation_output(time, species_data)
            
            # Create biosensor and apply measurement
            biosensor = create_biosensor(biosensor_config)
            noise_model = get_noise_model(noise_preset)
            engine = BiosensorEngine(biosensor, noise_model)
            
            measured_signal, measurement_metadata = engine.measure(
                time, species_data, add_noise=True
            )
            
            # Compute error metrics
            true_sclerostin = species_data.get('Sclerostin_sensor',
                                              species_data.get('Sclerostin_bone'))
            
            # RMSE
            # Note: biosensor signal and true concentration not directly comparable
            # but we can compute deviation from true baseline
            baseline_true = np.mean(true_sclerostin[:10])
            baseline_measured = np.mean(measured_signal[:10])
            
            # --- Apply cleanup function to all dictionaries that might contain NumPy types ---
            # 1. Environment Parameters
            cleaned_params = cleanup_for_json(params)

            # 2. Biosensor Configuration (might contain Kd, kon, koff as NumPy floats)
            cleaned_biosensor_config = cleanup_for_json(biosensor_config)

            # 3. Noise Parameters
            cleaned_noise_params = cleanup_for_json(noise_model.to_dict())

            # 4. Measurement Metadata
            cleaned_measurement_metadata = cleanup_for_json(measurement_metadata)
            # ---------------------------------------------------------------------------------

            # Collect comprehensive metadata
            simulation_record = {
                'run_id': run_id,
                'timestamp': timestamp,
                'scenario': scenario_name,
                'duration': duration,
                'num_points': num_points,
                
                # Environment parameters
                'environment_params': cleaned_params, # <-- Use CLEANED
                
                # Biosensor configuration
                'biosensor_config': cleaned_biosensor_config, # <-- Use CLEANED
                
                # Noise configuration
                'noise_preset': noise_preset,
                'noise_params': cleaned_noise_params, # <-- Use CLEANED
                
                # Measurement metadata
                'measurement': cleaned_measurement_metadata, # <-- Use CLEANED
                
                # Summary statistics (already converted via float() which is good)
                'sclerostin_mean': float(np.mean(true_sclerostin)),
                'sclerostin_max': float(np.max(true_sclerostin)),
                'sclerostin_min': float(np.min(true_sclerostin)),
                'sclerostin_std': float(np.std(true_sclerostin)),
                
                # File paths (relative to output_dir)
                'metadata_file': f"metadata/{run_id}.json",
                'timeseries_file': f"timeseries/{run_id}.csv"
            }
            
            # Save metadata JSON
            metadata_path = os.path.join(self.output_dir, 
                                        simulation_record['metadata_file'])
            with open(metadata_path, 'w') as f:
                json.dump(simulation_record, f, indent=2)
            
            # Save time-series CSV
            timeseries_data = {
                'time': time,
                'sclerostin_bone': species_data.get('Sclerostin_bone', np.zeros_like(time)),
                'sclerostin_sensor': species_data.get('Sclerostin_sensor', true_sclerostin),
                'rankl_bone': species_data.get('RANKL_bone', np.zeros_like(time)),
                'opg_bone': species_data.get('OPG_bone', np.zeros_like(time)),
                'osteocytes': species_data.get('Osteocytes', np.zeros_like(time)),
                'osteoblasts': species_data.get('Osteoblasts', np.zeros_like(time)),
                'osteoclasts': species_data.get('Osteoclasts', np.zeros_like(time)),
                'biosensor_signal': measured_signal,
                'detection_event': engine.biosensor.is_detected(measured_signal).astype(int)
            }
            
            df = pd.DataFrame(timeseries_data)
            timeseries_path = os.path.join(self.output_dir,
                                          simulation_record['timeseries_file'])
            df.to_csv(timeseries_path, index=False)
            
            logger.info(f"Completed simulation {run_id[:8]}...")
            
            return simulation_record
            
        except Exception as e:
            logger.error(f"Simulation {run_id[:8]}... failed: {e}", exc_info=True)
            raise
    
    def generate_dataset(self,
                        n_simulations: int = 1000,
                        scenario_distribution: Optional[Dict[str, float]] = None,
                        biosensor_types: Optional[List[str]] = None,
                        noise_distribution: Optional[Dict[str, float]] = None,
                        duration: float = 3600.0,
                        num_points: int = 361) -> pd.DataFrame:
        """
        Generate complete dataset with multiple simulations.
        
        Args:
            n_simulations: Total number of simulations to generate
            scenario_distribution: Dict of scenario → probability
                                 (default: equal distribution)
            biosensor_types: List of biosensor circuit types to include
                           (default: all types)
            noise_distribution: Dict of noise preset → probability
                              (default: equal distribution)
            duration: Simulation duration (seconds)
            num_points: Number of time points per simulation
        
        Returns:
            DataFrame with master index of all simulations
        """
        # Default distributions
        if scenario_distribution is None:
            scenarios = list_scenarios()
            scenario_distribution = {s: 1.0/len(scenarios) for s in scenarios}
        
        if noise_distribution is None:
            noise_distribution = {'low': 0.2, 'medium': 0.5, 'high': 0.3}
        
        logger.info(f"Generating dataset: {n_simulations} simulations")
        logger.info(f"Scenario distribution: {scenario_distribution}")
        logger.info(f"Noise distribution: {noise_distribution}")
        
        # Generate simulations
        for i in range(n_simulations):
            # Sample scenario
            scenario = np.random.choice(
                list(scenario_distribution.keys()),
                p=list(scenario_distribution.values())
            )
            
            # Sample noise level
            noise_preset = np.random.choice(
                list(noise_distribution.keys()),
                p=list(noise_distribution.values())
            )
            
            # Generate random biosensor config
            biosensor_type = None
            if biosensor_types is not None:
                biosensor_type = np.random.choice(biosensor_types)
            
            biosensor_config = generate_random_biosensor_config(
                circuit_type=biosensor_type,
                seed=None
            )
            
            try:
                # Run simulation
                record = self.generate_single_simulation(
                    scenario_name=scenario,
                    biosensor_config=biosensor_config,
                    noise_preset=noise_preset,
                    duration=duration,
                    num_points=num_points,
                    apply_variability=True
                )
                
                # Add to master index
                self.master_index.append(record)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i+1}/{n_simulations} simulations completed")
                    
            except Exception as e:
                logger.error(f"Simulation {i} failed, skipping: {e}")
                continue
        
        # Create master index DataFrame
        master_df = pd.DataFrame([
            {
                'run_id': r['run_id'],
                'timestamp': r['timestamp'],
                'scenario': r['scenario'],
                'biosensor_type': r['biosensor_config']['circuit_type'],
                'noise_preset': r['noise_preset'],
                'snr_db': r['measurement']['snr_db'],
                'n_detections': r['measurement']['n_detections'],
                'time_to_detection': r['measurement']['time_to_detection'],
                'sclerostin_mean': r['sclerostin_mean'],
                'metadata_file': r['metadata_file'],
                'timeseries_file': r['timeseries_file']
            }
            for r in self.master_index
        ])
        
        # Save master index
        master_index_path = os.path.join(self.output_dir, 'master_index.csv')
        master_df.to_csv(master_index_path, index=False)
        
        logger.info(f"Dataset generation complete: {len(self.master_index)} simulations")
        logger.info(f"Master index saved to: {master_index_path}")
        
        return master_df
