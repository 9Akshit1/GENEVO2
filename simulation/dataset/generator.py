"""
Dataset generation orchestrator - v5.0 (Paired-Propagation Instrumentation).

INTEGRATION OF INSTRUMENTATION:
===============================
- Paired propagation: Single biological realization through all stages
- No resampling: Biology generated once, propagated through ODE→biosensor→noise→threshold→TTD
- Scalar metrics: Stage metrics extracted (mean, AUC, terminal_mean, etc.)
- Stage deltas: Explicit degradation quantification between stages
- Reproducibility: RNG seeds and parameter hashes preserved
"""

import numpy as np
import pandas as pd
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from simulation.models.environment_configs import get_config, list_scenarios
from simulation.models.biosensors import (
    generate_random_biosensor_config,
    generate_random_array_config,
    create_biosensor,
)
from simulation.models.noise import get_noise_model
from simulation.models.instrumentation_utilities import (
    BiologicalRealization,
    PairedPropagationRun,
    create_instrumented_run,
)
from simulation.simulator import BoneEnvironmentSimulator
from simulation.biosensor_engine import BiosensorEngine

logger = logging.getLogger(__name__)


def cleanup_for_json(data):
    """Recursively convert all types to JSON-serializable natives."""
    if data is None:
        return None
    
    if isinstance(data, dict):
        return {k: cleanup_for_json(v) for k, v in data.items()}
    
    if isinstance(data, (list, tuple)):
        return [cleanup_for_json(e) for e in data]
    
    if type(data).__name__ == 'NamedArray':
        try:
            return data.tolist()
        except:
            try:
                return list(data)
            except:
                return [float(x) for x in data]
    
    if isinstance(data, np.ndarray):
        return data.tolist()
    
    if isinstance(data, (np.floating, float)):
        val = float(data)
        return None if np.isnan(val) else val
    
    if isinstance(data, (np.integer, int)):
        return int(data)
    
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    
    if hasattr(data, 'tolist') and not isinstance(data, str):
        try:
            return cleanup_for_json(data.tolist())
        except:
            pass
    
    if isinstance(data, str):
        return data
    
    return str(data)


class DatasetGenerator:
    """
    Orchestrates generation of simulation dataset.
    V5.0: Implements paired-propagation instrumentation.
    """
    
    def __init__(self,
                 antimony_model_path: str,
                 output_dir: str = "data",
                 seed: Optional[int] = None,
                 sigma_measurement: float = 0.0):
        """Initialize dataset generator.

        Args:
            sigma_measurement: Lognormal sigma for Layer 4 measurement noise,
                applied as: measured = true_value * lognormal(0, sigma_measurement).
                Represents assay CV + pre-analytical variation (circadian, freeze-thaw).
                0.0 = no noise (default; preserves backward compatibility).
                0.15-0.18 = realistic for clinical bone marker assays.
        """
        self.antimony_model_path = antimony_model_path
        self.output_dir = output_dir
        self.seed = seed
        self.sigma_measurement = float(sigma_measurement)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.metadata_dir = os.path.join(output_dir, 'metadata')
        self.timeseries_dir = os.path.join(output_dir, 'timeseries')
        
        for directory in [self.output_dir, self.metadata_dir, self.timeseries_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.simulator = BoneEnvironmentSimulator(antimony_model_path)
        self.master_index = []
        
        self.generation_stats = {
            'total_attempted': 0,
            'total_succeeded': 0,
            'total_failed': 0,
            'failed_reasons': {},
            'rejected_parameters': 0,
        }
        
        logger.info(f"Initialized DatasetGenerator, output to: {output_dir}")
    
    def generate_single_simulation_instrumented(self,
                                            scenario_name: str,
                                            biosensor_config: Dict,
                                            noise_preset: str = 'medium',
                                            duration: float = 3600.0,
                                            num_points: int = 361,
                                            apply_variability: bool = True,
                                            instrument: bool = True,
                                            rng_seed: Optional[int] = None) -> Optional[Dict]:
        """
        Generate a single simulation run with instrumentation.
        
        CRITICAL DESIGN:
          - Single biological realization (params drawn once)
          - Propagated through ALL stages without resampling
          - Stage metrics extracted at each transformation
          - Stage deltas computed to quantify degradation
        
        Args:
            scenario_name: 'healthy', 'pmo', or 'ckd_mbd'
            biosensor_config: Biosensor circuit configuration
            noise_preset: 'low', 'medium', or 'high'
            duration: Simulation duration (seconds)
            num_points: Number of output time points
            apply_variability: Whether to apply parameter variability
            instrument: Whether to record instrumentation metadata
            rng_seed: RNG seed for reproducibility (optional)
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.debug(f"Starting simulation {run_id[:8]}... ({scenario_name})")
        
        try:
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION SETUP
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                instrumented = create_instrumented_run(
                    scenario_name,
                    biosensor_config,
                    noise_preset
                )
            else:
                instrumented = None
            
            # ════════════════════════════════════════════════════════════════
            # GENERATE BIOLOGICAL REALIZATION (ONCE - CRITICAL!)
            # ════════════════════════════════════════════════════════════════
            
            # Set seed if provided
            if rng_seed is not None:
                np.random.seed(rng_seed)
            else:
                rng_seed = np.random.randint(0, 2**31)
            
            # Get environment configuration
            env_config = get_config(scenario_name)
            
            # FIX 2: Rejection sampling for out-of-bounds parameters
            max_resample_attempts = 10
            for attempt in range(max_resample_attempts):
                try:
                    if apply_variability:
                        params = env_config.apply_variability(variability=0.15, use_correlations=True)
                    else:
                        params = env_config.get_params()
                    break  # Success
                except ValueError as e:
                    self.generation_stats['rejected_parameters'] += 1
                    if attempt < max_resample_attempts - 1:
                        logger.debug(f"Parameter out of bounds, resampling ({attempt+1}/{max_resample_attempts})...")
                        continue
                    else:
                        raise ValueError(f"Max resample attempts exceeded: {e}")
            
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION: Record biological realization
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                bio_realization = BiologicalRealization(
                    params=params,
                    rng_seed=rng_seed,
                    scenario=scenario_name
                )
            
            # ════════════════════════════════════════════════════════════════
            # RUN ODE SIMULATION (SINGLE RUN - NO RESAMPLING)
            # ════════════════════════════════════════════════════════════════
            
            # Set parameters in simulator
            self.simulator.reset()
            self.simulator.set_parameters(params)
            
            # FIX 1: Equilibrate to new parameters before measurement
            logger.debug(f"Running equilibration phase...")
            self.simulator.equilibrate_to_new_params(duration_seconds=600.0)
            
            # Run actual measurement
            time, species_data = self.simulator.simulate(
                duration=duration,
                num_points=num_points,
                reset=False
            )
            
            # Convert NamedArray to numpy
            time = np.array(time, dtype=np.float64)
            species_data_clean = {}
            for key, value in species_data.items():
                if type(value).__name__ == 'NamedArray':
                    species_data_clean[key] = np.array(value, dtype=np.float64)
                else:
                    species_data_clean[key] = value
            species_data = species_data_clean
            
            # FIX 5: Check diffusion equilibrium
            self.simulator.check_diffusion_equilibrium(species_data)
            
            # FIX 1 (validation): Check system reached equilibrium
            equilibrium_stats = self.simulator.validate_equilibrium(species_data, time)
            
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION: Record ODE output (Stage 0)
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                instrumented.set_biological_realization(bio_realization, time, species_data)
                instrumented.record_stage_0_ode()
            
            # Extract analyte concentrations
            true_sclerostin = species_data.get('Sclerostin_sensor',
                                               species_data.get('Sclerostin_bone'))
            true_ctx  = species_data.get('CTX_sensor',  species_data.get('CTX_bone'))
            true_p1np = species_data.get('P1NP_sensor', species_data.get('P1NP_bone'))

            sclerostin_mean = np.mean(true_sclerostin)
            sclerostin_std = np.std(true_sclerostin)
            sclerostin_max = np.max(true_sclerostin)
            sclerostin_min = np.min(true_sclerostin)

            ctx_mean  = float(np.mean(true_ctx))  if true_ctx  is not None else float('nan')
            p1np_mean = float(np.mean(true_p1np)) if true_p1np is not None else float('nan')

            # ════════════════════════════════════════════════════════════════
            # LAYER 4: MEASUREMENT PROCESS NOISE
            # ════════════════════════════════════════════════════════════════
            # Applied ONLY to scalar summary values (what an assay would report).
            # The ODE time-series and biosensor binding see true concentrations.
            # sigma_measurement = 0.0 by default (backward-compatible; no noise).
            # Realistic clinical value: 0.15-0.18 (assay CV ~4-8% + pre-analytical
            # variation ~10-15% in quadrature; Ricos 1999; Vasikaran 2011).
            if self.sigma_measurement > 0.0:
                sclerostin_mean = float(sclerostin_mean * np.random.lognormal(0.0, self.sigma_measurement))
                if not np.isnan(ctx_mean):
                    ctx_mean = float(ctx_mean * np.random.lognormal(0.0, self.sigma_measurement))
                if not np.isnan(p1np_mean):
                    p1np_mean = float(p1np_mean * np.random.lognormal(0.0, self.sigma_measurement))

            # ════════════════════════════════════════════════════════════════
            # BIOSENSOR MEASUREMENT (NO RESAMPLING)
            # ════════════════════════════════════════════════════════════════
            
            # Create biosensor from config
            biosensor = create_biosensor(biosensor_config)
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 1: Clean biosensor output (NO NOISE)
            # ════════════════════════════════════════════════════════════════
            
            engine_clean = BiosensorEngine(biosensor, noise_model=None)
            clean_signal, _ = engine_clean.measure(time, species_data, add_noise=False)
            
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION: Record clean biosensor output
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                instrumented.record_stage_1_biosensor_clean(clean_signal)
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 2: Apply noise (TO SAME CLEAN SIGNAL - NO RE-MEASUREMENT)
            # ════════════════════════════════════════════════════════════════
            
            # Create noise model
            noise_model = get_noise_model(noise_preset)
            
            # CRITICAL: Apply noise to SAME clean signal
            # Do NOT re-measure or re-simulate
            noisy_signal, _ = noise_model.apply_noise(clean_signal, time)
            snr = noise_model.get_snr(clean_signal, noisy_signal)
            
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION: Record noisy output
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                instrumented.record_stage_2_biosensor_noisy(noisy_signal, snr)
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 3: Threshold detection (ON SAME NOISY SIGNAL)
            # ════════════════════════════════════════════════════════════════

            threshold = biosensor.threshold
            # Route through BiosensorEngine — single detection implementation
            detected, max_consecutive, _, _ = engine_clean._detect(noisy_signal)
            elevation_mask = noisy_signal >= threshold  # raw crossings for CSV/instrumentation
            
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION: Record threshold metrics
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                instrumented.record_stage_3_thresholded(noisy_signal, threshold)
            
            # ════════════════════════════════════════════════════════════════
            # STAGE 4: TTD calculation
            # ════════════════════════════════════════════════════════════════
            
            # TTD: SNR-dependent delay via BiosensorEngine (fixes hardcoded 400s)
            time_to_detection = engine_clean._calculate_ttd(noisy_signal, time, snr)
            
            # ════════════════════════════════════════════════════════════════
            # INSTRUMENTATION: Record TTD
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                instrumented.record_stage_4_ttd(elevation_mask, time_to_detection)
            
            # ════════════════════════════════════════════════════════════════
            # Calculate FNR — empirical, multi-trial
            # ════════════════════════════════════════════════════════════════
            # Run N_FNR_TRIALS independent noise draws on the SAME clean signal.
            # FNR = fraction of trials where engine_clean._detect() returns False.
            # Uses the same detection logic (detection_window, rolling_window) as Stage 3.
            N_FNR_TRIALS = 50
            fnr_miss_count = 0
            for _ in range(N_FNR_TRIALS):
                trial_noisy, _ = noise_model.apply_noise(clean_signal, time)
                trial_detected, _, _, _ = engine_clean._detect(trial_noisy)
                if not trial_detected:
                    fnr_miss_count += 1
            false_negative_rate = fnr_miss_count / N_FNR_TRIALS
            
            # ════════════════════════════════════════════════════════════════
            # BUILD SIMULATION RECORD
            # ════════════════════════════════════════════════════════════════
            
            simulation_record = {
                'run_id': run_id,
                'timestamp': timestamp,
                'scenario': scenario_name,
                'biosensor_config': cleanup_for_json(biosensor_config),
                'noise_preset': noise_preset,
                'measurement': {
                    'snr_db': float(snr),
                    'n_detections': int(1 if detected else 0),
                    'detection_rate': float(1.0 if detected else 0.0),
                    'time_to_detection': float(time_to_detection),
                    'false_negative_rate': float(false_negative_rate),
                },
                'environment_params': cleanup_for_json(params),
                'equilibration_check': cleanup_for_json(equilibrium_stats),
                'sclerostin_mean': sclerostin_mean,
                'sclerostin_max': sclerostin_max,
                'sclerostin_min': sclerostin_min,
                'sclerostin_std': sclerostin_std,
                'ctx_mean':  ctx_mean,
                'p1np_mean': p1np_mean,
                'metadata_file': f"metadata/{run_id}.json",
                'timeseries_file': f"timeseries/{run_id}.csv"
            }
            
            # ════════════════════════════════════════════════════════════════
            # ADD INSTRUMENTATION TO RECORD
            # ════════════════════════════════════════════════════════════════
            
            if instrument:
                simulation_record['instrumentation'] = instrumented.get_metadata()
            
            # ════════════════════════════════════════════════════════════════
            # VERIFY AND SAVE
            # ════════════════════════════════════════════════════════════════
            
            # Verify JSON serializability
            try:
                json.dumps(simulation_record)
            except TypeError as e:
                raise ValueError(f"Record not JSON serializable: {e}")
            
            # Save metadata
            metadata_path = os.path.join(self.output_dir, simulation_record['metadata_file'])
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(simulation_record, f, indent=2, ensure_ascii=False)
            
            # Save timeseries
            timeseries_data = {
                'time': time,
                'sclerostin_bone': species_data.get('Sclerostin_bone', np.zeros_like(time)),
                'sclerostin_sensor': species_data.get('Sclerostin_sensor', true_sclerostin),
                'rankl_bone': species_data.get('RANKL_bone', np.zeros_like(time)),
                'opg_bone': species_data.get('OPG_bone', np.zeros_like(time)),
                'ctx_bone': species_data.get('CTX_bone', np.zeros_like(time)),
                'ctx_sensor': species_data.get('CTX_sensor', np.zeros_like(time)),
                'p1np_bone': species_data.get('P1NP_bone', np.zeros_like(time)),
                'p1np_sensor': species_data.get('P1NP_sensor', np.zeros_like(time)),
                'osteocytes': species_data.get('Osteocytes', np.zeros_like(time)),
                'osteoblasts': species_data.get('Osteoblasts', np.zeros_like(time)),
                'osteoclasts': species_data.get('Osteoclasts', np.zeros_like(time)),
                'biosensor_signal': clean_signal,
                'detection_event': elevation_mask.astype(int)
            }
            
            df = pd.DataFrame(timeseries_data)
            timeseries_path = os.path.join(self.output_dir, simulation_record['timeseries_file'])
            df.to_csv(timeseries_path, index=False, encoding='utf-8')
            
            logger.info(f"[OK] Completed simulation {run_id[:8]}... ({scenario_name})")
            return simulation_record
            
        except Exception as e:
            logger.error(f"[FAIL] Simulation {run_id[:8]}... failed: {str(e)}", exc_info=True)
            
            reason = type(e).__name__
            if reason not in self.generation_stats['failed_reasons']:
                self.generation_stats['failed_reasons'][reason] = 0
            self.generation_stats['failed_reasons'][reason] += 1
            
            return None
    
    def generate_dataset(self,
                        n_simulations: int = 1000,
                        scenario_distribution: Optional[Dict[str, float]] = None,
                        biosensor_types: Optional[List[str]] = None,
                        noise_distribution: Optional[Dict[str, float]] = None,
                        duration: float = 3600.0,
                        num_points: int = 361,
                        run_validator: bool = True) -> pd.DataFrame:
        """Generate complete dataset with paired-propagation instrumentation."""
        
        if scenario_distribution is None:
            scenario_distribution = {
                'healthy': 0.30,
                'pmo': 0.35,
                'ckd_mbd': 0.35
            }
        
        if noise_distribution is None:
            noise_distribution = {
                'low': 0.15,
                'medium': 0.55,
                'high': 0.30
            }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DATASET GENERATION INITIATED (v5.0 Paired-Propagation Instrumentation)")
        logger.info(f"{'='*80}")
        logger.info(f"Total simulations: {n_simulations}")
        logger.info(f"Scenario distribution: {scenario_distribution}")
        logger.info(f"Noise distribution: {noise_distribution}")
        
        for i in range(n_simulations):
            self.generation_stats['total_attempted'] += 1
            
            scenario = np.random.choice(
                list(scenario_distribution.keys()),
                p=list(scenario_distribution.values())
            )
            
            noise_preset = np.random.choice(
                list(noise_distribution.keys()),
                p=list(noise_distribution.values())
            )
            
            biosensor_type = None
            if biosensor_types is not None:
                biosensor_type = np.random.choice(biosensor_types)

            if biosensor_type == 'array':
                biosensor_config = generate_random_array_config(seed=None)
            else:
                biosensor_config = generate_random_biosensor_config(
                    circuit_type=biosensor_type,
                    seed=None
                )
            
            # NEW: Generate reproducible RNG seed for instrumentation
            rng_seed = np.random.randint(0, 2**31)
            
            record = self.generate_single_simulation_instrumented(
                scenario_name=scenario,
                biosensor_config=biosensor_config,
                noise_preset=noise_preset,
                duration=duration,
                num_points=num_points,
                apply_variability=True,
                instrument=True,  # Enable instrumentation
                rng_seed=rng_seed  # Pass seed for reproducibility
            )
            
            if record is not None:
                self.master_index.append(record)
                self.generation_stats['total_succeeded'] += 1
            else:
                self.generation_stats['total_failed'] += 1
            
            if (i + 1) % max(1, n_simulations // 10) == 0:
                progress_pct = 100 * (i + 1) / n_simulations
                logger.info(f"Progress: {i+1}/{n_simulations} ({progress_pct:.1f}%) "
                           f"[Success: {self.generation_stats['total_succeeded']}, "
                           f"Failed: {self.generation_stats['total_failed']}, "
                           f"Rejected params: {self.generation_stats['rejected_parameters']}]")
        
        # Create master index
        master_df = pd.DataFrame([
            {
                'run_id': r['run_id'],
                'timestamp': r['timestamp'],
                'scenario': r['scenario'],
                'biosensor_type': r['biosensor_config']['circuit_type'],
                'noise_preset': r['noise_preset'],
                # Single-channel design parameters (NaN for array biosensors)
                'kd': float(r['biosensor_config'].get('kd', float('nan'))),
                'sensitivity': float(r['biosensor_config'].get('sensitivity', float('nan'))),
                'threshold': float(r['biosensor_config'].get('threshold', float('nan'))),
                'response_time': float(
                    r['biosensor_config'].get('response_time', float('nan'))
                    if r['biosensor_config'].get('circuit_type') == 'amplifying'
                    else float('nan')
                ),
                # Array biosensor per-channel parameters (NaN for single-channel)
                'kd_scl':  float(r['biosensor_config'].get('kd_scl',  float('nan'))),
                'kd_ctx':  float(r['biosensor_config'].get('kd_ctx',  float('nan'))),
                'kd_p1np': float(r['biosensor_config'].get('kd_p1np', float('nan'))),
                'w_scl':   float(r['biosensor_config'].get('w_scl',   float('nan'))),
                'w_ctx':   float(r['biosensor_config'].get('w_ctx',   float('nan'))),
                'w_p1np':  float(r['biosensor_config'].get('w_p1np',  float('nan'))),
                'snr_db': float(r['measurement'].get('snr_db', 0.0)),
                'n_detections': int(r['measurement'].get('n_detections', 0)),
                'detection_rate': float(r['measurement'].get('detection_rate', 0.0)),
                'time_to_detection': float(r['measurement'].get('time_to_detection', 0.0)),
                'false_negative_rate': float(r['measurement'].get('false_negative_rate', 0.0)),
                'sclerostin_mean': r['sclerostin_mean'],
                'sclerostin_std':  r['sclerostin_std'],
                'ctx_mean':  r.get('ctx_mean',  float('nan')),
                'p1np_mean': r.get('p1np_mean', float('nan')),
                'metadata_file': r['metadata_file'],
                'timeseries_file': r['timeseries_file']
            }
            for r in self.master_index
        ])
        
        master_index_path = os.path.join(self.output_dir, 'master_index.csv')
        master_df.to_csv(master_index_path, index=False, encoding='utf-8')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DATASET GENERATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total simulations generated: {len(self.master_index)}/{n_simulations}")
        logger.info(f"Success rate: {100*self.generation_stats['total_succeeded']/max(1,self.generation_stats['total_attempted']):.1f}%")
        logger.info(f"Parameter rejections (out-of-bounds): {self.generation_stats['rejected_parameters']}")
        
        if self.generation_stats['failed_reasons']:
            logger.warning(f"Failure reasons: {self.generation_stats['failed_reasons']}")
        
        logger.info(f"\nSCENARIO DISTRIBUTION:")
        scenario_counts = master_df['scenario'].value_counts()
        for scenario, count in scenario_counts.items():
            pct = 100 * count / len(master_df)
            logger.info(f"  {scenario:12s}: {count:5d} ({pct:5.1f}%)")
        
        logger.info(f"\nNOISE LEVEL DISTRIBUTION:")
        noise_counts = master_df['noise_preset'].value_counts()
        for noise, count in noise_counts.items():
            pct = 100 * count / len(master_df)
            logger.info(f"  {noise:8s}: {count:5d} ({pct:5.1f}%)")
        
        logger.info(f"\nKEY METRICS:")
        logger.info(f"  Time-to-Detection: mean={master_df['time_to_detection'].mean():.1f}s, "
                   f"std={master_df['time_to_detection'].std():.1f}s")
        logger.info(f"  False Negative Rate: mean={master_df['false_negative_rate'].mean():.2%}")
        logger.info(f"  SNR: mean={master_df['snr_db'].mean():.1f}dB")
        
        logger.info(f"\nMaster index saved to: {master_index_path}")
        logger.info(f"Instrumentation enabled: All metadata includes stagewise metrics")
        logger.info(f"Next: python analyze_instrumentation.py {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        return master_df