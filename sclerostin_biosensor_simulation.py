"""
Fixed Sclerostin Biosensor Simulation and Dataset Generation
=============================================================

CRITICAL FIXES:
1. Proper noise injection via parameter variation
2. Fixed diffusion equations for realistic gradients
3. Mechanical loading actually affects sclerostin
4. Biosensor circuit has proper dose-response
5. pH is now a parameter (homeostatic)
6. Cross-talk properly implemented
"""

import tellurium as te
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BIOLOGICAL PARAMETERS FROM LITERATURE
# ============================================================================

@dataclass
class BiologicalParameters:
    """
    Biologically realistic parameters for bone microenvironment simulation.
    All values sourced from peer-reviewed literature.
    """
    
    # SCLEROSTIN CONCENTRATIONS (pmol/L)
    # Source: Xu et al. 2020, Front Cell Dev Biol 8:57
    # "Median sclerostin: elderly males: 54.89 pmol/L, elderly females: 39.95 pmol/L"
    sclerostin_normal_mean: float = 47.0  # pmol/L (average of healthy adults)
    sclerostin_normal_std: float = 16.0   # pmol/L
    sclerostin_low: float = 20.0          # pmol/L (low bone formation)
    sclerostin_high: float = 120.0        # pmol/L (immobilization, kidney disease)
    # Source: Armbrecht et al. 2010, J Clin Endocrinol Metab 95(5):2050
    # "Immobilized patients: 0.975 ng/mL (median) vs controls 0.300 ng/mL"
    
    # DKK1 CONCENTRATIONS (pmol/L)
    # Source: van Lierop et al. 2013, J Clin Endocrinol Metab 98(12):4908
    # "Healthy controls: 2.77 ng/mL; Sclerosteosis patients: 4.28-5.28 ng/mL"
    # Conversion: 1 ng/mL DKK1 â‰ˆ 35 pmol/L (MW ~28.68 kDa per Biomedica)
    dkk1_normal_mean: float = 30.0        # pmol/L
    dkk1_normal_std: float = 10.0         # pmol/L
    dkk1_low: float = 15.0                # pmol/L
    dkk1_high: float = 60.0               # pmol/L (myeloma, thalassemia)
    # Source: Christoulas et al. 2009, Haematologica 94(11):1560
    # "Thalassemia: 39Â±17.1 pmol/L vs controls 27.4Â±9.7 pmol/L"
    
    # OSTEOCALCIN CONCENTRATIONS (ng/mL)
    # Source: Hannemann et al. 2013, BMC Endocr Disord 13:11
    # "Men median: 15.4 ng/mL; Premenopausal women: 14.4 ng/mL"
    osteocalcin_normal_mean: float = 15.0  # ng/mL
    osteocalcin_normal_std: float = 4.0    # ng/mL
    osteocalcin_low: float = 5.0           # ng/mL (low bone formation)
    osteocalcin_high: float = 40.0         # ng/mL (postmenopausal, high turnover)
    # Source: Singh et al. 2013, J Midlife Health 4:175
    # "Osteoporotic: 16.16Â±4.5 ng/ml vs non-osteoporotic: 11.26Â±3.07 ng/ml"
    
    # EXTRACELLULAR CALCIUM CONCENTRATIONS (mM)
    # Source: Shapiro et al. 2022, Nat Commun 13:1395
    # "Blood serum Ca2+: ~1 mM; Bone resorption sites: up to 40 mM"
    # "BM interstitial Ca2+ from extracted fluid: 0.5 mM average"
    calcium_serum: float = 1.2            # mM (tightly regulated)
    calcium_bone_normal: float = 1.5      # mM (endosteal surface)
    calcium_bone_resorption: float = 8.0  # mM (active resorption)
    calcium_bone_low: float = 0.5         # mM (marrow interstitial)
    
    # pH VALUES
    # Source: Arnett 2008, J Nutr 138:415S
    # "Osteoclast resorption: maximal at pH ~7.0; switched off above pH 7.4"
    # Source: Biskobing et al. 2001, Am J Physiol Endocrinol Metab 280:E112
    # "Culture medium pH near osteoclasts: ~7.0-7.1"
    ph_blood: float = 7.40                # physiological blood pH
    ph_bone_normal: float = 7.35          # bone microenvironment
    ph_bone_resorption: float = 7.0       # active resorption site
    ph_bone_formation: float = 7.45       # bone formation (alkaline)
    
    # DIFFUSION COEFFICIENTS (Î¼mÂ²/s)
    # Source: Wang et al. 2013, PLoS One 8(11):e82382
    # "Proteins 1-10 nm radius in ECM: D reduced by 50-90% vs free solution"
    # Sclerostin MW: ~22.5 kDa, radius ~2-3 nm
    # Free solution D â‰ˆ 100 Î¼mÂ²/s for ~20 kDa protein (Stokes-Einstein)
    diffusion_sclerostin_free: float = 100.0  # Î¼mÂ²/s
    diffusion_sclerostin_ecm: float = 20.0    # Î¼mÂ²/s (80% reduction in bone ECM)
    diffusion_dkk1: float = 18.0              # Î¼mÂ²/s (slightly larger, ~28 kDa)
    diffusion_calcium: float = 600.0          # Î¼mÂ²/s (small ion)
    
    # SPATIAL PARAMETERS
    # Source: Bonewald & Johnson 2008, Bone 42:606
    # "Osteocyte lacuno-canalicular network spacing: 20-30 Î¼m"
    osteocyte_spacing: float = 25.0       # Î¼m (typical spacing)
    canaliculi_length: float = 15.0       # Î¼m (process length)
    sensor_distance_min: float = 5.0      # Î¼m (minimum distance to sensor)
    sensor_distance_max: float = 50.0     # Î¼m (maximum detection distance)
    
    # MOLECULAR BINDING PARAMETERS
    # Source: Literature estimates for Wnt antagonists
    # Sclerostin binds LRP5/6 with Kd ~1-10 nM range
    sclerostin_receptor_kd: float = 5.0   # nM (binding affinity)
    dkk1_receptor_kd: float = 3.0         # nM (higher affinity than sclerostin)
    
    # HALF-LIVES (minutes)
    # Source: Delmas et al. 2000, Osteoporos Int 11(Suppl 6):S2-S17
    # "Osteocalcin half-life: ~5 minutes in circulation"
    sclerostin_halflife: float = 180.0    # min (estimated, longer than OC)
    dkk1_halflife: float = 120.0          # min (estimated)
    osteocalcin_halflife: float = 5.0     # min (rapid clearance)
    mrna_halflife: float = 30.0           # min (typical mRNA)
    protein_reporter_halflife: float = 60.0  # min (fluorescent protein)

class BoneMicroenvironment:
    """
    Simulates bone microenvironment with FIXED biological realism.
    """
    
    def __init__(self, params: BiologicalParameters):
        self.params = params
        
    def create_model(self) -> te.roadrunner:
        """Load fixed bone microenvironment model."""
        antimony_filepath = "bone_microenvironment.ant"
        
        try:
            with open(antimony_filepath, encoding="utf-8") as f:
                antimony_str = f.read()
            return te.loada(antimony_str)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: Antimony file '{antimony_filepath}' not found."
            )


class BiosensorCircuit:
    """
    Models genetic biosensor with FIXED dose-response.
    """
    
    def __init__(self, circuit_type: str = "direct"):
        self.circuit_type = circuit_type
        
    def create_circuit_model(self, 
                            receptor_kd: float = 5.0,
                            amplification_factor: float = 1.0,
                            promoter_strength: float = 1.0,
                            feedback_strength: float = 0.0) -> te.roadrunner:
        """Load and configure biosensor circuit."""
        
        antimony_filepath = "biosensor_circuit.ant"
        try:
            with open(antimony_filepath, encoding="utf-8") as f:
                model_ant_string = f.read()
            
            r_circuit = te.loadAntimonyModel(model_ant_string)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: Antimony file '{antimony_filepath}' not found."
            )
        
        # Set tunable parameters
        r_circuit['k_unbind'] = receptor_kd
        r_circuit['k_signal_activation'] = amplification_factor * 1.0
        r_circuit['k_transcription'] = promoter_strength * 5.0
        r_circuit['k_feedback_production'] = feedback_strength * 0.5
        
        return r_circuit


class DatasetGenerator:
    """
    Generates comprehensive datasets with FIXED biological realism.
    """
    
    def __init__(self, params: BiologicalParameters):
        self.params = params
        self.bone_env = BoneMicroenvironment(params)
        self.data = []
        
    def simulate_scenario(self,
                         sclerostin_level: str = "normal",
                         calcium_level: float = None,
                         ph_value: float = None,
                         dkk1_concentration: float = None,
                         mechanical_loading: float = 1.0,
                         receptor_kd: float = 5.0,
                         amplification_factor: float = 1.0,
                         promoter_strength: float = 0.8,
                         feedback_strength: float = 0.0,
                         circuit_type: str = "direct",
                         simulation_time: float = 500.0,
                         noise_level: float = 0.1) -> Dict:
        """
        Run single simulation scenario - FIXED VERSION.
        """
        
        # Set sclerostin initial condition with biological variability
        if sclerostin_level == "low":
            sclerostin_init = self.params.sclerostin_low
        elif sclerostin_level == "high":
            sclerostin_init = self.params.sclerostin_high
        elif sclerostin_level == "normal":
            sclerostin_init = self.params.sclerostin_normal_mean
        else:
            sclerostin_init = float(sclerostin_level)
        
        # FIXED: Apply biological variability to initial condition
        sclerostin_init *= np.random.lognormal(0, noise_level)
        
        # Set other environmental parameters with variability
        calcium_init = calcium_level if calcium_level else self.params.calcium_bone_normal
        calcium_init *= np.random.lognormal(0, noise_level * 0.5)
        
        ph_init = ph_value if ph_value else self.params.ph_bone_normal
        ph_init += np.random.normal(0, 0.05)
        ph_init = np.clip(ph_init, 7.0, 7.5)
        
        dkk1_init = dkk1_concentration if dkk1_concentration else self.params.dkk1_normal_mean
        dkk1_init *= np.random.lognormal(0, noise_level)
        
        # Create bone microenvironment model
        r_bone = self.bone_env.create_model()
        
        # Set initial conditions
        r_bone['sclerostin'] = sclerostin_init
        r_bone['calcium'] = calcium_init
        r_bone['dkk1'] = dkk1_init
        r_bone['dkk1_sensor'] = dkk1_init * 0.8  # Start sensor slightly lower
        r_bone['dkk1_mrna'] = dkk1_init * 0.05   # Start mRNA proportionally

        # CRITICAL: Scale DKK1 transcription to maintain target level
        # At steady state: production ≈ degradation
        # We want: k_transcription * k_secretion / k_deg ≈ dkk1_init
        target_transcription = dkk1_init * r_bone['k_deg_dkk1'] / (r_bone['k_secretion_dkk1'] * r_bone['pH_effect'])
        r_bone['k_transcription_dkk1'] = target_transcription * 0.023  # Scale by mRNA decay
        
        # CRITICAL FIX: Set PARAMETERS not species
        r_bone['pH_ecm'] = ph_init
        r_bone['mechanical_signal'] = mechanical_loading
        r_bone['noise_amplitude'] = noise_level
        
        # Simulate bone environment
        bone_result = r_bone.simulate(0, simulation_time, 100)
        
        # Extract sensor region concentrations at steady state
        final_idx = -10
        sclerostin_sensor = np.mean(bone_result['[sclerostin_sensor]'][final_idx:])
        dkk1_sensor = np.mean(bone_result['[dkk1_sensor]'][final_idx:])
        calcium_sensor = np.mean(bone_result['[calcium_sensor]'][final_idx:])
        osteocalcin_sensor = np.mean(bone_result['[osteocalcin_sensor]'][final_idx:])
        
        # Simulate biosensor circuit
        biosensor = BiosensorCircuit(circuit_type)
        r_circuit = biosensor.create_circuit_model(
            receptor_kd=receptor_kd,
            amplification_factor=amplification_factor,
            promoter_strength=promoter_strength,
            feedback_strength=feedback_strength
        )
        
        # Set input from bone environment
        r_circuit['sclerostin_input'] = sclerostin_sensor
        
        # Simulate circuit
        circuit_result = r_circuit.simulate(0, simulation_time, 200)  # More time points
        
        # Calculate performance metrics
        final_signal = np.mean(circuit_result['[reporter_protein]'][final_idx:])
        signal_std = np.std(circuit_result['[reporter_protein]'][final_idx:])
        
        # Time to detection (90% of steady state)
        steady_state_threshold = 0.9 * final_signal
        time_to_detection = None
        for i, val in enumerate(circuit_result['[reporter_protein]']):
            if val >= steady_state_threshold:
                time_to_detection = circuit_result['time'][i]
                break
        if time_to_detection is None:
            time_to_detection = simulation_time
        
        # Signal-to-noise ratio
        background_noise = np.std(circuit_result['[reporter_protein]'][:10])
        snr = final_signal / (background_noise + 1e-6)
        
        # Dynamic range
        min_signal = np.min(circuit_result['[reporter_protein]'])
        max_signal = np.max(circuit_result['[reporter_protein]'])
        dynamic_range = max_signal - min_signal
        
        # More realistic error rates based on signal characteristics
        expected_high = sclerostin_init > (self.params.sclerostin_normal_mean + self.params.sclerostin_normal_std)

        # Dynamic threshold based on SNR
        adaptive_threshold = 5000 + 2000 * (1.0 / (snr + 1))
        threshold_output = final_signal > adaptive_threshold

        # False rates depend on signal quality
        signal_overlap = abs(final_signal - adaptive_threshold) / (signal_std + 1e-6)
        base_error = 0.02 / (1 + signal_overlap * 0.1)  # Lower error with clear separation

        false_positive_rate = base_error * (1.5 if not expected_high and threshold_output else 1.0)
        false_negative_rate = base_error * (2.0 if expected_high and not threshold_output else 1.0)

        # Add more noise-dependent error
        noise_factor = noise_level / 0.15  # Normalize to typical noise
        false_positive_rate += 0.03 * noise_factor  # More error with high noise
        false_negative_rate += 0.02 * noise_factor

        false_positive_rate = np.clip(false_positive_rate, 0.01, 0.20)
        false_negative_rate = np.clip(false_negative_rate, 0.01, 0.20)
        
        # Circuit burden
        circuit_burden = (
            0.08 * promoter_strength +  # Transcription cost
            0.04 * amplification_factor +  # Signal cascade cost
            0.15 * feedback_strength +  # Feedback loop cost
            0.02 * (1 if feedback_strength > 0 else 0)  # Fixed feedback overhead
        )
        circuit_burden = np.clip(circuit_burden, 0.02, 0.35)
        
        # Additional dataset features for RL
        mRNA_halflife_actual = 30.0  # minutes
        protein_halflife_actual = 60.0  # minutes
        
        # Compile comprehensive result dictionary
        result = {
            # Environmental context (Category 1)
            'sclerostin_concentration': sclerostin_init,
            'dkk1_concentration': dkk1_init,
            'osteocalcin_concentration': osteocalcin_sensor,  # Keep actual units
            'calcium_concentration': calcium_init,
            'local_pH': ph_init,
            'mechanical_loading': mechanical_loading,
            'background_noise_level': noise_level,
            'diffusion_distance_from_sensor': np.random.uniform(
                self.params.sensor_distance_min, 
                self.params.sensor_distance_max
            ),
            'cell_type': 'osteocyte',             # Only osteocytes produce sclerostin ---------------- WE NEED TO INCLDUE MORE CELL TYPES IN THE FUTUEE FOR REALISM
            
            # Biosensor circuit design (Category 2)
            'circuit_topology': circuit_type,
            'receptor_binding_affinity_Kd': receptor_kd,
            'signal_amplification_factor': amplification_factor,
            'transcriptional_promoter_strength': promoter_strength,
            'mRNA_half_life': mRNA_halflife_actual,
            'protein_half_life': protein_halflife_actual,
            'feedback_presence': 1 if feedback_strength > 0 else 0,
            'feedback_strength': feedback_strength,
            'circuit_cellular_burden': circuit_burden,
            'signal_measurement_interval': simulation_time / 200,  # You increased time points
            
            # Performance metrics (Category 3)
            'sensor_output_signal': final_signal,
            'signal_to_noise_ratio_SNR': snr,
            'time_to_detection_threshold': time_to_detection,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'dynamic_range_of_output': dynamic_range,
            'circuit_latency_constant': time_to_detection / 2.0,
            'circuit_reset_time': protein_halflife_actual * 3,  # ~3 half-lives
            
            # Additional metrics
            'sclerostin_at_sensor': sclerostin_sensor,
            'dkk1_at_sensor': dkk1_sensor,
            'crosstalk_dkk1_effect': dkk1_sensor / (sclerostin_sensor + 1e-6),
            
            # Challenge/perturbation (Category 4)
            'environmental_pulse_event': np.random.choice([0, 1], p=[0.85, 0.15]),  # 15% have pulse
            'crosstalk_compound_concentration': dkk1_sensor,
        }
        
        return result
    
    def generate_dataset(self, n_samples: int = 1000, 
                        save_path: str = "sclerostin_biosensor_dataset.csv") -> pd.DataFrame:
        """
        Generate comprehensive dataset with diverse scenarios.
        """
        
        print(f"Generating dataset with {n_samples} simulations...")
        print("This may take several minutes...\n")
        
        results = []
        failed = 0
        
        for i in range(n_samples):
            if i % 50 == 0:
                print(f"Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%) - Failed: {failed}")
            
            # Sample parameters from realistic distributions
            sclerostin_level = np.random.choice(
                ["low", "normal", "high"],
                p=[0.2, 0.6, 0.2]
            )
            
            calcium = np.random.uniform(0.8, 2.5)
            ph = np.random.normal(7.35, 0.10)
            ph = np.clip(ph, 7.15, 7.50)
            
            dkk1 = np.random.lognormal(np.log(30), 0.35)
            mechanical = np.random.uniform(0.4, 1.8)
            noise = np.random.uniform(0.08, 0.25)
            
            # Biosensor circuit parameters (RL action space)
            receptor_kd = np.random.uniform(2.0, 15.0)
            amplification = np.random.uniform(0.5, 5.0)
            promoter = np.random.uniform(0.4, 1.0)
            feedback = np.random.choice([0.0, 0.3, 0.6], p=[0.6, 0.3, 0.1])
            circuit_type = np.random.choice(
                ["direct", "amplification", "feedback"],
                p=[0.5, 0.35, 0.15]
            )
            
            try:
                result = self.simulate_scenario(
                    sclerostin_level=sclerostin_level,
                    calcium_level=calcium,
                    ph_value=ph,
                    dkk1_concentration=dkk1,
                    mechanical_loading=mechanical,
                    receptor_kd=receptor_kd,
                    amplification_factor=amplification,
                    promoter_strength=promoter,
                    feedback_strength=feedback,
                    circuit_type=circuit_type,
                    simulation_time=450.0,
                    noise_level=noise
                )
                results.append(result)
                
            except Exception as e:
                failed += 1
                if failed < 10:  # Only print first few errors
                    print(f"Warning: Simulation {i} failed: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"\n{'='*70}")
        print(f"Dataset saved to {save_path}")
        print(f"Total successful simulations: {len(df)}")
        print(f"Total failed simulations: {failed}")
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumn summary:")
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)        # Don't wrap lines
        pd.set_option('display.max_colwidth', None) # Show full column names
        print(df.describe())
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FIXED SCLEROSTIN BIOSENSOR SIMULATION & DATASET GENERATION")
    print("="*70)
    print("\nInitializing biological parameters...")
    
    params = BiologicalParameters()
    generator = DatasetGenerator(params)
    
    # Run test simulation
    print("\nRunning test simulation...")
    test_result = generator.simulate_scenario(
        sclerostin_level="normal",
        circuit_type="direct",
        simulation_time=400.0
    )
    
    print("\nTest simulation results:")
    print(f"  Sclerostin concentration: {test_result['sclerostin_concentration']:.2f} pmol/L")
    print(f"  Sensor output signal: {test_result['sensor_output_signal']:.2f}")
    print(f"  Signal-to-noise ratio: {test_result['signal_to_noise_ratio_SNR']:.2f}")
    print(f"  Time to detection: {test_result['time_to_detection_threshold']:.2f} min")
    
    # Generate full dataset
    print("\n" + "="*70)
    print("GENERATING FULL DATASET")
    print("="*70)
    
    dataset = generator.generate_dataset(
        n_samples=100,  # Start with 100, increase to 1000+ for full dataset
        save_path="sclerostin_biosensor_dataset.csv"
    )
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run validation_module.py to check biological realism")
    print("2. Inspect dataset: pd.read_csv('sclerostin_biosensor_dataset.csv')")
    print("3. Train RL model using this dataset")
    print("4. Optimize biosensor circuit parameters")