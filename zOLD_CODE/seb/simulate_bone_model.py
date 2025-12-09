"""
SBML Bone Remodeling Model Simulator
Runs multiple disease scenarios and generates plots
"""

import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Conversion factors
NA = 6.022e23
V_bone = 0.01  # L
V_blood = 5.0  # L

def molecules_to_nM(molecules, volume_L):
    """Convert molecule count to nanomolar concentration."""
    return molecules / (NA * volume_L * 1e-9)

def molecules_to_mM(molecules, volume_L):
    """Convert molecule count to millimolar concentration."""
    return molecules / (NA * volume_L * 1e-3)

def print_separator(char='=', width=60):
    print(char * width)

def run_simulation(model_file, scenario_name, modifications=None, duration=4320, points=500):
    """
    Run a simulation scenario.
    
    Parameters:
    - model_file: path to SBML file
    - scenario_name: name of the scenario
    - modifications: dict of parameter changes
    - duration: simulation time in hours
    - points: number of time points
    """
    print(f"\n{'-'*60}")
    print(f"SIMULATION: {scenario_name}")
    print(f"{'-'*60}")
    
    # Load model (use loadSBMLModel for .xml files)
    r = te.loadSBMLModel(model_file)
    
    # Configure integrator for stiff systems
    r.integrator = 'cvode'
    r.integrator.setValue('stiff', True)  # Use stiff solver
    r.integrator.setValue('absolute_tolerance', 1e-12)
    r.integrator.setValue('relative_tolerance', 1e-8)
    r.integrator.setValue('maximum_num_steps', 50000)  # Increase max steps
    r.integrator.setValue('maximum_time_step', 10.0)  # Limit time step to 10 hours
    r.integrator.setValue('minimum_time_step', 1e-6)  # Minimum time step
    
    # Apply modifications
    if modifications:
        print(f"Applying modifications:")
        for param, value in modifications.items():
            try:
                r[param] = value
                print(f"  - {param} = {value}")
            except Exception as e:
                print(f"  WARNING: Could not set {param}: {e}")
    
    # Get initial values
    initial_bone_mass = r['BoneMass']
    initial_osteoblasts = r['Osteoblasts']
    initial_osteoclasts = r['Osteoclasts']
    initial_sclerostin_molecules = r['Sclerostin_ECM']
    initial_sclerostin_nM = molecules_to_nM(initial_sclerostin_molecules, V_bone)
    
    print(f"\nInitial Conditions:")
    print(f"  Bone Mass: {initial_bone_mass:.3f} mg/mm")
    print(f"  Osteoblasts: {initial_osteoblasts:.1f} cells")
    print(f"  Osteoclasts: {initial_osteoclasts:.1f} cells")
    print(f"  Sclerostin: {initial_sclerostin_nM:.3f} nM")
    
    # Run simulation
    print(f"\nRunning simulation for {duration} hours ({duration/24:.1f} days)...")
    
    try:
        result = r.simulate(0, duration, points)
    except RuntimeError as e:
        print(f"⚠️  Simulation failed at full duration: {e}")
        print(f"Attempting shorter simulation (720 hours / 30 days)...")
        try:
            result = r.simulate(0, 720, points)  # Try 1 month instead
            duration = 720  # Update duration for reporting
        except RuntimeError as e2:
            print(f"❌ Simulation failed even at reduced duration: {e2}")
            print(f"This indicates parameter instability. Trying 168 hours (1 week)...")
            result = r.simulate(0, 168, min(points, 100))
            duration = 168
    
    # Get final values
    final_bone_mass = result['BoneMass'][-1]
    final_osteoblasts = result['Osteoblasts'][-1]
    final_osteoclasts = result['Osteoclasts'][-1]
    final_sclerostin_molecules = result['Sclerostin_ECM'][-1]
    final_sclerostin_nM = molecules_to_nM(final_sclerostin_molecules, V_bone)
    final_biomarker_active = result['biomarker_active'][-1]
    
    # Diagnostic: Check for explosive growth
    if final_osteoblasts > 10000 or final_osteoclasts > 1000:
        print(f"\n⚠️  WARNING: Cell populations exploded!")
        print(f"    This indicates parameter imbalance in differentiation rates.")
    if final_bone_mass > 100 or final_bone_mass < 0.1:
        print(f"\n⚠️  WARNING: Bone mass out of physiological range!")
        print(f"    Formation/resorption rates may be unbalanced.")
    if final_sclerostin_nM > 100:
        print(f"\n⚠️  WARNING: Sclerostin concentration unrealistic!")
        print(f"    Production/degradation rates need adjustment.")
    
    # Calculate changes
    bone_change = ((final_bone_mass - initial_bone_mass) / initial_bone_mass) * 100
    
    print(f"\nFinal Conditions:")
    print(f"  Bone Mass: {final_bone_mass:.3f} mg/mm")
    print(f"  Change: {bone_change:+.2f}%")
    print(f"  Osteoblasts: {final_osteoblasts:.1f} cells")
    print(f"  Osteoclasts: {final_osteoclasts:.1f} cells")
    print(f"  Sclerostin: {final_sclerostin_nM:.3f} nM")
    print(f"  Biomarker Active: {'YES' if final_biomarker_active > 0.5 else 'NO'}")
    
    # Check for fracture risk
    if final_bone_mass < 3.0:
        print(f"  ⚠️  WARNING: Bone mass below fracture threshold (3.0 mg/mm)!")
    
    return result, r

def plot_results(results_dict, filename='bone_model_simulation_results.png'):
    """
    Create comprehensive plots of simulation results.
    """
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    scenarios = list(results_dict.keys())
    colors = ['blue', 'red', 'purple', 'orange']
    
    # 1. Bone Mass
    ax1 = fig.add_subplot(gs[0, :])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        ax1.plot(time_days, result['BoneMass'], label=scenario, color=colors[i], linewidth=2)
    ax1.axhline(y=3.0, color='red', linestyle='--', label='Fracture Threshold', linewidth=1)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Bone Mass (mg/mm)')
    ax1.set_title('Bone Mass Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cell Populations
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        ax2.plot(time_days, result['Osteoblasts'], label=scenario, color=colors[i], linewidth=2)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Cell Count')
    ax2.set_title('Osteoblasts')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        ax3.plot(time_days, result['Osteoclasts'], label=scenario, color=colors[i], linewidth=2)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Cell Count')
    ax3.set_title('Osteoclasts')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 2])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        ax4.plot(time_days, result['Osteocytes'], label=scenario, color=colors[i], linewidth=2)
    ax4.axhline(y=250, color='red', linestyle='--', label='Min Threshold', linewidth=1)
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Cell Count')
    ax4.set_title('Osteocytes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 3. Sclerostin (convert to nM)
    ax5 = fig.add_subplot(gs[2, 0])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        sclerostin_nM = molecules_to_nM(result['Sclerostin_ECM'], V_bone)
        ax5.plot(time_days, sclerostin_nM, label=scenario, color=colors[i], linewidth=2)
    ax5.axhline(y=1.5, color='red', linestyle='--', label='Biomarker Threshold', linewidth=1)
    ax5.set_xlabel('Time (days)')
    ax5.set_ylabel('Sclerostin (nM)')
    ax5.set_title('Sclerostin Concentration')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 4. PTH (convert to nM)
    ax6 = fig.add_subplot(gs[2, 1])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        PTH_nM = molecules_to_nM(result['PTH_blood'], V_blood)
        ax6.plot(time_days, PTH_nM, label=scenario, color=colors[i], linewidth=2)
    ax6.set_xlabel('Time (days)')
    ax6.set_ylabel('PTH (nM)')
    ax6.set_title('Parathyroid Hormone')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 5. Biomarker Activation
    ax7 = fig.add_subplot(gs[2, 2])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        ax7.plot(time_days, result['biomarker_active'], label=scenario, color=colors[i], linewidth=2)
    ax7.set_xlabel('Time (days)')
    ax7.set_ylabel('Activation State')
    ax7.set_title('Biomarker Activation')
    ax7.set_ylim(-0.1, 1.1)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 6. RANKL/OPG Ratio
    ax8 = fig.add_subplot(gs[3, 0])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        RANKL_nM = molecules_to_nM(result['RANKL_soluble'], V_bone)
        OPG_nM = molecules_to_nM(result['OPG_ECM'], V_bone)
        ratio = RANKL_nM / (OPG_nM + 0.1)  # Avoid division by zero
        ax8.plot(time_days, ratio, label=scenario, color=colors[i], linewidth=2)
    ax8.set_xlabel('Time (days)')
    ax8.set_ylabel('RANKL/OPG Ratio')
    ax8.set_title('RANKL/OPG Ratio (Resorption Signal)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 7. Estrogen (convert to nM)
    ax9 = fig.add_subplot(gs[3, 1])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        estrogen_nM = molecules_to_nM(result['Estrogen_blood'], V_blood)
        ax9.plot(time_days, estrogen_nM, label=scenario, color=colors[i], linewidth=2)
    ax9.set_xlabel('Time (days)')
    ax9.set_ylabel('Estrogen (nM)')
    ax9.set_title('Estrogen Levels')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 8. Wnt Signaling
    ax10 = fig.add_subplot(gs[3, 2])
    for i, (scenario, (result, _)) in enumerate(results_dict.items()):
        time_days = result['time'] / 24
        ax10.plot(time_days, result['Wnt_LRP_active'], label=scenario, color=colors[i], linewidth=2)
    ax10.set_xlabel('Time (days)')
    ax10.set_ylabel('Activity Level')
    ax10.set_title('Wnt Signaling (Anabolic)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to '{filename}'")
    plt.close()

def main():
    print_separator('=', 60)
    print("BONE REMODELING MODEL SIMULATION")
    print_separator('=', 60)
    
    model_file = 'bone_remodeling_model.xml'
    
    print(f"\nLoading SBML model from '{model_file}'...")
    
    # Simulation parameters
    duration = 4320  # hours (6 months)
    points = 500
    
    results = {}
    
    # 1. Normal Conditions (Baseline)
    results['Normal'] = run_simulation(
        model_file,
        'Normal Conditions (Earth Gravity)',
        modifications={},
        duration=duration,
        points=points
    )
    
    # 2. Spaceflight (Microgravity)
    results['Spaceflight'] = run_simulation(
        model_file,
        'Spaceflight (Microgravity)',
        modifications={
            'Gravity': 0,  # No gravity
            'ExerciseLoad': 100,  # Reduced exercise capability
        },
        duration=duration,
        points=points
    )
    
    # 3. Postmenopausal Osteoporosis
    # Start with reduced estrogen
    results['Osteoporosis'] = run_simulation(
        model_file,
        'Postmenopausal Osteoporosis',
        modifications={
            'Estrogen_blood': 0.1 * (NA * V_blood * 1e-9),  # 0.1 nM in molecules
        },
        duration=duration,
        points=points
    )
    
    # 4. CKD-MBD (Chronic Kidney Disease - Mineral Bone Disorder)
    results['CKD-MBD'] = run_simulation(
        model_file,
        'CKD-MBD',
        modifications={
            'PTH_dysregulation_factor': 2.5,  # Elevated PTH
            'k_phosphate_clearance': 0.001,  # Reduced phosphate clearance
        },
        duration=duration,
        points=points
    )
    
    # Generate plots
    plot_results(results, 'bone_model_simulation_results.png')
    
    # Print summary
    print_separator('=', 60)
    print("SIMULATION SUMMARY")
    print_separator('=', 60)
    print("\nExpected physiological ranges:")
    print("  Bone Mass: 3.0-5.0 mg/mm (fracture threshold: 3.0)")
    print("  Osteoblasts: 100-1000 cells (baseline ~500)")
    print("  Osteoclasts: 10-200 cells (baseline ~50)")
    print("  Sclerostin: 0.3-3.0 nM (biomarker threshold: 1.5)")
    print("  PTH: 0.002-0.050 nM")
    
    print("\nFinal Results:")
    for scenario, (result, _) in results.items():
        final_bone_mass = result['BoneMass'][-1]
        initial_bone_mass = result['BoneMass'][0]
        bone_change = ((final_bone_mass - initial_bone_mass) / initial_bone_mass) * 100
        final_sclerostin_nM = molecules_to_nM(result['Sclerostin_ECM'][-1], V_bone)
        final_biomarker = result['biomarker_active'][-1]
        
        print(f"\n{scenario}:")
        print(f"  Initial: {initial_bone_mass:.3f} mg/mm")
        print(f"  Final:   {final_bone_mass:.3f} mg/mm")
        print(f"  Change:  {bone_change:+.2f}%")
        print(f"  Sclerostin: {final_sclerostin_nM:.3f} nM")
        print(f"  Biomarker: {'ACTIVATED' if final_biomarker > 0.5 else 'Not activated'}")
    
    print_separator('=', 60)
    print("SIMULATION COMPLETE")
    print_separator('=', 60)

if __name__ == '__main__':
    main()