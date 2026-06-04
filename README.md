# GENEVO2 -- Bone Microenvironment Biosensor Simulation Framework

## Project Structure

```
bone_biosensor_project/
├── models/
│   ├── __init__.py
│   ├── bone_environment.ant          # Antimony model for bone microenvironment
│   ├── biosensors.py                 # Biosensor circuit implementations
│   ├── environment_configs.py        # Scenario configurations
│   └── noise.py                      # Noise modeling functions
├── simulation/
│   ├── __init__.py
│   ├── simulator.py                  # Tellurium simulation wrapper
│   └── biosensor_engine.py           # Biosensor measurement engine
├── dataset/
│   ├── __init__.py
│   └── generator.py                  # Dataset generation orchestrator
├── utils/
│   ├── __init__.py
│   ├── logging_config.py             # Logging configuration
│   └── validators.py                 # Model validation functions
├── logs/                             # Runtime logs (created at runtime)
├── data/                             # Generated datasets (created at runtime)
│   ├── metadata/                     # JSON config files
│   ├── timeseries/                   # CSV time-series data
│   └── master_index.csv              # Master dataset index
├── main.py                           # Main execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
tellurium>=2.2.8
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
h5py>=3.0.0
pyarrow>=5.0.0
```

## Usage

Typical Scenarios
Scenario 1: Quick Test
bashpython main.py --n_simulations 100 --output_dir data_test
# Runtime: ~2 minutes
# Use: To test the system works
Scenario 2: ML Training
bashpython main.py --n_simulations 5000 --seed 42
# Runtime: ~50 minutes
# Use: Production dataset for surrogate training
Scenario 3: High-Quality ML
bashpython main.py --n_simulations 10000 --seed 42 --noise_distribution 0.3,0.6,0.1
# Runtime: ~100 minutes
# Use: Best ML model performance
Scenario 4: Reproducible Results
bashpython main.py --n_simulations 5000 --seed 42
python main.py --n_simulations 5000 --seed 42  # Re-run
# Both produce IDENTICAL results

Troubleshooting
"ModuleNotFoundError: No module named 'tellurium'"
bashpip install tellurium
"Could not find models/bone_environment.ant"

Make sure you're running from project root
File created automatically, check permissions

"Very slow generation" (>1 hour for 5000)

Normal - Tellurium simulations are CPU-intensive
Or: Use fewer simulations (--n_simulations 1000)
Or: Run in background: nohup python main.py &

Validation shows "High censoring rate"

Check detection rates in report
If <60%, parameter issue (see README)

Dataset size too large
bash# Move data to different location
mv data /mnt/large_drive/data
# Adjust output_dir in code

Next Steps
After dataset is generated and validated:

Train surrogate models

bash   python rl_model.py --mode train_surrogates

Run RL optimization

bash   python rl_model.py --mode train_rl --timesteps 5000000

Analyze results

Check RL/sclerostin_rl_results/
Review surrogate model performance
Examine optimized biosensor designs




Performance Tips
Make it Faster
bash# Fewer time points (less detail, faster)
python main.py --num_points 100

# Shorter simulation (30 min instead of 60)
python main.py --duration 1800

# Smaller dataset
python main.py --n_simulations 1000
Make it Better
bash# Larger dataset (better ML training)
python main.py --n_simulations 10000

# More balanced noise
python main.py --noise_distribution 0.33,0.34,0.33

# Reproducible
python main.py --seed 42
Parallel Execution
bash# Run multiple in parallel
python main.py --n_simulations 2500 --output_dir data_part1 &
python main.py --n_simulations 2500 --output_dir data_part2 &
python main.py --n_simulations 2500 --output_dir data_part3 &
python main.py --n_simulations 2500 --output_dir data_part4 &

wait  # Wait for all to complete

### Configuration

The framework simulates three scenarios:
1. **Healthy**: Normal bone homeostasis
2. **PMO** (Post-Menopausal Osteoporosis): Low estrogen, high RANKL, low OPG
3. **CKD-MBD** (Chronic Kidney Disease-Mineral Bone Disorder): Dysregulated PTH, high sclerostin

## Output

Each simulation generates:
- `data/metadata/{run_id}.json` - Configuration and parameters
- `data/timeseries/{run_id}.csv` - Time-series data
- `data/master_index.csv` - Index of all simulations

## Biological Model

The Antimony model implements:
- Sclerostin (SOST) production by osteocytes
- RANKL/OPG signaling axis
- Osteoblast and osteoclast dynamics
- Estrogen and PTH regulatory effects
- Compartmental transport (bone → sensor chamber)

## Biosensor Configurations

Multiple biosensor circuit types:
- Direct binding sensors
- Amplifying enzyme cascade sensors
- Threshold-based digital sensors
- Ratiometric multi-analyte sensors

## References

All biological parameters are sourced from peer-reviewed literature.
See inline citations in code for specific references.