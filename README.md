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

### Generate Dataset

```bash
# Generate 1000 simulations (default)
python main.py

# Generate custom number of simulations
python main.py --n_simulations 5000

# Parallel execution with 8 cores
python main.py --n_simulations 10000 --n_workers 8

# Verbose logging
python main.py --log_level DEBUG
```

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