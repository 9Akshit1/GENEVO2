# GENEVO2 Redesigned RL Environment Specification

## Overview

This document specifies a **stage-based sequential biosensor design environment** for sclerostin detection optimization.

**Core Principle**: The agent makes 5 sequential design decisions, each constraining downstream choices, resulting in meaningful hierarchical optimization.

---

## Episode Structure

Each episode consists of exactly **5 stages** (not 2048 steps):

1. **Stage 1: Architecture Selection** (discrete choice)
2. **Stage 2: Biochemical Sensitivity Tuning** (continuous parameters)
3. **Stage 3: Signal Processing Design** (continuous parameters)
4. **Stage 4: Detection Logic Tuning** (continuous parameters)
5. **Stage 5: Robustness Validation** (evaluation phase, no action)

**Episode length**: 5 steps total (not 2048)

**Reward**: Computed once at end of episode (not every step)

---

## Stage 1: Architecture Selection

**Action Type**: Discrete (one-hot categorical)

**Available Architectures**:

| ID | Name | Characteristics | Amplification | Complexity |
|----|----|---|---|---|
| 0 | direct_electrochemical | Raw electrochemical sensing, no inherent amplification | 1.0× | Low |
| 1 | nanoparticle_amplified | NP-based electrochemical amplification | 5–20× | Medium |
| 2 | enzyme_amplified | Enzymatic cascade amplification | 10–50× | High |
| 3 | redox_cycling | Redox-cycling electrochemical amplification | 3–15× | Medium |

**Agent Action**: `architecture_id ∈ {0, 1, 2, 3}`

**State Update**:
```python
state['architecture'] = architecture_id
state['base_amplification'] = architecture_amplification[architecture_id]
state['stage'] = 1

# Architecture constrains downstream viable ranges
state['constraints']['gain_range'] = get_viable_gains(architecture_id)
state['constraints']['filter_range'] = get_viable_filters(architecture_id)
state['constraints']['threshold_range'] = get_viable_thresholds(architecture_id)
```

**Observation to Agent**:
- Current architecture choice
- Implied amplification baseline
- Remaining design budget (5 - stage_num)

---

## Stage 2: Biochemical Sensitivity Tuning

**Action Type**: Continuous, 3-dimensional

**Parameters**:

| Parameter | Range | Units | Meaning |
|-----------|-------|-------|---------|
| `kd` | [0.5, 2.0] | nM | Binding affinity (dissociation constant) |
| `hill_coeff` | [2, 8] | - | Cooperativity/sharpness of activation |
| `sensitivity_scale` | [0.5, 5.0] | - | Output amplification from binding |

**Agent Action**: `[kd, hill_coeff, sensitivity_scale]` (continuous, normalized to [-1, 1])

**State Update**:
```python
state['kd'] = clip(kd, 0.5, 2.0)
state['hill_coeff'] = clip(hill_coeff, 2, 8)
state['sensitivity_scale'] = clip(sensitivity_scale, 0.5, 5.0)
state['stage'] = 2

# Predict biochemical performance
state['predicted_snr_biochem'] = surrogate_snr(
    kd, hill_coeff, sensitivity_scale, architecture
)
state['predicted_separation'] = surrogate_separation(
    kd, hill_coeff, state['architecture']
)
```

**Observation to Agent**:
- Current biochemical parameters (kd, hill_coeff, sensitivity_scale)
- Predicted SNR from biochemistry alone
- Predicted disease separation (DR gap between healthy/PMO/CKD-MBD)
- Uncertainty estimates from surrogates
- Architecture-based constraints still apply

---

## Stage 3: Signal Processing Design

**Action Type**: Continuous, 4-dimensional

**Parameters**:

| Parameter | Range | Units | Meaning |
|-----------|-------|-------|---------|
| `gain` | Arch-dependent | - | Electrochemical amplification factor |
| `smoothing` | [0.0, 1.0] | - | Temporal smoothing (0=none, 1=max) |
| `filter_strength` | [0.0, 5.0] | - | Low-pass filter cutoff (Hz) |
| `averaging_window` | [1, 100] | samples | Temporal averaging window |

**Viable Ranges by Architecture**:
- `direct_electrochemical`: gain ∈ [0.1, 2.0] (limited by physics)
- `nanoparticle_amplified`: gain ∈ [1.0, 10.0]
- `enzyme_amplified`: gain ∈ [2.0, 50.0]
- `redox_cycling`: gain ∈ [1.0, 15.0]

**Agent Action**: `[gain, smoothing, filter_strength, averaging_window]` (continuous)

**State Update**:
```python
state['gain'] = clip(gain, viable_min, viable_max)
state['smoothing'] = clip(smoothing, 0.0, 1.0)
state['filter_strength'] = clip(filter_strength, 0.0, 5.0)
state['averaging_window'] = clip(int(averaging_window), 1, 100)
state['stage'] = 3

# Predict signal processing impact
state['predicted_snr_after_processing'] = surrogate_snr(
    state['kd'], state['hill_coeff'], state['sensitivity_scale'],
    state['gain'], state['smoothing'], state['filter_strength'],
    state['architecture']
)
state['predicted_noise_floor'] = estimate_noise(
    state['gain'], state['filter_strength']
)
state['predicted_ttd'] = estimate_ttd(
    state['gain'], state['averaging_window']
)
```

**Observation to Agent**:
- Current signal processing parameters
- Predicted SNR after processing
- Estimated noise floor
- Estimated time-to-detection
- Uncertainty estimates
- Architecture constraints still apply

---

## Stage 4: Detection Logic Tuning

**Action Type**: Continuous, 4-dimensional

**Parameters**:

| Parameter | Range | Units | Meaning |
|-----------|-------|-------|---------|
| `threshold` | [0.0, 1.0] | normalized | Detection activation threshold |
| `debounce` | [0, 10] | seconds | Hysteresis to prevent chatter |
| `persistence_window` | [1, 100] | samples | Confirmatory window for detection |
| `confidence_threshold` | [0.0, 1.0] | - | Minimum confidence for positive call |

**Agent Action**: `[threshold, debounce, persistence_window, confidence_threshold]` (continuous)

**State Update**:
```python
state['threshold'] = clip(threshold, 0.0, 1.0)
state['debounce'] = clip(debounce, 0, 10)
state['persistence_window'] = clip(int(persistence_window), 1, 100)
state['confidence_threshold'] = clip(confidence_threshold, 0.0, 1.0)
state['stage'] = 4

# Predict detection logic impact (via surrogate models)
state['predicted_dr'] = surrogate_detection_rate(
    state['kd'], state['hill_coeff'], state['sensitivity_scale'],
    state['gain'], state['smoothing'], state['filter_strength'],
    state['threshold'], state['debounce'], state['architecture']
)
state['predicted_fnr'] = surrogate_fnr(...)
state['predicted_fpr'] = surrogate_fpr(...)
```

**Observation to Agent**:
- Current detection logic parameters
- Predicted detection rate
- Predicted false negative rate
- Predicted false positive rate
- All previous parameters (architecture, biochemistry, signal processing)
- Uncertainty estimates

---

## Stage 5: Robustness Validation & Reward

**Action Type**: None (automatic evaluation)

**Process**:

Evaluate the complete biosensor design across all realistic conditions:

1. **Disease conditions**: Healthy, PMO, CKD-MBD
2. **Noise scenarios**: Low noise, medium noise, high noise
3. **Parameter robustness**: ±10% perturbations
4. **Temporal robustness**: Drift over time (simulated)

**Final Evaluation Metrics**:

```python
# For each condition, compute:
detection_rates = {}  # per condition
fnr_rates = {}        # per condition
fpr_rates = {}        # per condition
ttd_values = {}       # per condition

# Aggregate across all conditions
mean_dr = mean(detection_rates.values())
mean_fnr = mean(fnr_rates.values())
mean_fpr = mean(fpr_rates.values())
mean_ttd = mean(ttd_values.values())

# Robustness = low variance across conditions
robustness_penalty = std(detection_rates.values())

# Overall fitness
fitness = (
    0.40 * mean_dr                    # maximize detection rate
    - 0.30 * mean_fnr                 # minimize false negatives
    - 0.20 * normalized_ttd           # minimize time to detection
    - 0.10 * robustness_penalty       # penalize high variance
)
```

**Episode Reward**:

```python
reward = fitness  # Single reward at end of episode

# Uncertainty penalty (prevent surrogate exploitation)
surrogate_uncertainty = estimate_ensemble_variance(state)
reward -= 0.05 * surrogate_uncertainty

# Physical constraint penalties
reward -= check_constraint_violations(state)  # Hard constraints

return reward
```

---

## Constraints (CRITICAL)

### Physical Constraints

Hard bounds on parameter ranges (agent actions are clipped):

```python
PHYSICAL_BOUNDS = {
    'kd': (0.5, 2.0),
    'hill_coeff': (2, 8),
    'sensitivity_scale': (0.5, 5.0),
    'gain': (0.1, 50.0),  # depends on architecture
    'smoothing': (0.0, 1.0),
    'filter_strength': (0.0, 5.0),
    'averaging_window': (1, 100),
    'threshold': (0.0, 1.0),
    'debounce': (0, 10),
    'persistence_window': (1, 100),
    'confidence_threshold': (0.0, 1.0),
}
```

### Statistical Constraints

**Out-of-distribution penalty**:

```python
# Estimate training data density at current state
density = estimate_density(state_features, training_data)

if density < confidence_threshold:
    # Penalize OOD exploration
    reward -= uncertainty_penalty_coefficient * (1 - density)
    # OR truncate episode
    terminated = True
```

### Architecture-Specific Constraints

Example: Direct binding has lower gain range than amplified architectures

```python
ARCHITECTURE_CONSTRAINTS = {
    'direct_electrochemical': {
        'gain_range': (0.1, 2.0),
        'max_amplification': 1.0,
    },
    'nanoparticle_amplified': {
        'gain_range': (1.0, 10.0),
        'max_amplification': 20.0,
    },
    'enzyme_amplified': {
        'gain_range': (2.0, 50.0),
        'max_amplification': 50.0,
    },
    'redox_cycling': {
        'gain_range': (1.0, 15.0),
        'max_amplification': 15.0,
    },
}
```

---

## State Representation

**Full State Vector** (fed to agent as observation):

```python
state = {
    # Stage information
    'stage': int,  # 1-5
    'steps_remaining': int,  # 5 - stage
    
    # Architecture (from Stage 1)
    'architecture': int,  # 0-3
    'base_amplification': float,
    
    # Biochemistry (from Stage 2)
    'kd': float,
    'hill_coeff': float,
    'sensitivity_scale': float,
    'predicted_snr_biochem': float,
    'predicted_separation': float,
    
    # Signal Processing (from Stage 3)
    'gain': float,
    'smoothing': float,
    'filter_strength': float,
    'averaging_window': int,
    'predicted_snr_processing': float,
    'predicted_noise_floor': float,
    'predicted_ttd': float,
    
    # Detection Logic (from Stage 4)
    'threshold': float,
    'debounce': float,
    'persistence_window': int,
    'confidence_threshold': float,
    'predicted_dr': float,
    'predicted_fnr': float,
    'predicted_fpr': float,
    
    # Uncertainty (at every stage)
    'surrogate_uncertainty': float,  # ensemble variance
    'data_density': float,  # training data density at current state
    
    # Constraints
    'constraints': dict,  # current viable parameter ranges
}
```

**Observation Space**: Normalized vector of continuous and embedded categorical features

---

## Rewards Summary

| Event | Reward | When |
|-------|--------|------|
| High detection rate | +0.40 | End of episode |
| Low FNR | +0.30 | End of episode |
| Low TTD | +0.20 | End of episode |
| High robustness | +0.10 | End of episode |
| OOD exploration | -0.05 | Any stage |
| Constraint violation | -0.5–1.0 | Any stage (truncates) |

---

## Key Design Principles

1. **Sequential causality**: Earlier decisions constrain later ones
2. **Short episodes**: 5 steps instead of 2048
3. **Semantic actions**: Each action is a real engineering decision
4. **Rich observation**: Agent sees predicted metrics after each stage
5. **End-of-episode reward**: Reflects real design evaluation
6. **Constraint enforcement**: Hard bounds + uncertainty penalties
7. **Science first**: Design matches actual biosensor engineering workflow

---

## What Changed From Old Environment

| Aspect | Old | New |
|--------|-----|-----|
| Episode length | 2048 steps | 5 steps |
| Action semantics | Random deltas | Real design decisions |
| Reward timing | Every step | End of episode |
| Observation richness | Static | Dynamic (grows each stage) |
| Constraints | None | Hard physical + statistical |
| Architecture | Fixed | Agent-chosen (discrete) |
| State causality | Independent | Hierarchical/causal |

---

## Implementation Checklist

- [ ] Define exact surrogate interface (take state dict, return predictions + uncertainty)
- [ ] Implement stage-based observation normalization
- [ ] Add constraint checking and clipping
- [ ] Implement OOD detection and uncertainty quantification
- [ ] Rewrite reward function
- [ ] Shorten episodes to 5 steps
- [ ] Test with dummy agent (random actions)
- [ ] Verify state transitions are correct
- [ ] Log full reward decomposition per episode
