#!/usr/bin/env python
"""Test v2_rl surrogates on the previous best config."""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
sys.path.insert(0, 'BO')

from BO.core.surrogate_loader_v2_rl import SurrogateLoaderV2RL
from BO.evaluation.physics_forward_model import PhysicsForwardModel

# Load best config from previous run
with open('BO/bo_results/results/best_config.json') as f:
    best_config = json.load(f)

design = best_config['biosensor_design']
env = best_config['measurement_environment']

print('='*70)
print('TESTING V2_RL SURROGATES ON PREVIOUS BEST CONFIG')
print('='*70)

print(f'\nConfiguration:')
print(f'  Type: {design["type"]}')
print(f'  Kd: {design["kd_nm"]:.4f} nM')
print(f'  Sensitivity: {design["sensitivity"]:.4f}')
print(f'  Noise: {env["noise_preset"]}')
print(f'  Scenario: {env["target_scenario"]}')

# Get v2_rl predictions
print(f'\nLoading v2_rl surrogates...')
loader = SurrogateLoaderV2RL()

# Initialize physics model
physics = PhysicsForwardModel()

# Compute SNR
snr_db = physics.estimate_snr(
    biosensor_type=design['type'],
    kd_nm=design['kd_nm'],
    sensitivity=design['sensitivity'],
    response_time_s=design.get('response_time_s', 500.0),
    noise_preset=env['noise_preset'],
    target_scenario=env['target_scenario']
)

print(f'Computed SNR: {snr_db:.2f} dB')

# Encode and predict
biosensor_enc, noise_enc = loader.encode_categorical_v2(
    design['type'],
    env['noise_preset']
)

X_raw = np.array([[snr_db, biosensor_enc, noise_enc]], dtype=np.float32)
print(f'Raw features: [SNR={snr_db:.2f}, biosensor_enc={biosensor_enc:.1f}, noise_enc={noise_enc:.1f}]')

X_scaled = loader.scaler.transform(X_raw)
print(f'Scaled features: {X_scaled[0]}')

dr_v2, fnr_v2, ttd_v2 = loader.predict_metrics_v2(X_scaled)

print(f'\nv2_RL PREDICTIONS:')
print(f'  Detection Rate: {dr_v2:.4f}')
print(f'  False Negative Rate: {fnr_v2:.4f}')
print(f'  Time-to-Detection: {ttd_v2:.1f}s')

print(f'\nv1 (OLD - OVERCONFIDENT):')
print(f'  Detection Rate: 1.0000')
print(f'  False Negative Rate: 0.0000')
print(f'  Time-to-Detection: 400.0s')

print(f'\nACTUAL SIMULATOR RESULT:')
print(f'  Detection Rate: 0.0000')
print(f'  False Negative Rate: 1.0000')
print(f'  Time-to-Detection: 3600.0s')

print(f'\n' + '='*70)
print('ASSESSMENT')
print('='*70)

v2_error = abs(dr_v2 - 0.0)
v1_error = abs(1.0 - 0.0)

print(f'\nDetection Rate Prediction Error:')
print(f'  v1 (old): |1.0 - 0.0| = {v1_error:.1f} (TERRIBLE)')
print(f'  v2 (new): |{dr_v2:.4f} - 0.0| = {v2_error:.4f} (Much better!)')

if dr_v2 < 0.5:
    print(f'\n[OK] v2_RL predicts conservative detection rate')
    print(f'  (More realistic than v1 overconfident 1.0)')
    print(f'  (BO will focus on better configs)')
else:
    print(f'\n[WARN] v2_RL still predicting moderately high')

print(f'\nKEY INSIGHT:')
print(f'  v2_RL uses SNR={snr_db:.2f} dB')
print(f'  SNR is computed from physics, not regressed')
print(f'  Surrogates trained on [SNR, biosensor, noise]')
print(f'  This is RL\'s proven, working approach')
print(f'\n  v1 tried to predict from [kd, sensitivity, ...]')
print(f'  Those parameters don\'t exist in training data!')
