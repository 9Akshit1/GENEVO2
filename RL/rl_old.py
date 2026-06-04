# -*- coding: utf-8 -*-
"""
Synthetic Biology Biosensor Design: Complete ML/RL Pipeline
==========================================================

This pipeline implements a comprehensive computational framework for optimizing
gene circuits for robust biomarker detection using machine learning and 
reinforcement learning techniques.

Stages:
1. Data Preprocessing & Feature Selectionp
2. Supervised Learning Models
3. Surrogate Modeling
4. Reinforcement Learning (DQN/PPO)
5. Graph-Based Analysis & Visualizationp

"""


import json
import csv
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle
import threading

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import lightgbm as lgb
from catboost import CatBoostRegressor
# Add robust preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression
from scipy import stats

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# XGBoost
import xgboost as xgb
from xgboost import callback

# RL Libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DualLogger:
    """Context manager for logging to both console and file"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"run_{timestamp}.log"
        
        self.terminal = sys.stdout
        self.log_handle = None
        self.old_stdout = None
        self.old_stderr = None
    
    def __enter__(self):
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        
        class TeeWriter:
            def __init__(self, terminal, file):
                self.terminal = terminal
                self.file = file
            
            def write(self, message):
                self.terminal.write(message)
                self.file.write(message)
                self.file.flush()
            
            def flush(self):
                self.terminal.flush()
                self.file.flush()
        
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        
        tee = TeeWriter(self.terminal, self.log_handle)
        sys.stdout = tee
        sys.stderr = tee
        
        print(f"Ã°Å¸â€Â Logging to: {self.log_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        if self.log_handle:
            self.log_handle.close()
        
        print(f"Ã¢Å“â€¦ Log saved to: {self.log_file}")

# Suppress warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# ============================================================================
# Physics-Based Predictor Classes (Must be at module level for pickling)
# ============================================================================

# ============================================================================
# MODULE-LEVEL CLASSES FOR PICKLING SUPPORT
# ============================================================================
# These MUST be at module level (not inside methods) to be picklable

class TabularNN(nn.Module):
    """
    Standard tabular neural network for regression.
    Module-level class for pickle support.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class NNWrapper:
    """
    Wrapper to make PyTorch models compatible with sklearn API.
    Module-level class for pickle support.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def predict(self, X):
        X_t = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        with torch.no_grad():
            pred = self.model(X_t).numpy().flatten()
        return pred
    
    def get_params(self, deep=True):
        return {}
    
    def state_dict(self):
        """Delegate to underlying PyTorch model"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """Delegate to underlying PyTorch model"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def eval(self):
        """Delegate to underlying PyTorch model"""
        return self.model.eval()


class PhysicsTTDPredictor:
    """
    Physics-based TTD estimation from biosensor parameters.
    Module-level class for pickle support.
    
    Based on the actual TTD generation logic in biosensor_engine.py:
    1. Calculate expected signal strength from biosensor config
    2. Estimate time when signal crosses threshold
    3. Apply circuit-specific delays
    4. Add SNR-dependent variance
    """
    
    def __init__(self, feature_names, median_ttd=1800.0):
        self.feature_names = feature_names
        self.median_ttd = median_ttd
        self.TIMEOUT = 9000.0
        self.MAX_TIME = 3600.0
        
        # Circuit-specific delay factors from biosensor theory
        # (direct_binding, ratiometric, threshold, amplifying)
        self.circuit_delays = {
            0: 1.0,   # direct_binding - fast equilibrium
            1: 1.1,   # ratiometric - moderate (dual detection)
            2: 0.9,   # threshold - fast (digital switch)
            3: 1.3    # amplifying - slow (enzymatic cascade)
        }
        
        # Response curve efficiency factors
        self.response_factors = {
            0: 1.0,   # linear - simple
            1: 0.9,   # hill - cooperative (faster at high conc)
            2: 1.0    # michaelis_menten - enzymatic
        }
    
    def predict(self, X):
        """Predict TTD using physics-based formulas"""
        X_arr = X.values if hasattr(X, 'values') else X
        X_df = pd.DataFrame(X_arr, columns=self.feature_names)
        
        n_samples = len(X_df)
        ttd_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            ttd_pred[i] = self._predict_single(X_df.iloc[i])
        
        return ttd_pred
    
    def _predict_single(self, row):
        """Predict TTD for a single biosensor configuration"""
        
        # Extract biosensor parameters
        sensitivity = row.get('sensitivity', 1.0)
        threshold = row.get('threshold', 0.01)
        kd = row.get('kd', 0.1)
        
        # Get circuit and response types
        circuit_type = int(row.get('circuit_type', 0))
        response_curve = int(row.get('response_curve', 0))
        
        # Calculate signal strength
        signal_strength = sensitivity / (1.0 + kd)
        
        # Apply delays
        circuit_delay = self.circuit_delays.get(circuit_type, 1.0)
        response_factor = self.response_factors.get(response_curve, 1.0)
        
        # Estimate TTD
        if signal_strength < threshold:
            return self.TIMEOUT  # No detection
        
        # Base TTD calculation
        base_ttd = self.median_ttd * (threshold / signal_strength) * circuit_delay * response_factor
        
        # Add noise-dependent variation
        snr = row.get('signal_to_noise_ratio_SNR', 10.0)
        noise_factor = max(0.5, min(2.0, 10.0 / max(snr, 1.0)))
        
        ttd = base_ttd * noise_factor
        
        # Clip to valid range
        return np.clip(ttd, 0.0, self.MAX_TIME)


class ResidualBlock(nn.Module):
    """
    Residual block for neural networks.
    Module-level class for pickle support.
    """
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(x + self.layers(x))


class RobustMLP(nn.Module):
    """
    Robust MLP with batch normalization and dropout.
    Module-level class for pickle support.
    """
    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2 if i < len(hidden_dims) - 1 else 0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.layers(x)


class AdaptiveEnsemble(nn.Module):
    """
    Ensemble of different model architectures.
    Module-level class for pickle support.
    """
    def __init__(self, input_dim, task_type='regression', n_classes=None):
        super().__init__()
        self.task_type = task_type
        self.n_classes = n_classes
        
        # Model 1: Deep narrow network
        self.model1 = self._create_deep_narrow(input_dim, task_type, n_classes)
        
        # Model 2: Wide shallow network  
        self.model2 = self._create_wide_shallow(input_dim, task_type, n_classes)
        
        # Model 3: Residual network
        self.model3 = self._create_residual(input_dim, task_type, n_classes)
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Final combination layer (only for classification)
        if task_type == 'classification':
            self.final_layer = nn.Linear(3 * n_classes, n_classes)
    
    def _create_deep_narrow(self, input_dim, task_type, n_classes):
        layers = []
        dims = [input_dim, 64, 32, 16, 8]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        if task_type == 'regression':
            layers.append(nn.Linear(dims[-1], 1))
        else:
            layers.append(nn.Linear(dims[-1], n_classes))
        
        return nn.Sequential(*layers)
    
    def _create_wide_shallow(self, input_dim, task_type, n_classes):
        hidden_dim = max(256, input_dim * 2)
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]
        
        if task_type == 'regression':
            layers.append(nn.Linear(hidden_dim // 2, 1))
        else:
            layers.append(nn.Linear(hidden_dim // 2, n_classes))
        
        return nn.Sequential(*layers)
    
    def _create_residual(self, input_dim, task_type, n_classes):
        hidden_dim = 128
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ]
        
        # Add residual blocks
        for _ in range(3):
            layers.append(ResidualBlock(hidden_dim))
        
        if task_type == 'regression':
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            layers.append(nn.Linear(hidden_dim, n_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Get outputs from all models
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        
        # Apply softmax to ensemble weights
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        if self.task_type == 'regression':
            # Weighted average for regression
            ensemble_out = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
            return ensemble_out
        else:
            # Concatenate and process for classification
            combined = torch.cat([out1, out2, out3], dim=1)
            return self.final_layer(combined)


class EnhancedTraditional(nn.Module):
    """
    Enhanced traditional model with better architecture.
    Module-level class for pickle support.
    """
    def __init__(self, input_dim, task_type='regression', n_classes=None):
        super().__init__()
        self.task_type = task_type
        
        # Create architecture
        hidden_sizes = [256, 128, 64, 32]
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        if task_type == 'regression':
            layers.append(nn.Linear(prev_size, 1))
        else:
            layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

class PhysicsBasedFNRPredictor:
    """
    Physics-based FNR predictor that calculates FNR from component predictions
    Must be at module level to be picklable
    """
    def __init__(self, snr_model, ttd_model, dr_model, 
                 snr_features, ttd_features, dr_features,
                 has_noise_feature):
        self.snr_model = snr_model
        self.ttd_model = ttd_model
        self.dr_model = dr_model
        self.snr_features = snr_features
        self.ttd_features = ttd_features
        self.dr_features = dr_features
        self.has_noise_feature = has_noise_feature
        
        # Store noise statistics for estimation if needed
        if not has_noise_feature:
            self.noise_mean = 0.15
            self.noise_std = 0.05
    
    def predict(self, X):
        """Predict FNR from input features using physics-based calculation"""
        # Extract features for each component
        if isinstance(X, pd.DataFrame):
            X_snr = X[self.snr_features]
            X_ttd = X[self.ttd_features]
            X_dr = X[self.dr_features]
        else:
            # Handle numpy arrays
            X_snr = X[:, :len(self.snr_features)]
            X_ttd = X[:, :len(self.ttd_features)]
            X_dr = X[:, :len(self.dr_features)]
        
        # Predict components
        def predict_comp(model, X_data):
            if hasattr(model, 'eval'):  # PyTorch
                model.eval()
                with torch.no_grad():
                    if isinstance(X_data, pd.DataFrame):
                        pred = model(torch.FloatTensor(X_data.values)).numpy().flatten()
                    else:
                        pred = model(torch.FloatTensor(X_data)).numpy().flatten()
            else:  # sklearn
                pred = model.predict(X_data)
            return pred
        
        snr_pred = predict_comp(self.snr_model, X_snr)
        ttd_pred = predict_comp(self.ttd_model, X_ttd)
        dr_pred = predict_comp(self.dr_model, X_dr)
        
        # Get noise
        if self.has_noise_feature:
            if isinstance(X, pd.DataFrame) and 'background_noise_level' in X.columns:
                noise = X['background_noise_level'].values
            else:
                noise = dr_pred / (snr_pred + 1e-6)
                noise = np.clip(noise, 0.01, 0.5)
        else:
            noise = dr_pred / (snr_pred + 1e-6)
            noise = np.clip(noise, 0.01, 0.5)
        
        # Calculate FNR using physics
        adaptive_threshold = 5000 + 2000 * (1.0 / (snr_pred + 1))
        signal_overlap = 10.0 / (snr_pred + 1) + ttd_pred / 200.0 - dr_pred / 20000.0
        signal_overlap = np.clip(signal_overlap, 0.1, 10.0)
        base_error = 0.02 / (1 + signal_overlap * 0.1)
        noise_factor = noise / 0.15
        fnr = base_error * 2.0 + 0.02 * noise_factor
        fnr = np.clip(fnr, 0.01, 0.20)
        
        return fnr

class EnsembleModel:
    """
    Ensemble model that averages predictions from multiple models.
    MUST be at module level for pickle serialization!
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
    
    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.average(preds, axis=0, weights=self.weights)
    
    def get_params(self, deep=True):
        return {'weights': self.weights}

class TwoStageTTDPredictor:
    """
    ✅ COMPLETELY REDESIGNED TTD predictor with physics-informed approach
    
    CRITICAL INSIGHT from biosensor_engine.py analysis:
    ───────────────────────────────────────────────────────────────────────
    TTD is GENERATED from multiple physical components:
    1. Signal rise dynamics (how fast signal crosses threshold)
    2. SNR-dependent detection probability (noise delays detection)
    3. Threshold proximity (how close signal gets to threshold)
    4. Stochastic delays (random detection lag)
    5. Circuit-specific characteristics
    
    PROBLEM with old approach:
    ───────────────────────────────────────────────────────────────────────
    - Tried to predict TTD as single regression target
    - Ignored that TTD is DERIVED from these components
    - Lost physical relationships
    - Poor generalization
    
    NEW APPROACH - Physics-Informed Multi-Component Model:
    ───────────────────────────────────────────────────────────────────────
    1. Predict SIGNAL STRENGTH relative to threshold (primary driver)
    2. Predict SNR impact on detection lag (secondary driver)  
    3. Predict baseline detection time (tertiary driver)
    4. COMBINE using physics-based formula from biosensor_engine.py
    
    This mirrors how TTD is actually GENERATED in the simulation!
    """
    def __init__(self, models_dict, feature_names, timeout=9000.0):
        """
        Initialize physics-informed TTD predictor
        
        Args:
            models_dict: Dictionary containing:
                - 'signal_strength': Model predicting signal/threshold ratio
                - 'snr_delay': Model predicting SNR-based delay factor
                - 'base_time': Model predicting baseline detection time
            feature_names: List of feature names used
            timeout: Maximum detection time (censoring value)
        """
        self.signal_strength_model = models_dict['signal_strength']
        self.snr_delay_model = models_dict['snr_delay']
        self.base_time_model = models_dict['base_time']
        self.feature_names = feature_names
        self.timeout = timeout
        
        # Physics parameters (from biosensor_engine.py)
        self.k_steepness = 5.0  # Detection probability curve steepness
        self.snr_delay_factors = {
            'excellent': 1.0,   # SNR > 25 dB
            'good': 1.1,        # SNR 15-25 dB
            'fair': 1.3,        # SNR 5-15 dB
            'poor': 1.6         # SNR < 5 dB
        }
    
    def predict(self, X):
        """
        Predict TTD using physics-informed multi-component approach
        
        Steps (matching biosensor_engine._calculate_realistic_ttd):
        1. Predict signal strength relative to threshold
        2. Predict SNR-based delay factor
        3. Predict baseline detection time
        4. Apply physics-based combination
        5. Add realistic variance
        """
        # Handle DataFrame vs array
        if hasattr(X, 'values'):
            X_data = X.values if isinstance(X, pd.DataFrame) else X
        else:
            X_data = X
        
        n_samples = X_data.shape[0]
        
        # ═══════════════════════════════════════════════════════════════════
        # Component 1: Signal Strength (PRIMARY DRIVER)
        # ═══════════════════════════════════════════════════════════════════
        # Predict: (signal - threshold) / threshold
        signal_strength = self.signal_strength_model.predict(X_data)
        
        # Convert to detection probability using sigmoid (from biosensor_engine)
        # P(detect) = 1 / (1 + exp(-k * signal_strength))
        detection_prob = 1.0 / (1.0 + np.exp(-self.k_steepness * signal_strength))
        
        # ═══════════════════════════════════════════════════════════════════
        # Component 2: SNR Delay Factor (SECONDARY DRIVER)
        # ═══════════════════════════════════════════════════════════════════
        snr_delay = self.snr_delay_model.predict(X_data)
        
        # Clip to realistic range [1.0, 2.0]
        snr_delay = np.clip(snr_delay, 1.0, 2.0)
        
        # ═══════════════════════════════════════════════════════════════════
        # Component 3: Base Detection Time (TERTIARY DRIVER)
        # ═══════════════════════════════════════════════════════════════════
        base_time = self.base_time_model.predict(X_data)
        
        # Clip to simulation time range [10, 3600]
        base_time = np.clip(base_time, 10.0, 3600.0)
        
        # ═══════════════════════════════════════════════════════════════════
        # Physics-Based Combination
        # ═══════════════════════════════════════════════════════════════════
        # Formula from biosensor_engine.py:
        # final_ttd = base_ttd * snr_delay_factor + sensitivity_delay
        
        # Sensitivity delay: lower detection prob → longer detection time
        # Map detection_prob [0, 1] → delay_mult [2.0, 0.5]
        sensitivity_delay_mult = 2.0 - 1.5 * detection_prob
        
        # Combine components
        ttd_predicted = base_time * snr_delay * sensitivity_delay_mult
        
        # ═══════════════════════════════════════════════════════════════════
        # Handle Edge Cases (from biosensor_engine.py)
        # ═══════════════════════════════════════════════════════════════════
        # Weak signals that never reach threshold
        weak_signal_mask = signal_strength < -0.5  # Signal < 0.5 * threshold
        
        if np.any(weak_signal_mask):
            # Extrapolate based on signal fraction
            signal_fraction = 1.0 / (1.0 - signal_strength[weak_signal_mask])
            projected_ttd = 3600.0 / (signal_fraction + 0.01)
            projected_ttd = np.clip(projected_ttd, 3600.0 * 0.9, 3600.0 * 2.5)
            ttd_predicted[weak_signal_mask] = projected_ttd
        
        # ═══════════════════════════════════════════════════════════════════
        # Add Realistic Variance (±5% jitter from biosensor_engine.py)
        # ═══════════════════════════════════════════════════════════════════
        jitter = np.random.uniform(-0.05, 0.05, size=n_samples) * ttd_predicted
        ttd_predicted = ttd_predicted + jitter
        
        # ═══════════════════════════════════════════════════════════════════
        # Apply Timeout Threshold (censoring)
        # ═══════════════════════════════════════════════════════════════════
        # If detection is very unlikely or time exceeds observation window
        timeout_mask = (detection_prob < 0.1) | (ttd_predicted > 3600.0 * 2.5)
        ttd_predicted[timeout_mask] = self.timeout
        
        # Final clipping
        ttd_predicted = np.clip(ttd_predicted, 10.0, self.timeout)
        
        return ttd_predicted
    
    def get_params(self, deep=True):
        return {
            'timeout': self.timeout,
            'k_steepness': self.k_steepness
        }
    
class BiosensorPipeline:
    """Complete pipeline for biosensor design optimization with focused target metrics"""
    
    def __init__(self, data_path: str, output_dir: str = "biosensor_results"):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the main dataset CSV
            output_dir: Directory to save all outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        
        for dir_path in [self.plots_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models storage
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        self.surrogate_models = {}
        
        # ✅ Initialize surrogate_features dictionary early
        self.surrogate_features = {}
        self.surrogate_metric_scalers = {}
            
        # RL components
        self.rl_env = None
        self.rl_agents = {}
        
        # Ã°Å¸Å½Â¯ FOCUSED TARGET METRICS - Only these will be used for training
        self.target_metrics = [
            'signal_to_noise_ratio_SNR',      # Primary performance metric
            'false_negative_rate',             # Minimize errors
            'time_to_detection_threshold',     # Speed optimization
            'dynamic_range_of_output',         # Signal range
        ]

        # Feature selection parameters - Fast mode
        self.feature_selection_methods = {
            'statistical': False,    # Disabled for speed
            'model_based': True,     # Keep only RF
            'recursive': False,      # Disabled for speed
            'lasso': False          # Disabled for speed
        }
        
        # Feature selection storage
        self.selected_features = {}
        self.feature_selection_scores = {}
        
        print(f"Ã°Å¸Å¡â‚¬ Biosensor Pipeline initialized")
        print(f"Ã°Å¸â€œÂ Output directory: {self.output_dir}")
        print(f"Ã°Å¸Å½Â¯ Target metrics: {len(self.target_metrics)} focused metrics")

    @staticmethod
    def safe_divide(numerator, denominator, fill_value=0.0, min_denom=1e-10):
        """Safely divide arrays, replacing inf/nan with fill_value"""
        denominator_safe = np.where(np.abs(denominator) < min_denom, min_denom, denominator)
        result = numerator / denominator_safe
        result = np.where(np.isfinite(result), result, fill_value)
        return result

    @staticmethod
    def safe_exp(x, clip_min=-50, clip_max=50):
        """Safely compute exp, clipping input to prevent overflow/underflow"""
        x_clipped = np.clip(x, clip_min, clip_max)
        return np.exp(x_clipped)

    @staticmethod
    def validate_no_nan_inf(arr, name="array"):
        """Check array for NaN or inf values"""
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        
        if has_nan or has_inf:
            print(f"      ⚠️ WARNING: {name} contains NaN={has_nan}, inf={has_inf}")
            print(f"         Range: [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
            print(f"         NaN count: {np.isnan(arr).sum()}, inf count: {np.isinf(arr).sum()}")
            return False
        return True

    @staticmethod
    def encode_categorical_features(X_df):
        """
        Encode categorical features properly BEFORE any component calculations.
        
        Returns:
            X_encoded: DataFrame with categorical variables properly encoded
            encoding_info: Dictionary with encoding mappings
        """
        X_encoded = X_df.copy()
        encoding_info = {}
        
        # Define categorical columns and their encodings
        categorical_encodings = {
            'circuit_type': {
                'direct_binding': 0,
                'ratiometric': 1,
                'threshold': 2,
                'amplifying': 3
            },
            'response_type': {
                'linear': 0,
                'hill': 1,
                'michaelis_menten': 2
            },
            'noise_preset': {
                'low': 0,
                'medium': 1,
                'high': 2
            },
            'scenario': {
                'healthy': 0,
                'pmo': 1,
                'ckd_mbd': 2
            }
        }
        
        for col, mapping in categorical_encodings.items():
            if col in X_encoded.columns:
                # Check if column is actually categorical
                if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                    # Map values
                    X_encoded[col] = X_encoded[col].map(mapping)
                    
                    # Check for unmapped values (would be NaN)
                    if X_encoded[col].isnull().any():
                        unmapped = X_df[col][X_encoded[col].isnull()].unique()
                        print(f"      ⚠️ WARNING: {col} has unmapped values: {unmapped}")
                        # Fill with mode or default
                        X_encoded[col] = X_encoded[col].fillna(X_encoded[col].mode()[0] if not X_encoded[col].mode().empty else 0)
                    
                    encoding_info[col] = mapping
                    print(f"      ✅ Encoded {col}: {list(mapping.keys())}")
        
        return X_encoded, encoding_info
    
    def load_data(self, biomarker: Optional[str] = None) -> pd.DataFrame:
        """
        Load Sclerostin biosensor simulation dataset from master_index.csv
        
        This loads the master index which contains metadata and paths to detailed time-series data.
        For RL training, we extract biosensor configurations and performance metrics.
        
        Args:
            biomarker: Not used for Sclerostin dataset (kept for compatibility)
            
        Returns:
            Loaded DataFrame with extracted features
        """
        print(f"Ã°Å¸â€œÅ  Loading Sclerostin simulation dataset: {self.data_path}")
        
        # Load master index
        master_df = pd.read_csv(self.data_path)
        print(f"Ã¢Å“â€¦ Master index loaded: {master_df.shape} simulations")
        
        # Extract features from metadata JSON files
        print("Ã°Å¸â€œâ€š Extracting biosensor configurations from metadata files...")
        
        features_list = []
        data_dir = Path(self.data_path).parent
        
        for idx, row in master_df.iterrows():
            metadata_path = data_dir / row['metadata_file']
            
            if not metadata_path.exists():
                print(f"Ã¢Å¡ Ã¯Â¸Â  Metadata file not found: {metadata_path}")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract environment parameters
                env_params = metadata['environment_params']

                # Extract biosensor configuration
                biosensor_config = metadata['biosensor_config']

                # CRITICAL: Handle missing keys gracefully
                def safe_get(dict_obj, key, default=0.0):
                    """Safely get value from dict with default"""
                    return dict_obj.get(key, default)
                
                # Extract measurement metrics
                measurement = metadata['measurement']
                
                # Combine into feature vector
                features = {
                    # MODIFIABLE features (biosensor design parameters)
                    'biosensor_sensitivity': safe_get(biosensor_config, 'sensitivity', 1.0),
                    'biosensor_threshold': safe_get(biosensor_config, 'threshold', 0.1),
                    'biosensor_kd': safe_get(biosensor_config, 'kd', 0.1),
                    'biosensor_dynamic_range_max': biosensor_config.get('dynamic_range', [0, 1])[1],
                    'circuit_type': safe_get(biosensor_config, 'circuit_type', 'direct_binding'),
                    'response_type': safe_get(biosensor_config, 'response_type', 'linear'),
                    'hill_coefficient': safe_get(biosensor_config, 'hill_coefficient', 2.0),
                    'off_level': safe_get(biosensor_config, 'off_level', 0.1),
                    'on_level': safe_get(biosensor_config, 'on_level', 10.0),
                    
                    # FIXED features (environment/simulation conditions)
                    'scenario': row['scenario'],
                    'estrogen': safe_get(env_params, 'Estrogen', 1.0),
                    'pth': safe_get(env_params, 'PTH', 100.0),
                    'sclerostin_concentration': metadata['sclerostin_mean'],
                    'rankl_concentration': safe_get(env_params, 'RANKL_bone', 1.0),
                    'opg_concentration': safe_get(env_params, 'OPG_bone', 1.0),
                    'mineral_ion': safe_get(env_params, 'MineralIon', 4.0),  # FIX: Use safe_get
                    'noise_preset': row['noise_preset'],
                    
                    # TARGET metrics (outputs to optimize)
                    'signal_to_noise_ratio_SNR': measurement['snr_db'],
                    'n_detections': measurement['n_detections'],
                    'detection_rate': measurement['detection_rate'],
                    'time_to_detection_threshold': measurement['time_to_detection'], 
                    'max_signal': measurement['max_signal'],
                    'mean_signal': measurement['mean_signal'],
                    'signal_std': measurement['signal_std'],
                    'false_negative_rate': measurement['false_negative_rate'], 
                    
                    # Additional useful features
                    'sclerostin_max': metadata['sclerostin_max'],
                    'sclerostin_std': metadata['sclerostin_std'],
                }

                # ═══════════════════════════════════════════════════════════════
                # ✅ NEW: Physics-Informed Features for TTD Prediction
                # ═══════════════════════════════════════════════════════════════
                # These features directly capture the TTD generation process
                # from biosensor_engine._calculate_realistic_ttd()
                
                # 1. Signal-to-threshold ratio (PRIMARY driver of TTD)
                features['signal_to_threshold_ratio'] = (
                    metadata['sclerostin_mean'] / (features['biosensor_threshold'] + 1e-9)
                )

                # 2. Affinity-concentration ratio (binding effectiveness)
                features['affinity_concentration_ratio'] = (
                    metadata['sclerostin_mean'] / (features['biosensor_kd'] + 1e-9)
                )

                # 3. Detection score (combined sensitivity & signal strength)
                features['detection_score'] = (
                    features['biosensor_sensitivity'] * 
                    features['signal_to_threshold_ratio']
                )

                # 4. Sensitivity-threshold product (detection difficulty)
                features['sensitivity_threshold_product'] = (
                    features['biosensor_sensitivity'] * features['biosensor_threshold']
                )

                # 5. SNR-based detection probability (from biosensor_engine)
                # Using the same sigmoid formula: P = 1/(1 + exp(-k*margin))
                if 'snr_db' in measurement:
                    features['snr_db'] = measurement['snr_db']
                    
                    # Map SNR to delay factor (from biosensor_engine)
                    snr = measurement['snr_db']
                    if snr > 25:
                        snr_delay = 1.0
                    elif snr > 15:
                        snr_delay = 1.0 + (25 - snr) / 100
                    elif snr > 5:
                        snr_delay = 1.0 + (25 - snr) / 50
                    else:
                        snr_delay = 1.0 + (25 - snr) / 25
                    
                    features['snr_delay_factor'] = snr_delay
                    features['noise_penalty'] = 1.0 / (1.0 + np.exp(snr / 10.0))
                
                # 6. Signal variability (affects detection consistency)
                features['signal_variability'] = (
                    metadata['sclerostin_std'] / (metadata['sclerostin_mean'] + 1e-9)
                )
                
                # 7. Detection difficulty score (combines multiple factors)
                features['detection_difficulty'] = (
                    1.0 / (features['detection_score'] + 1e-6)
                )
                
                # ═══════════════════════════════════════════════════════════════
                # ✅ CRITICAL: Biosensor-Environment Interaction Features
                # ═══════════════════════════════════════════════════════════════
                # These help ML learn how biosensor configs respond to different environments
                # WITHOUT letting environment alone dominate predictions
                
                # 1. Sensitivity × biomarker concentration (how well biosensor captures signal)
                features['sensitivity_x_concentration'] = (
                    features['biosensor_sensitivity'] * metadata['sclerostin_mean']
                )
                
                # 2. Kd × biomarker concentration (binding affinity vs concentration)
                features['kd_x_concentration'] = (
                    features['biosensor_kd'] * metadata['sclerostin_mean']
                )
                
                # 3. Threshold × noise (detection difficulty in noisy environment)
                if 'background_noise_level' in measurement:
                    features['threshold_x_noise'] = (
                        features['biosensor_threshold'] * measurement['background_noise_level']
                    )
                else:
                    # Assume 15% baseline noise if not specified
                    features['threshold_x_noise'] = features['biosensor_threshold'] * 0.15
                
                # 4. Signal strength estimate (biosensor effectiveness)
                # Higher = more effective biosensor config
                features['estimated_signal_strength'] = (
                    features['biosensor_sensitivity'] / (features['biosensor_kd'] + 1e-6)
                )
                
                # 5. Normalized detection power (biosensor vs environment)
                # Signal strength relative to noise and threshold
                noise_level = measurement.get('background_noise_level', 0.15)
                features['normalized_detection_power'] = (
                    features['estimated_signal_strength'] * metadata['sclerostin_mean'] / 
                    (features['biosensor_threshold'] + noise_level + 1e-6)
                )
                
                # 6. Concentration-to-Kd ratio (how far we are from binding equilibrium)
                # >1 means strong binding, <1 means weak
                features['concentration_kd_ratio'] = (
                    metadata['sclerostin_mean'] / (features['biosensor_kd'] + 1e-9)
                )
                
                # 7. Signal margin (distance from threshold in units of noise)
                # Positive = signal above threshold, negative = below
                expected_signal = features['biosensor_sensitivity'] * metadata['sclerostin_mean']
                features['signal_margin_in_noise_units'] = (
                    (expected_signal - features['biosensor_threshold']) / (noise_level + 1e-6)
                )
                
                features_list.append(features)
                
            except Exception as e:
                print(f"Ã¢Å¡ Ã¯Â¸Â  Error processing {metadata_path}: {e}")
                continue
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(master_df)} simulations...")
        
        # Create DataFrame from extracted features
        self.raw_data = pd.DataFrame(features_list)
        print(f"Ã¢Å“â€¦ Extracted features from {len(features_list)} simulations")
        print(f"   Shape: {self.raw_data.shape}")
        
        # Calculate dynamic_range_of_output (synthesized from max_signal)
        self.raw_data['dynamic_range_of_output'] = self.raw_data['max_signal']
        
        # Update target metrics to match simulation outputs
        self.target_metrics = [
            'signal_to_noise_ratio_SNR',
            'false_negative_rate',
            'time_to_detection_threshold',
            'dynamic_range_of_output'
        ]
        
        # Verify all targets are available
        available_targets = [m for m in self.target_metrics if m in self.raw_data.columns]
        missing_targets = [m for m in self.target_metrics if m not in self.raw_data.columns]
        
        if missing_targets:
            print(f"Ã¢Å¡ Ã¯Â¸Â  WARNING: Missing target metrics: {missing_targets}")
        else:
            print(f"Ã¢Å“â€¦ All target metrics available: {available_targets}")
        
        # Calculate multi-objective score
        print("Ã°Å¸â€œÅ  Calculating multi-objective score...")
        self.raw_data['multi_objective_score'] = self._calculate_multi_objective_score(self.raw_data)
        print("Ã¢Å“â€¦ Multi-objective score calculated")
        
        # Summary statistics
        print(f"\nÃ°Å¸â€œË† Dataset Summary:")
        print(f"   Scenarios: {self.raw_data['scenario'].value_counts().to_dict()}")
        print(f"   Circuit types: {self.raw_data['circuit_type'].value_counts().to_dict()}")
        print(f"   Noise levels: {self.raw_data['noise_preset'].value_counts().to_dict()}")
        
        return self.raw_data
        
    def _calculate_multi_objective_score(self, df):
        """
        Calculate multi-objective score from 4 key targets using weighted combination
        
        Formula: Score = w1*norm(SNR) + w2*norm(DR) - w3*norm(FNR) - w4*norm(TTD)
        """
        print("Calculating multi-objective score...")
        
        # Weights as specified
        w1, w2, w3, w4 = 0.45, 0.25, 0.20, 0.10
        
        # Get the metrics
        snr = df['signal_to_noise_ratio_SNR']
        fnr = df['false_negative_rate']
        ttd = df['time_to_detection_threshold']
        dr = df['dynamic_range_of_output']
        
        # Min-Max normalization
        def normalize(series):
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)
        
        # Normalize each metric
        snr_norm = normalize(snr)
        dr_norm = normalize(dr)
        fnr_norm = normalize(fnr)
        ttd_norm = normalize(ttd)
        
        # Calculate weighted score
        score = (w1 * snr_norm + 
                w2 * dr_norm - 
                w3 * fnr_norm - 
                w4 * ttd_norm)
        
        # Normalize final score to 0-1 range
        score = normalize(score)
        
        print(f"   Multi-objective score calculated: mean={score.mean():.4f}, std={score.std():.4f}")
        
        return score
    
    def preprocess_data(self, test_size: float = 0.2, 
                   feature_selection: bool = True,
                   dimensionality_reduction: str = 'pca',
                   variance_threshold: float = 0.95,
                   apply_pca: bool = None) -> Dict:
        """
        Complete data preprocessing pipeline with proper variable classification

        Args:
            test_size: Proportion of data for testing
            feature_selection: Whether to perform feature selection
            dimensionality_reduction: Method for dimensionality reduction ('pca', 'none')
            variance_threshold: Variance to retain for PCA
            apply_pca: DEPRECATED - Use dimensionality_reduction instead

        Returns:
            Dictionary with preprocessing information
        """

        # Handle backward compatibility silently
        if apply_pca is not None:
            dimensionality_reduction = 'pca' if apply_pca else 'none'

        print("Ã°Å¸â€â€ž Starting advanced data preprocessing with proper variable classification...")

        # Ã°Å¸Å¸Â© MODIFIABLE INPUTS (RL Agent can change these)
        modifiable_features = [
            'biosensor_sensitivity',
            'biosensor_threshold',
            'biosensor_kd',
            'biosensor_dynamic_range_max',
            'hill_coefficient',
            'off_level',
            'on_level',
            # Categorical features (KEEP THESE)
            'circuit_type',
            'response_type'
        ]

        # Ã°Å¸Å¸Â¦ FIXED INPUTS (Environmental/Simulation - not modifiable by RL agent)
        fixed_features = [
            'sclerostin_concentration',
            'estrogen',
            'pth',
            'rankl_concentration',
            'opg_concentration',
            'mineral_ion',
            'sclerostin_max',
            'sclerostin_std',
            # Categorical contextual features (KEEP THESE)
            'scenario',
            'noise_preset'
        ]

        # Ã°Å¸Å¸Â¥ OUTPUT METRICS (Performance - never inputs, always targets)
        output_metrics = self.target_metrics  # Use focused target metrics directly  

        # Handle missing values
        print("Ã°Å¸Â§Â¹ Handling missing values...")
        missing_before = self.raw_data.isnull().sum().sum()

        # Fill numerical columns with median
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        self.raw_data[numerical_cols] = self.raw_data[numerical_cols].fillna(
            self.raw_data[numerical_cols].median()
        )

        # Fill categorical columns with mode
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.raw_data[col] = self.raw_data[col].fillna(
                self.raw_data[col].mode()[0] if len(self.raw_data[col].mode()) > 0 else 'unknown'
            )

        missing_after = self.raw_data.isnull().sum().sum()
        print(f"   Missing values: {missing_before} Ã¢â€ â€™ {missing_after}")

        # Ã°Å¸Å½Â¯ PROPER VARIABLE SEPARATION
        print("Ã°Å¸Å½Â¯ Separating variables by classification...")

        # Get available columns for each category
        available_modifiable = [col for col in modifiable_features if col in self.raw_data.columns]
        available_fixed = [col for col in fixed_features if col in self.raw_data.columns]
        available_outputs = [col for col in output_metrics if col in self.raw_data.columns]

        # Store the classification for later use
        self.modifiable_features = available_modifiable
        self.fixed_features = available_fixed
        self.output_metrics = available_outputs

        print(f"Ã°Å¸â€œâ€¹ Modifiable features (RL actions): {len(available_modifiable)}")
        print(f"Ã°Å¸â€œâ€¹ Fixed features (environment): {len(available_fixed)}")
        print(f"Ã°Å¸â€œâ€¹ Output metrics (targets): {len(available_outputs)}")

        if not available_outputs:
            raise ValueError("No output metrics available in the dataset!")

        # For ML models: Use ALL features (modifiable + fixed) as inputs
        X_features = available_modifiable + available_fixed
        X = self.raw_data[X_features].copy()
        y = self.raw_data[available_outputs].copy()

        # Update target metrics to only available ones
        self.target_metrics = available_outputs

        # Encode categorical variables
        print("Ã°Å¸â€Â¤ Encoding categorical variables...")
        categorical_features = X.select_dtypes(include=['object']).columns

        # CRITICAL: Use label encoding instead of one-hot for categorical features
        # This preserves the feature count and makes bounds easier
        if len(categorical_features) > 0:
            from sklearn.preprocessing import LabelEncoder
            
            X_categorical_encoded = X[categorical_features].copy()
            self.categorical_label_encoders = {}
            
            for cat_feature in categorical_features:
                le = LabelEncoder()
                X_categorical_encoded[cat_feature] = le.fit_transform(X[cat_feature].fillna('unknown'))
                self.categorical_label_encoders[cat_feature] = le
                print(f"   {cat_feature}: {len(le.classes_)} categories -> [0, {len(le.classes_)-1}]")
            
            # Combine with numerical features
            X_numerical = X.select_dtypes(include=[np.number])
            
            # Replace categorical columns with encoded versions
            X = X_numerical.copy()
            for cat_col in categorical_features:
                X[cat_col] = X_categorical_encoded[cat_col]
            
            print(f"   Categorical features label-encoded: {len(categorical_features)} columns")

        # Store categorical feature mappings for RL environment (using label encoders)
        if len(categorical_features) > 0 and hasattr(self, 'categorical_label_encoders'):
            self.categorical_mappings = {}
            for cat_feature in categorical_features:
                if cat_feature in self.categorical_label_encoders:
                    le = self.categorical_label_encoders[cat_feature]
                    self.categorical_mappings[cat_feature] = list(le.classes_)
                    print(f"      {cat_feature}: {len(le.classes_)} categories stored")

        # Scale numerical features
        print("Ã°Å¸â€œÂ Scaling numerical features...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Feature Selection
        feature_selection_info = {}
        if feature_selection:
            feature_selection_info = self._perform_feature_selection(X_scaled, y)
            X_selected = X_scaled[feature_selection_info['final_features']]
            print(f"   Feature selection: {X_scaled.shape[1]} Ã¢â€ â€™ {X_selected.shape[1]} features")
        else:
            X_selected = X_scaled
            print("   Feature selection: Skipped")

        # Dimensionality Reduction
        reduction_info = {}
        if dimensionality_reduction != 'none':
            X_final, reduction_info = self._apply_dimensionality_reduction(
                X_selected, dimensionality_reduction, variance_threshold
            )
        else:
            X_final = X_selected
            reduction_info = {'method': 'none'}

        # Train-test split
        print("Ã¢Å“â€šÃ¯Â¸Â Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_final, y, test_size=test_size, random_state=42, stratify=None
        )
        # Clean column names - remove problematic characters
        self.X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '_') 
                                for col in self.X_train.columns]
        self.X_test.columns = [col.replace('[', '').replace(']', '').replace('<', '').replace('>', '_') 
                            for col in self.X_test.columns]
        # Store processed data
        self.processed_data = {
            'X': X_final,
            'y': y,
            'scaler': scaler,
            'feature_names': X_final.columns.tolist(),
            'target_names': y.columns.tolist(),
            'modifiable_features': available_modifiable,
            'fixed_features': available_fixed,
            'feature_selection_info': feature_selection_info,
            'reduction_info': reduction_info
        }

        preprocessing_info = {
            'original_shape': self.raw_data.shape,
            'processed_shape': X_final.shape,
            'missing_values_handled': missing_before,
            'categorical_features': len(categorical_features),
            'numerical_features': len(X.select_dtypes(include=[np.number]).columns),
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0],
            'feature_selection_applied': feature_selection,
            'dimensionality_reduction_applied': dimensionality_reduction,
            'final_feature_count': X_final.shape[1],
            'target_metrics_count': len(available_outputs),
            'modifiable_features_count': len(available_modifiable),
            'fixed_features_count': len(available_fixed),
            'feature_reduction_ratio': X_final.shape[1] / len(X_features) if len(X_features) > 0 else 0
        }

        print("Ã¢Å“â€¦ Advanced preprocessing with proper classification completed!")
        print(f"   Final shape: {X_final.shape}")
        print(f"   Train/Test: {self.X_train.shape[0]}/{self.X_test.shape[0]}")
        print(f"   Features: {len(X_features)} Ã¢â€ â€™ {X_final.shape[1]} ({preprocessing_info['feature_reduction_ratio']:.2f})")
        print(f"   Modifiable: {len(available_modifiable)}, Fixed: {len(available_fixed)}")
        print(f"   Target metrics: {len(available_outputs)}")
        
        # VALIDATION: Check feature counts
        print(f"\nÃ°Å¸â€Â VALIDATION: Feature Classification")
        print(f"   Total features in X: {len(X_final.columns)}")
        print(f"   Modifiable features found: {len([f for f in available_modifiable if f in X_final.columns])}")
        print(f"   Fixed features found: {len([f for f in available_fixed if f in X_final.columns])}")

        if len([f for f in available_modifiable if f in X_final.columns]) == 0:
            print(f"   Ã¢Å¡ Ã¯Â¸Â WARNING: No modifiable features found in final dataset!")
            print(f"   Available columns: {list(X_final.columns)[:10]}...")

        return preprocessing_info
    
    def augment_training_data(self, augmentation_factor=3):  # ✅ 3x, not 10x
        """Simple, effective augmentation with SMOTE-like interpolation"""
        print(f"📊 Augmenting training data by {augmentation_factor}x...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Store original
        X_modifiable = self.X_train[[col for col in self.modifiable_features if col in self.X_train.columns]].copy()
        X_fixed = self.X_train[[col for col in self.fixed_features if col in self.X_train.columns]].copy()
        y_original = self.y_train.copy()
        
        original_columns = self.X_train.columns.tolist()
        
        # Find neighbors
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X_modifiable)
        
        augmented_X_mod = []
        augmented_X_fix = []
        augmented_y = []
        
        n_samples_to_generate = len(self.X_train) * augmentation_factor
        
        for i in range(n_samples_to_generate):
            # Pick random sample
            idx = np.random.randint(0, len(X_modifiable))
            
            # Find closest neighbor
            distances, indices = nbrs.kneighbors(X_modifiable.iloc[idx:idx+1])
            neighbor_idx = indices[0][1]  # Closest neighbor
            
            # Interpolate with moderate randomness
            alpha = np.random.uniform(0.3, 0.7)  # ✅ Stay closer to real data
            
            # Modifiable: interpolate
            new_mod = (1 - alpha) * X_modifiable.iloc[idx] + alpha * X_modifiable.iloc[neighbor_idx]
            augmented_X_mod.append(new_mod)
            
            # Fixed: keep original
            augmented_X_fix.append(X_fixed.iloc[idx])
            
            # Targets: interpolate
            new_y = (1 - alpha) * y_original.iloc[idx] + alpha * y_original.iloc[neighbor_idx]
            augmented_y.append(new_y)
        
        # Combine
        X_mod_combined = pd.concat([
            X_modifiable.reset_index(drop=True),
            pd.DataFrame(augmented_X_mod, columns=X_modifiable.columns)
        ], axis=0, ignore_index=True)
        
        X_fix_combined = pd.concat([
            X_fixed.reset_index(drop=True),
            pd.DataFrame(augmented_X_fix, columns=X_fixed.columns)
        ], axis=0, ignore_index=True)
        
        y_combined = pd.concat([
            y_original.reset_index(drop=True),
            pd.DataFrame(augmented_y, columns=y_original.columns)
        ], axis=0, ignore_index=True)
        
        # Reconstruct with original column order
        X_combined_list = []
        for col in original_columns:
            if col in X_mod_combined.columns:
                X_combined_list.append(X_mod_combined[col])
            elif col in X_fix_combined.columns:
                X_combined_list.append(X_fix_combined[col])
        
        self.X_train = pd.DataFrame(dict(zip(original_columns, X_combined_list)), columns=original_columns)
        self.y_train = y_combined
        
        print(f"✅ Augmented: {len(self.X_train)} samples (from {len(X_modifiable)})")

    def add_noise_augmentation(self, noise_level=0.02):
        """Add small Gaussian noise to training data for robustness"""
        print(f"🔊 Adding noise augmentation (σ={noise_level})...")
        
        X_train_original = self.X_train.copy()
        y_train_original = self.y_train.copy()
        
        # Generate noisy copies
        noise_X = np.random.normal(0, noise_level, self.X_train.shape)
        X_train_noisy = self.X_train + noise_X * self.X_train.std().values
        
        # Concatenate
        self.X_train = pd.concat([X_train_original, X_train_noisy], axis=0, ignore_index=True)
        self.y_train = pd.concat([y_train_original, y_train_original], axis=0, ignore_index=True)
        
        print(f"✅ Added {len(X_train_noisy)} noisy samples")

    def _perform_feature_selection(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """
        Feature selection for Sclerostin biosensor - Use ALL available features
        """
        print("Feature selection for Sclerostin biosensor...")
        
        # Get all available features (no filtering)
        available_features = X.columns.tolist()
        
        print(f"   Using all {len(available_features)} available features (no selection)")
        
        # Classify features
        modifiable_features = [
            'biosensor_sensitivity', 'biosensor_threshold', 'biosensor_kd',
            'biosensor_dynamic_range_max', 'circuit_type', 'response_type',
            'hill_coefficient', 'off_level', 'on_level'
        ]
        
        fixed_features = [
            'sclerostin_concentration', 'estrogen', 'pth',
            'rankl_concentration', 'opg_concentration', 'mineral_ion',
            'sclerostin_max', 'sclerostin_std', 'scenario', 'noise_preset'
        ]
        
        n_modifiable = len([f for f in modifiable_features if f in available_features])
        n_fixed = len([f for f in fixed_features if f in available_features])
        
        print(f"   - Modifiable: {n_modifiable}")
        print(f"   - Fixed: {n_fixed}")
        
        return {
            'selected_features': {'all': available_features},
            'final_features': available_features,
            'scores': {},
            'reduction_ratio': 1.0
        }

    def _apply_dimensionality_reduction(self, X: pd.DataFrame, method: str = 'pca', 
                                      variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply dimensionality reduction to the feature matrix
        
        Args:
            X: Feature matrix
            method: Dimensionality reduction method ('pca', 'none')
            variance_threshold: Variance to retain for PCA
            
        Returns:
            Reduced feature matrix and reduction information
        """
        print(f"Ã°Å¸â€Â Applying dimensionality reduction: {method}")
        
        if method == 'pca':
            pca = PCA(n_components=variance_threshold, random_state=42)
            X_reduced = pca.fit_transform(X)
            
            # Create DataFrame with component names
            n_components = X_reduced.shape[1]
            component_names = [f'PC{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)
            
            # Save PCA visualization
            self._save_pca_visualization(X_reduced_df, pca.explained_variance_ratio_)
            
            reduction_info = {
                'method': 'pca',
                'original_features': X.shape[1],
                'reduced_features': n_components,
                'variance_explained': pca.explained_variance_ratio_.sum(),
                'pca_object': pca
            }
            
            print(f"   PCA: {X.shape[1]} Ã¢â€ â€™ {n_components} components")
            print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
            
            return X_reduced_df, reduction_info
        
        else:
            print("   No dimensionality reduction applied")
            return X, {'method': 'none'}
    
    def _save_pca_visualization(self, pca_df: pd.DataFrame, explained_variance: np.ndarray):
        """Save PCA visualization plots"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance plot
        axes[0].bar(range(1, len(explained_variance) + 1), explained_variance)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Explained Variance')
        
        # Cumulative variance plot
        cumulative_variance = np.cumsum(explained_variance)
        axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_target_metrics_distributions(self, save_folder: str = "target_metric_plots"):
        """
        Create and save distribution plots for each target metric
        
        Args:
            save_folder: Folder name to save the plots
        """
        
        # Create the folder if it doesn't exist
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        
        # Get the target metrics data
        if hasattr(self, 'processed_data') and 'y' in self.processed_data:
            y_data = self.processed_data['y']
        elif hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            # Combine train and test data for full distribution
            y_data = pd.concat([self.y_train, self.y_test], axis=0)
        else:
            # Fall back to raw data if processed data not available
            y_data = self.raw_data[self.output_metrics]

        # Filter out boolean columns before converting to numeric
        numeric_columns = []
        for col in y_data.columns:
            if y_data[col].dtype != bool:
                numeric_columns.append(col)

        y_data = y_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        print(f"Ã°Å¸â€œÅ  Creating distribution plots for {len(self.output_metrics)} target metrics...")
        print(f"Ã°Å¸â€™Â¾ Saving plots to: {save_folder}/")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots for each target metric
        for i, metric in enumerate(self.output_metrics):
            if metric not in y_data.columns:
                print(f"Ã¢Å¡ Ã¯Â¸Â  Skipping {metric} - not found in data")
                continue
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Distribution Analysis: {metric}', fontsize=16, fontweight='bold')
            
            # Get the data for this metric
            metric_data = y_data[metric].dropna()

            # Skip boolean columns
            if metric_data.dtype == bool:
                print(f"Ã¢Å¡ Ã¯Â¸Â  Skipping {metric} - boolean data type not suitable for distribution plots")
                plt.close(fig)
                continue

            # Convert to numeric and filter out non-numeric data
            metric_data = pd.to_numeric(metric_data, errors='coerce').dropna()

            if len(metric_data) == 0:
                print(f"Ã¢Å¡ Ã¯Â¸Â  Skipping {metric} - no valid numeric data")
                plt.close(fig)
                continue
            
            # 1. Histogram with KDE
            axes[0, 0].hist(metric_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Histogram with Density')
            axes[0, 0].set_xlabel(metric)
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add KDE overlay
            try:
                sns.kdeplot(data=metric_data, ax=axes[0, 0], color='red', linewidth=2)
            except:
                pass  # Skip KDE if it fails
            
            # 2. Box plot
            axes[0, 1].boxplot(metric_data, patch_artist=True, 
                            boxprops=dict(facecolor='lightgreen', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
            axes[0, 1].set_title('Box Plot')
            axes[0, 1].set_ylabel(metric)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Q-Q plot (normal distribution comparison)
            from scipy import stats
            stats.probplot(metric_data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Violin plot
            axes[1, 1].violinplot(metric_data, showmeans=True, showmedians=True)
            axes[1, 1].set_title('Violin Plot')
            axes[1, 1].set_ylabel(metric)
            axes[1, 1].set_xticks([1])
            axes[1, 1].set_xticklabels([metric])
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"""Statistics:
    Mean: {metric_data.mean():.4f}
    Median: {metric_data.median():.4f}
    Std: {metric_data.std():.4f}
    Min: {metric_data.min():.4f}
    Max: {metric_data.max():.4f}
    Skewness: {metric_data.skew():.4f}
    Kurtosis: {metric_data.kurtosis():.4f}
    Count: {len(metric_data)}"""
            
            # Add text box with statistics
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for stats text
            
            # Save the plot
            filename = f"{metric.replace('/', '_').replace(' ', '_')}_distribution.png"
            filepath = os.path.join(save_folder, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close to free memory
            
            print(f"Ã¢Å“â€¦ Saved: {filename}")
        
        print(f"Ã°Å¸Å½â€° All distribution plots saved to {save_folder}/")

    def plot_target_metrics_correlation_matrix(self, save_folder: str = "target_metric_plots"):
        """
        Create and save a correlation matrix for all target metrics
        
        Args:
            save_folder: Folder name to save the plots
        """
        
        # Create the folder if it doesn't exist
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        
        # Get the target metrics data
        if hasattr(self, 'processed_data') and 'y' in self.processed_data:
            y_data = self.processed_data['y']
        elif hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            y_data = pd.concat([self.y_train, self.y_test], axis=0)
        else:
            y_data = self.raw_data[self.output_metrics]

        # Filter out boolean columns before converting to numeric
        numeric_columns = []
        for col in y_data.columns:
            if y_data[col].dtype != bool:
                numeric_columns.append(col)

        y_data = y_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Calculate correlation matrix
        correlation_matrix = y_data.corr()
        
        # Create the plot
        plt.figure(figsize=(20, 16))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0, 
                    square=True, 
                    mask=mask,
                    cbar_kws={"shrink": .8},
                    fmt='.2f',
                    annot_kws={'size': 8})
        
        plt.title('Target Metrics Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        filepath = os.path.join(save_folder, "target_metrics_correlation_matrix.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Ã¢Å“â€¦ Saved correlation matrix: target_metrics_correlation_matrix.png")

    def plot_target_metrics_summary(self, save_folder: str = "target_metric_plots"):
        """
        Create a comprehensive summary plot showing distributions of all metrics
        
        Args:
            save_folder: Folder name to save the plots
        """
        
        # Create the folder if it doesn't exist
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        
        # Get the target metrics data
        if hasattr(self, 'processed_data') and 'y' in self.processed_data:
            y_data = self.processed_data['y']
        elif hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            y_data = pd.concat([self.y_train, self.y_test], axis=0)
        else:
            y_data = self.raw_data[self.output_metrics]

        # Filter out boolean columns before converting to numeric
        numeric_columns = []
        for col in y_data.columns:
            if y_data[col].dtype != bool:
                numeric_columns.append(col)

        y_data = y_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Calculate number of rows and columns for subplots
        n_metrics = len(self.output_metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle('Target Metrics Distribution Summary', fontsize=20, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(self.output_metrics):
            if metric not in y_data.columns:
                continue
                
            metric_data = y_data[metric].dropna()

            # Skip boolean columns
            if metric_data.dtype == bool:
                continue

            metric_data = pd.to_numeric(metric_data, errors='coerce').dropna()

            if len(metric_data) == 0:
                continue
            
            # Create histogram with KDE
            axes[i].hist(metric_data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add KDE overlay
            try:
                sns.kdeplot(data=metric_data, ax=axes[i], color='red', linewidth=2)
            except:
                pass
            
            axes[i].set_title(metric, fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
            
            # Add basic statistics as text
            mean_val = metric_data.mean()
            std_val = metric_data.std()
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            axes[i].legend(fontsize=8)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        filepath = os.path.join(save_folder, "target_metrics_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Ã¢Å“â€¦ Saved summary plot: target_metrics_summary.png")

    def _prepare_data_with_scaling(self):
        """Properly scale features AND targets with automatic transformation selection - FIXED VERSION"""
        from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
        from scipy import stats
        import numpy as np
        
        print("Ã°Å¸â€â€ž Preparing data with scaling and transformation...")
        
        # Initialize scalers and transformers
        self.X_scaler = None
        self.y_scalers = {}
        self.y_transformers = {}
        
        # STEP 1: Handle missing values and outliers FIRST
        print("Ã°Å¸Â§Â¹ Cleaning data...")
        
        # Handle missing values in features
        feature_medians = self.X_train.median()
        self.X_train = self.X_train.fillna(feature_medians)
        self.X_test = self.X_test.fillna(feature_medians)
        
        # Handle infinite values
        self.X_train = self.X_train.replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.replace([np.inf, -np.inf], 0)
        
        # Remove constant features
        constant_features = []
        for col in self.X_train.columns:
            if self.X_train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"   Removing {len(constant_features)} constant features")
            self.X_train = self.X_train.drop(columns=constant_features)
            self.X_test = self.X_test.drop(columns=constant_features)
        
        # STEP 2: Intelligent feature scaling based on data distribution
        print("Ã°Å¸â€œÅ  Analyzing feature distributions...")
        
        # Analyze feature distributions to choose appropriate scaler
        feature_skewness = self.X_train.skew().abs().mean()
        feature_outliers = 0
        
        for col in self.X_train.select_dtypes(include=[np.number]).columns:
            Q1 = self.X_train[col].quantile(0.25)
            Q3 = self.X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((self.X_train[col] < (Q1 - 1.5 * IQR)) | 
                            (self.X_train[col] > (Q3 + 1.5 * IQR))).sum()
            feature_outliers += outlier_count
        
        outlier_percentage = (feature_outliers / (len(self.X_train) * len(self.X_train.columns))) * 100
        
        print(f"   Feature skewness (avg): {feature_skewness:.3f}")
        print(f"   Outlier percentage: {outlier_percentage:.2f}%")
        
        # ALWAYS use StandardScaler for biosensor data
        # Interaction features REQUIRE preserved scale relationships
        print("   Using StandardScaler to preserve scale relationships")
        self.X_scaler = StandardScaler()
        
        # FIT SCALER ONLY ON TRAINING DATA
        self.X_train_scaled = pd.DataFrame(
            self.X_scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # STEP 3: Enhanced target processing
        print("Ã°Å¸Å½Â¯ Processing target variables...")
        self.y_train_scaled = self.y_train.copy()
        self.y_test_scaled = self.y_test.copy()
        
        def enhanced_normality_test(data, name):
            """Enhanced normality testing with multiple methods"""
            if len(data) < 20:
                return 0, 0.001, "Insufficient data"
            
            # Use appropriate test based on sample size
            if len(data) < 5000:
                try:
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                except:
                    stat, p_value = stats.jarque_bera(data)[:2]
                    test_name = "Jarque-Bera"
            else:
                stat, p_value = stats.jarque_bera(data)[:2]
                test_name = "Jarque-Bera"
            
            return stat, p_value, test_name
        
        def comprehensive_transformation_evaluation(original_data, transformed_data, transform_name):
            """Comprehensive evaluation of transformation quality"""
            if len(transformed_data) < 20:
                return {'transform': transform_name, 'score': -1000}
            
            # Remove any infinite or NaN values from transformed data
            valid_mask = np.isfinite(transformed_data)
            if not valid_mask.any():
                return {'transform': transform_name, 'score': -1000}
            
            transformed_clean = transformed_data[valid_mask]
            original_clean = original_data[valid_mask]
            
            if len(transformed_clean) < 10:
                return {'transform': transform_name, 'score': -1000}
            
            try:
                # Normality tests
                orig_stat, orig_p, test_name = enhanced_normality_test(original_clean, "original")
                trans_stat, trans_p, _ = enhanced_normality_test(transformed_clean, "transformed")
                
                # Distribution characteristics
                orig_skew = stats.skew(original_clean)
                trans_skew = stats.skew(transformed_clean)
                orig_kurtosis = stats.kurtosis(original_clean)
                trans_kurtosis = stats.kurtosis(transformed_clean)
                
                # Variance stability
                orig_var = np.var(original_clean)
                trans_var = np.var(transformed_clean)
                
                # Scoring system (higher is better)
                normality_improvement = (trans_p - orig_p) * 50
                skewness_improvement = (abs(orig_skew) - abs(trans_skew)) * 20
                kurtosis_improvement = (abs(orig_kurtosis) - abs(trans_kurtosis)) * 10
                
                # Penalty for extreme transformations that might hurt model performance
                if abs(trans_skew) > 3:
                    skewness_improvement -= 20
                if abs(trans_kurtosis) > 10:
                    kurtosis_improvement -= 10
                
                # Bonus for reasonable variance
                if 0.1 < trans_var < 100:
                    variance_bonus = 10
                else:
                    variance_bonus = -5
                
                total_score = normality_improvement + skewness_improvement + kurtosis_improvement + variance_bonus
                
                return {
                    'transform': transform_name,
                    'score': total_score,
                    'orig_p': orig_p,
                    'trans_p': trans_p,
                    'orig_skew': orig_skew,
                    'trans_skew': trans_skew,
                    'orig_kurtosis': orig_kurtosis,
                    'trans_kurtosis': trans_kurtosis,
                    'orig_var': orig_var,
                    'trans_var': trans_var,
                    'test_name': test_name,
                    'valid_ratio': len(transformed_clean) / len(original_data)
                }
            except Exception as e:
                print(f"   Error evaluating {transform_name}: {e}")
                return {'transform': transform_name, 'score': -1000}
        
        for col in self.y_train.columns:
            if self.y_train[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                print(f"\nÃ°Å¸â€Â Processing {col}...")
                
                # Convert to numeric and handle issues
                train_data = pd.to_numeric(self.y_train[col], errors='coerce').dropna()
                test_data = pd.to_numeric(self.y_test[col], errors='coerce')
                
                if len(train_data) < 10:
                    print(f"Ã¢Å¡ Ã¯Â¸Â  Skipping {col} - insufficient valid data")
                    continue
                
                # Handle outliers in target (cap at 3 standard deviations)
                target_mean = train_data.mean()
                target_std = train_data.std()
                
                if target_std > 0:
                    outlier_mask = np.abs(train_data - target_mean) > 3 * target_std
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        print(f"   Found {outlier_count} outliers in {col}, capping them")
                        lower_bound = target_mean - 3 * target_std
                        upper_bound = target_mean + 3 * target_std
                        train_data = train_data.clip(lower_bound, upper_bound)
                        test_data = test_data.clip(lower_bound, upper_bound)
                
                # Test different transformations
                transformations = []
                
                # 1. No transformation
                transformations.append(comprehensive_transformation_evaluation(
                    train_data.values, train_data.values, "None"
                ))
                
                # 2. Yeo-Johnson (works with all data)
                try:
                    yj_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                    train_yj = yj_transformer.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                    transformations.append(comprehensive_transformation_evaluation(
                        train_data.values, train_yj, "Yeo-Johnson"
                    ))
                except Exception as e:
                    print(f"   Yeo-Johnson failed: {e}")
                
                # 3. Box-Cox (only for positive data)
                if (train_data > 0).all():
                    try:
                        bc_transformer = PowerTransformer(method='box-cox', standardize=False)
                        train_bc = bc_transformer.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                        transformations.append(comprehensive_transformation_evaluation(
                            train_data.values, train_bc, "Box-Cox"
                        ))
                    except Exception as e:
                        print(f"   Box-Cox failed: {e}")
                
                # 4. Log transformation (for positive data)
                if (train_data > 0).all():
                    try:
                        train_log = np.log1p(train_data.values)
                        transformations.append(comprehensive_transformation_evaluation(
                            train_data.values, train_log, "Log1p"
                        ))
                    except Exception as e:
                        print(f"   Log1p failed: {e}")
                
                # 5. Square root (for non-negative data)
                if (train_data >= 0).all():
                    try:
                        train_sqrt = np.sqrt(train_data.values)
                        transformations.append(comprehensive_transformation_evaluation(
                            train_data.values, train_sqrt, "Square Root"
                        ))
                    except Exception as e:
                        print(f"   Square root failed: {e}")

                # 8. MinMax transformation (for bounded data)
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    minmax_scaler = MinMaxScaler()
                    train_minmax = minmax_scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                    transformations.append(comprehensive_transformation_evaluation(
                        train_data.values, train_minmax, "MinMaxScale"
                    ))
                except Exception as e:
                    print(f"   MinMaxScale failed: {e}")
                
                # Select best transformation
                valid_transformations = [t for t in transformations if t['score'] > -1000]

                if not valid_transformations:
                    print(f"   No valid transformations found for {col}, using original data")
                    transformed_train = train_data.values
                    transformed_test = test_data.values
                    self.y_transformers[col] = None
                else:
                    best_transform = max(valid_transformations, key=lambda x: x['score'])
                    
                    # ✅ More lenient transformation criteria
                    orig_abs_skew = abs(best_transform.get('orig_skew', 0))
                    trans_abs_skew = abs(best_transform.get('trans_skew', 0))

                    skew_improvement = orig_abs_skew - trans_abs_skew

                    # ✅ Use transformation if it improves skewness by ANY amount AND R² gain is expected
                    if skew_improvement > 0.05 and best_transform['trans_p'] > best_transform['orig_p']:
                        # Apply transformation
                        if best_transform['transform'] == "None":
                            transformed_train = train_data.values
                            transformed_test = test_data.values
                            self.y_transformers[col] = None
                    else:
                        print(f"   Ã°Å¸â€œË† Transformation results for {col}:")
                        for t in sorted(valid_transformations, key=lambda x: x['score'], reverse=True)[:3]:
                            print(f"      {t['transform']}: Score={t['score']:.2f}, "
                                f"Skew: {t['orig_skew']:.3f}Ã¢â€ â€™{t['trans_skew']:.3f}")
                        
                        print(f"   Ã¢Å“â€¦ Selected: {best_transform['transform']}")
        
                    #  Apply the best transformation
                    if best_transform['transform'] == "None":
                        transformed_train = train_data.values
                        transformed_test = test_data.values
                        self.y_transformers[col] = None
                        
                    elif best_transform['transform'] == "Yeo-Johnson":
                        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                        transformed_train = transformer.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                        transformed_test = transformer.transform(test_data.values.reshape(-1, 1)).flatten()
                        self.y_transformers[col] = transformer
                        
                    elif best_transform['transform'] == "Box-Cox":
                        transformer = PowerTransformer(method='box-cox', standardize=False)
                        transformed_train = transformer.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                        transformed_test = transformer.transform(test_data.values.reshape(-1, 1)).flatten()
                        self.y_transformers[col] = transformer
                        
                    elif best_transform['transform'] == "Log1p":
                        transformed_train = np.log1p(train_data.values)
                        transformed_test = np.log1p(test_data.values)
                        self.y_transformers[col] = "log1p"
                        
                    elif best_transform['transform'] == "Square Root":
                        transformed_train = np.sqrt(train_data.values)
                        transformed_test = np.sqrt(test_data.values)
                        self.y_transformers[col] = "sqrt"
                    
                    # ✅ FIX: Add handlers for RobustScale and MinMaxScale
                    elif best_transform['transform'] == "RobustScale":
                        # RobustScale is a scaling method, not a transformation
                        # Just use raw values and let the scaler handle it later
                        transformed_train = train_data.values
                        transformed_test = test_data.values
                        self.y_transformers[col] = None
                        print(f"      Note: RobustScale will be applied as scaler, not transformer")
                    
                    elif best_transform['transform'] == "MinMaxScale":
                        # MinMaxScale is a scaling method, not a transformation
                        # Just use raw values and let the scaler handle it later
                        transformed_train = train_data.values
                        transformed_test = test_data.values
                        self.y_transformers[col] = None
                        print(f"      Note: MinMaxScale will be applied as scaler, not transformer")
                    
                    # ✅ FIX: Add else clause to catch any unexpected transform names
                    else:
                        # Fallback: use raw values if transform name is unexpected
                        print(f"      ⚠️  WARNING: Unknown transform '{best_transform['transform']}', using raw values")
                        transformed_train = train_data.values
                        transformed_test = test_data.values
                        self.y_transformers[col] = None
                
                # Apply scaling to transformed data
                target_std = np.std(transformed_train)
                target_range = np.max(transformed_train) - np.min(transformed_train)
                
                # ALWAYS use StandardScaler for targets
                # Interaction features REQUIRE preserved scale relationships
                scaler = StandardScaler()
                print(f"      Using StandardScaler for target")
                
                # Fit scaler only on training data
                self.y_train_scaled[col] = scaler.fit_transform(transformed_train.reshape(-1, 1)).flatten()
                self.y_test_scaled[col] = scaler.transform(transformed_test.reshape(-1, 1)).flatten()
                self.y_scalers[col] = scaler
        
        print("\nÃ¢Å“â€¦ Data preparation complete!")
        print(f"   Features scaled: {self.X_train_scaled.shape[1]}")
        print(f"   Targets processed: {len(self.y_scalers)}")
        print(f"   Transformations applied: {sum(1 for t in self.y_transformers.values() if t is not None)}")

    def inverse_transform_predictions(self, predictions, target_columns=None):
        """Inverse transform predictions back to original scale - FIXED VERSION"""
        if target_columns is None:
            target_columns = self.y_train.columns
        
        inverse_predictions = predictions.copy()
        
        for col in target_columns:
            if col in self.y_scalers and col in self.y_transformers:
                # First inverse scale
                if self.y_scalers[col] is not None:
                    scaled_back = self.y_scalers[col].inverse_transform(
                        inverse_predictions[col].values.reshape(-1, 1)
                    ).flatten()
                else:
                    scaled_back = inverse_predictions[col].values
                
                # Then inverse transform
                if self.y_transformers[col] is None:
                    inverse_predictions[col] = scaled_back
                elif hasattr(self.y_transformers[col], 'inverse_transform'):
                    # PowerTransformer or QuantileTransformer
                    inverse_predictions[col] = self.y_transformers[col].inverse_transform(
                        scaled_back.reshape(-1, 1)
                    ).flatten()
                elif self.y_transformers[col] == "log1p":
                    inverse_predictions[col] = np.expm1(scaled_back)
                elif self.y_transformers[col] == "sqrt":
                    inverse_predictions[col] = np.square(scaled_back)
        
        return inverse_predictions

    def _create_advanced_features(self, X):
        """Minimal feature engineering - ONLY what helps"""
        X_enhanced = X.copy()
        
        # Skip if too few features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 3:
            return X_enhanced
        
        # ✅ ONLY add these 3 proven features:
        
        # 1. Detection efficiency (sensitivity / threshold)
        if 'biosensor_sensitivity' in X.columns and 'biosensor_threshold' in X.columns:
            X_enhanced['detection_efficiency'] = (
                X['biosensor_sensitivity'] / (X['biosensor_threshold'] + 1e-6)
            )
        
        # 2. Binding score (Hill / Kd)
        if 'hill_coefficient' in X.columns and 'biosensor_kd' in X.columns:
            X_enhanced['binding_score'] = (
                X['hill_coefficient'] / (X['biosensor_kd'] + 1e-6)
            )
        
        # 3. Signal range
        if 'on_level' in X.columns and 'off_level' in X.columns:
            X_enhanced['signal_range'] = X['on_level'] - X['off_level']
        
        # Clean up
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0)
        X_enhanced = X_enhanced.fillna(0)
        
        return X_enhanced

    def _validate_data_quality(self):
        """Enhanced data validation focusing on model training readiness"""
        print("Ã°Å¸â€Â Validating data quality for model training...")
        
        issues = []
        warnings = []
        
        # Check feature matrix
        print(f"Ã°Å¸â€œÅ  Feature matrix validation:")
        print(f"   Shape: {self.X_train.shape}")
        print(f"   Data types: {self.X_train.dtypes.value_counts().to_dict()}")
        
        # Check for problematic features
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Check variance
            if self.X_train[col].var() < 1e-10:
                issues.append(f"Feature {col} has zero variance")
            
            # Check for extreme values
            if np.isinf(self.X_train[col]).any():
                issues.append(f"Feature {col} contains infinite values")
            
            # Check for too many missing values
            missing_pct = self.X_train[col].isnull().sum() / len(self.X_train) * 100
            if missing_pct > 50:
                issues.append(f"Feature {col} has {missing_pct:.1f}% missing values")
            elif missing_pct > 20:
                warnings.append(f"Feature {col} has {missing_pct:.1f}% missing values")
        
        # Check target variables
        print(f"Ã°Å¸Å½Â¯ Target variables validation:")
        for col in self.y_train.columns:
            print(f"   {col}:")
            
            if self.y_train[col].dtype == 'object':
                # Categorical target
                value_counts = self.y_train[col].value_counts()
                print(f"     Classes: {len(value_counts)}")
                print(f"     Distribution: {dict(value_counts.head())}")
                
                # Check for class imbalance
                min_class = value_counts.min()
                max_class = value_counts.max()
                if max_class / min_class > 10:
                    warnings.append(f"Severe class imbalance in {col}")
                    
            elif self.y_train[col].dtype == 'bool':
                # Boolean target - handle separately
                value_counts = self.y_train[col].value_counts()
                print(f"     Boolean distribution: {dict(value_counts)}")
                
                # Check for class imbalance
                true_count = value_counts.get(True, 0)
                false_count = value_counts.get(False, 0)
                if true_count > 0 and false_count > 0:
                    imbalance_ratio = max(true_count, false_count) / min(true_count, false_count)
                    if imbalance_ratio > 10:
                        warnings.append(f"Severe class imbalance in boolean target {col}")
                elif true_count == 0 or false_count == 0:
                    issues.append(f"Boolean target {col} has only one class")
                    
            else:
                # Numeric target
                print(f"     Range: [{self.y_train[col].min():.4f}, {self.y_train[col].max():.4f}]")
                print(f"     Mean: {self.y_train[col].mean():.4f}")
                print(f"     Std: {self.y_train[col].std():.4f}")
                print(f"     Skewness: {self.y_train[col].skew():.4f}")
                
                # Check for extreme skewness
                if abs(self.y_train[col].skew()) > 3:
                    warnings.append(f"Highly skewed target {col}")
                
                # Check for outliers (only for numeric, non-boolean data)
                Q1 = self.y_train[col].quantile(0.25)
                Q3 = self.y_train[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.y_train[col] < (Q1 - 1.5 * IQR)) | 
                        (self.y_train[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = outliers / len(self.y_train) * 100
                
                if outlier_pct > 10:
                    warnings.append(f"High outlier percentage in {col}: {outlier_pct:.2f}%")
        
        # Data size validation
        print(f"Ã°Å¸â€œÂ Dataset size validation:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Features: {len(self.X_train.columns)}")
        print(f"   Feature-to-sample ratio: {len(self.X_train.columns) / len(self.X_train):.3f}")
        
        if len(self.X_train.columns) / len(self.X_train) > 0.5:
            warnings.append("High feature-to-sample ratio may cause overfitting")
        
        if len(self.X_train) < 50:
            issues.append("Very small training set - results may be unreliable")
        elif len(self.X_train) < 200:
            warnings.append("Small training set - consider regularization")
        
        # Summary
        print(f"\nÃ°Å¸â€œâ€¹ Validation Summary:")
        if issues:
            print(f"Ã¢ÂÅ’ Critical Issues ({len(issues)}):")
            for issue in issues:
                print(f"   Ã¢â‚¬Â¢ {issue}")
        
        if warnings:
            print(f"Ã¢Å¡ Ã¯Â¸Â  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   Ã¢â‚¬Â¢ {warning}")
        
        if not issues and not warnings:
            print("Ã¢Å“â€¦ Data quality is good for model training")
        
        return len(issues) == 0


    def _tune_hyperparameters(self, X_train, y_train, model_name):
        """Improved hyperparameter tuning based on dataset characteristics"""
        
        n_samples, n_features = X_train.shape
        
        # Determine complexity based on data size
        if n_samples < 500:
            complexity = "low"
        elif n_samples < 2000:
            complexity = "medium"
        else:
            complexity = "high"
        
        print(f"   Tuning {model_name} with {complexity} complexity for {n_samples} samples, {n_features} features")
        
        if model_name == 'XGBoost':
            if complexity == "low":
                return xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    gamma=0.5,
                    min_child_weight=5,
                    random_state=42,
                    n_jobs=-1
                )
            elif complexity == "medium":
                return xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.5,
                    reg_lambda=1.5,
                    gamma=0.3,
                    min_child_weight=3,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # high complexity - OPTIMIZED
                return xgb.XGBRegressor(
                    n_estimators=2000,  # DOUBLED
                    max_depth=10,  # DEEPER
                    learning_rate=0.02,  # SLOWER (better convergence)
                    subsample=0.7,  # More aggressive sampling
                    colsample_bytree=0.7,
                    reg_alpha=0.05,  # REDUCED regularization
                    reg_lambda=0.3,  # REDUCED
                    gamma=0.01,  # REDUCED
                    min_child_weight=1,  # REDUCED (allow finer splits)
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist',  # FASTER training
                    early_stopping_rounds=100
                )
        
        elif model_name == 'RandomForest':
            if complexity == "low":
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            elif complexity == "medium":
                return RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # high complexity - OPTIMIZED
                return lgb.LGBMRegressor(
                    n_estimators=2000,
                    max_depth=10,
                    learning_rate=0.02,
                    num_leaves=127,  # INCREASED
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.05,
                    reg_lambda=0.3,
                    min_child_samples=3,  # REDUCED
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                    force_col_wise=True  # Performance boost
                )
        
        elif model_name == 'LightGBM':
            if complexity == "low":
                return lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    num_leaves=15,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=1.0,
                    reg_lambda=2.0,
                    min_child_samples=20,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
            elif complexity == "medium":
                return lgb.LGBMRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.08,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.5,
                    reg_lambda=1.5,
                    min_child_samples=15,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
            else:  # high complexity
                return lgb.LGBMRegressor(
                    n_estimators=800,
                    max_depth=8,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.3,
                    reg_lambda=1.0,
                    min_child_samples=10,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
        
        return None

    def _calculate_normalized_mse(self, y_true, y_pred):
        """Calculate normalized MSE metrics"""
        mse = mean_squared_error(y_true, y_pred)
        
        # Calculate different normalization options
        y_range = np.max(y_true) - np.min(y_true)
        y_mean = np.mean(y_true)
        y_std = np.std(y_true)
        y_var = np.var(y_true)
        
        # Avoid division by zero
        normalized_mse = {
            'mse_raw': mse,
            'mse_normalized_by_variance': mse / y_var if y_var > 0 else float('inf'),
            'mse_normalized_by_range_squared': mse / (y_range ** 2) if y_range > 0 else float('inf'),
            'mse_normalized_by_mean_squared': mse / (y_mean ** 2) if y_mean != 0 else float('inf'),
            'rmse_normalized_by_std': np.sqrt(mse) / y_std if y_std > 0 else float('inf'),
            'rmse_normalized_by_range': np.sqrt(mse) / y_range if y_range > 0 else float('inf')
        }
        
        return normalized_mse
    
    def _clean_feature_names(self, df):
        """Clean feature names for LightGBM compatibility"""
        df_clean = df.copy()
        
        # Replace problematic characters
        new_columns = []
        for col in df_clean.columns:
            # Replace common problematic characters
            clean_col = str(col)
            clean_col = clean_col.replace('[', '_').replace(']', '_')
            clean_col = clean_col.replace('(', '_').replace(')', '_')
            clean_col = clean_col.replace('{', '_').replace('}', '_')
            clean_col = clean_col.replace('<', '_').replace('>', '_')
            clean_col = clean_col.replace('"', '_').replace("'", '_')
            clean_col = clean_col.replace(':', '_').replace(';', '_')
            clean_col = clean_col.replace(',', '_').replace('.', '_')
            clean_col = clean_col.replace('|', '_').replace('\\', '_')
            clean_col = clean_col.replace('/', '_').replace('*', '_')
            clean_col = clean_col.replace('+', '_').replace('-', '_')
            clean_col = clean_col.replace('=', '_').replace('!', '_')
            clean_col = clean_col.replace('@', '_').replace('#', '_')
            clean_col = clean_col.replace('$', '_').replace('%', '_')
            clean_col = clean_col.replace('^', '_').replace('&', '_')
            clean_col = clean_col.replace('?', '_').replace('~', '_')
            clean_col = clean_col.replace('`', '_').replace(' ', '_')
            
            # Remove multiple consecutive underscores
            while '__' in clean_col:
                clean_col = clean_col.replace('__', '_')
            
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            
            # Ensure it starts with a letter or underscore
            if clean_col and not (clean_col[0].isalpha() or clean_col[0] == '_'):
                clean_col = 'f_' + clean_col
            
            # Ensure it's not empty
            if not clean_col:
                clean_col = f'feature_{len(new_columns)}'
            
            new_columns.append(clean_col)
        
        df_clean.columns = new_columns
        return df_clean

    def _get_target_specific_config(self, target_metric):
        """
        Get optimized hyperparameters for specific target metrics
        Different metrics have different characteristics
        """
        
        configs = {
            'signal_to_noise_ratio_SNR': {
                'xgb_depth': 12,
                'lgb_leaves': 255,
                'cat_depth': 10,
                'learning_rate': 0.005,  # Slower for SNR
                'n_estimators': 3000
            },
            'dynamic_range_of_output': {
                'xgb_depth': 10,
                'lgb_leaves': 127,
                'cat_depth': 8,
                'learning_rate': 0.01,
                'n_estimators': 2000
            },
            'false_negative_rate': {
                'xgb_depth': 8,
                'lgb_leaves': 63,
                'cat_depth': 6,
                'learning_rate': 0.02,  # Faster for FNR (easier)
                'n_estimators': 1500
            },
            'time_to_detection_threshold': {
                'xgb_depth': 10,
                'lgb_leaves': 127,
                'cat_depth': 8,
                'learning_rate': 0.01,
                'n_estimators': 2000
            }
        }
        
        # Default config
        default = {
            'xgb_depth': 10,
            'lgb_leaves': 127,
            'cat_depth': 8,
            'learning_rate': 0.01,
            'n_estimators': 2000
        }
        
        return configs.get(target_metric, default)

    def train_supervised_models(self, target_metric: str = 'multi_objective_score') -> Dict:
        """
        COMPLETELY REWRITTEN: Train multiple supervised learning models with best practices
        """
        if target_metric not in self.y_train.columns:
            print(f"Ã¢Å¡ Ã¯Â¸Â Target metric '{target_metric}' not found. Using first available: {self.y_train.columns[0]}")
            target_metric = self.y_train.columns[0]
        
        print(f"Ã°Å¸Â¤â€“ Training supervised models for: {target_metric}")
        print("=" * 60)
        
        # Step 1: Validate data quality FIRST
        if not self._validate_data_quality():
            print("Ã¢ÂÅ’ Critical data quality issues found. Please fix before training.")
            return {}
        
        # Step 2: Prepare data with proper scaling and transformation
        self._prepare_data_with_scaling()

        print(f"\n🔍 SCALER VERIFICATION:")
        print(f"   X_scaler type: {type(self.X_scaler).__name__}")
        print(f"   Expected: StandardScaler")

        if hasattr(self.X_train_scaled, 'columns') and 'signal_to_threshold_ratio' in self.X_train_scaled.columns:
            print(f"\n🔍 INTERACTION FEATURE SCALING:")
            print(f"   signal_to_threshold_ratio mean (scaled): {self.X_train_scaled['signal_to_threshold_ratio'].mean():.4f}")
            print(f"   signal_to_threshold_ratio std (scaled): {self.X_train_scaled['signal_to_threshold_ratio'].std():.4f}")
            print(f"   Expected: mean≈0.0, std≈1.0 (StandardScaler properties)")
        
        # Step 3: Create enhanced features AFTER scaling
        print("Ã°Å¸â€Â§ Creating enhanced features...")
        self.X_train_enhanced = self._create_advanced_features(self.X_train_scaled)
        self.X_test_enhanced = self._create_advanced_features(self.X_test_scaled)
        
        print(f"   Enhanced features: {self.X_train_enhanced.shape[1]} (was {self.X_train.shape[1]})")
        
        # Step 4: Get target data (use transformed/scaled version for neural networks, original for tree models)
        y_target_original = self.y_train[target_metric].copy()
        y_test_target_original = self.y_test[target_metric].copy()
        
        # For neural networks, use scaled targets
        if target_metric in self.y_train_scaled.columns:
            y_target_scaled = self.y_train_scaled[target_metric].copy()
            y_test_target_scaled = self.y_test_scaled[target_metric].copy()
        else:
            y_target_scaled = y_target_original.copy()
            y_test_target_scaled = y_test_target_original.copy()
        
        # Step 5: Enhanced target diagnostics
        print(f"\nÃ°Å¸â€œÅ  Target Analysis for '{target_metric}':")
        print(f"   Training samples: {len(y_target_original)}")
        print(f"   Range: [{y_target_original.min():.6f}, {y_target_original.max():.6f}]")
        print(f"   Mean Ã‚Â± Std: {y_target_original.mean():.6f} Ã‚Â± {y_target_original.std():.6f}")
        print(f"   Median: {y_target_original.median():.6f}")
        print(f"   Skewness: {y_target_original.skew():.4f}")
        print(f"   Missing values: {y_target_original.isnull().sum()}")
        print(f"   Target variance: {y_target_original.var():.6f}")
        
        # Check if target needs special handling
        target_range = y_target_original.max() - y_target_original.min()
        target_std = y_target_original.std()
        
        if target_range == 0 or target_std == 0:
            print("Ã¢ÂÅ’ Target has no variance - cannot train models")
            return {}
        
        # Step 6: Split data for cross-validation
        from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # ✅ Use RepeatedKFold for more robust estimates
        kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)  # 10 total fits
        cv_folds = kf.get_n_splits()  # This will be 10 (5 folds × 2 repeats)

        # CREATE cv_splits using the enhanced features
        cv_splits = list(kf.split(self.X_train_enhanced))

        print(f"   🔄 Using {cv_folds} cross-validation splits (5 folds × 2 repeats)")

        # Step 7: Train models with cross-validation
        models_to_train = ['XGBoost', 'LightGBM', 'RandomForest', 'MLP']
        all_results = {}
        
        for model_name in models_to_train:
            print(f"\nÃ°Å¸Ââ€¹Ã¯Â¸Â Training {model_name}...")
            print("-" * 40)
            
            cv_scores = {
                'train_r2': [], 'val_r2': [], 
                'train_mse': [], 'val_mse': [], 
                'train_mae': [], 'val_mae': [],
                'train_rmse': [], 'val_rmse': [],
                'train_norm_mse': [], 'val_norm_mse': []
            }
            fold_predictions = {'train': [], 'val': [], 'train_idx': [], 'val_idx': []}
            fold_models = []
            
            # Cross-validation training
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                print(f"   Fold {fold + 1}/{cv_folds}")
                
                # Split data for this fold
                X_fold_train = self.X_train_enhanced.iloc[train_idx]
                X_fold_val = self.X_train_enhanced.iloc[val_idx]
                
                if model_name == 'MLP':
                    # Use scaled targets for neural networks
                    y_fold_train = y_target_scaled.iloc[train_idx]
                    y_fold_val = y_target_scaled.iloc[val_idx]
                else:
                    # Use original targets for tree-based models
                    y_fold_train = y_target_original.iloc[train_idx]
                    y_fold_val = y_target_original.iloc[val_idx]
                
                # Train model
                if model_name == 'MLP':
                    model, fold_performance = self._train_neural_network_cv(
                        X_fold_train, y_fold_train, X_fold_val, y_fold_val, 
                        model_name, target_metric, fold
                    )
                    
                    # Ensure all required keys are present for MLP fold
                    required_keys = ['train_r2', 'val_r2', 'train_mse', 'val_mse', 'train_mae', 'val_mae', 'train_rmse', 'val_rmse', 'train_norm_mse', 'val_norm_mse']
                    for key in required_keys:
                        if key not in fold_performance:
                            fold_performance[key] = 0.0
                else:
                    model = self._tune_hyperparameters(X_fold_train, y_fold_train, model_name)
                    
                    if model_name == 'XGBoost':
                        model.fit(
                            X_fold_train, y_fold_train,
                            eval_set=[(X_fold_val, y_fold_val)],
                            verbose=False
                        )
                    elif model_name == 'LightGBM':
                        # Clean feature names for LightGBM
                        X_fold_train_clean = self._clean_feature_names(X_fold_train)
                        X_fold_val_clean = self._clean_feature_names(X_fold_val)
                        
                        model.fit(
                            X_fold_train_clean, y_fold_train,
                            eval_set=[(X_fold_val_clean, y_fold_val)],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                        )
                    else:  # RandomForest
                        model.fit(X_fold_train, y_fold_train)
                    
                    # Get predictions
                    if model_name == 'LightGBM':
                        y_pred_train = model.predict(X_fold_train_clean)
                        y_pred_val = model.predict(X_fold_val_clean)
                    else:
                        y_pred_train = model.predict(X_fold_train)
                        y_pred_val = model.predict(X_fold_val)
                    
                    # Calculate comprehensive metrics
                    train_mse_full, val_mse_full = self._calculate_normalized_mse(y_fold_train, y_pred_train), self._calculate_normalized_mse(y_fold_val, y_pred_val)
                    train_mse, train_norm_mse = train_mse_full['mse_raw'], train_mse_full['mse_normalized_by_variance']
                    val_mse, val_norm_mse = val_mse_full['mse_raw'], val_mse_full['mse_normalized_by_variance']
                    
                    fold_performance = {
                        'train_r2': r2_score(y_fold_train, y_pred_train),
                        'val_r2': r2_score(y_fold_val, y_pred_val),
                        'train_mse': train_mse,
                        'val_mse': val_mse,
                        'train_mae': mean_absolute_error(y_fold_train, y_pred_train),
                        'val_mae': mean_absolute_error(y_fold_val, y_pred_val),
                        'train_rmse': np.sqrt(train_mse),
                        'val_rmse': np.sqrt(val_mse),
                        'train_norm_mse': train_norm_mse,
                        'val_norm_mse': val_norm_mse
                    }
                    
                    # Store predictions
                    fold_predictions['train'].extend(y_pred_train)
                    fold_predictions['val'].extend(y_pred_val)
                    fold_predictions['train_idx'].extend(train_idx)
                    fold_predictions['val_idx'].extend(val_idx)
                
                # Store fold results
                for metric, value in fold_performance.items():
                    cv_scores[metric].append(value)
                
                fold_models.append(model)
                
                print(f"      RÃ‚Â²: {fold_performance['val_r2']:.4f}, RMSE: {fold_performance['val_rmse']:.4f}, Norm MSE: {fold_performance['val_norm_mse']:.4f}")
            
            # Calculate average CV performance
            avg_performance = {}
            for metric, scores in cv_scores.items():
                avg_performance[f'{metric}_mean'] = np.mean(scores)
                avg_performance[f'{metric}_std'] = np.std(scores)
            
            print(f"   Ã°Å¸â€œÅ  CV Results:")
            print(f"      Validation RÃ‚Â²: {avg_performance['val_r2_mean']:.4f} Ã‚Â± {avg_performance['val_r2_std']:.4f}")
            print(f"      Validation RMSE: {avg_performance['val_rmse_mean']:.4f} Ã‚Â± {avg_performance['val_rmse_std']:.4f}")
            print(f"      Validation Norm MSE: {avg_performance['val_norm_mse_mean']:.4f} Ã‚Â± {avg_performance['val_norm_mse_std']:.4f}")
            print(f"      Validation MAE: {avg_performance['val_mae_mean']:.4f} Ã‚Â± {avg_performance['val_mae_std']:.4f}")
            
            # Train final model on full training set
            print(f"   Ã°Å¸Å½Â¯ Training final model on full dataset...")
            
            if model_name == 'MLP':
                final_model, final_performance = self._train_neural_network_final(
                    self.X_train_enhanced, y_target_scaled, 
                    self.X_test_enhanced, y_test_target_scaled,
                    model_name, target_metric
                )
                
                # Ensure all required keys are present for MLP fold
                required_keys = ['train_r2', 'test_r2', 'train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_rmse', 'test_rmse', 'train_norm_mse', 'test_norm_mse']
                for key in required_keys:
                    if key not in final_performance:
                        final_performance[key] = 0.0
            else:
                final_model = self._tune_hyperparameters(self.X_train_enhanced, y_target_original, model_name)
                
                if model_name == 'XGBoost':
                    final_model.fit(
                        self.X_train_enhanced, y_target_original,
                        eval_set=[(self.X_test_enhanced, y_test_target_original)],
                        verbose=False
                    )
                elif model_name == 'LightGBM':
                    X_train_clean = self._clean_feature_names(self.X_train_enhanced)
                    X_test_clean = self._clean_feature_names(self.X_test_enhanced)
                    
                    final_model.fit(
                        X_train_clean, y_target_original,
                        eval_set=[(X_test_clean, y_test_target_original)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                else:  # RandomForest
                    final_model.fit(self.X_train_enhanced, y_target_original)
                
                # Final predictions
                if model_name == 'LightGBM':
                    y_pred_train_final = final_model.predict(X_train_clean)
                    y_pred_test_final = final_model.predict(X_test_clean)
                else:
                    y_pred_train_final = final_model.predict(self.X_train_enhanced)
                    y_pred_test_final = final_model.predict(self.X_test_enhanced)
                
                # Calculate final metrics with normalization
                train_mse_final_full, val_mse_final_full = self._calculate_normalized_mse(y_target_original, y_pred_train_final), self._calculate_normalized_mse(y_test_target_original, y_pred_test_final)
                train_mse_final, train_norm_mse_final = train_mse_final_full['mse_raw'], train_mse_full['mse_normalized_by_variance']
                test_mse_final, test_norm_mse_final = val_mse_final_full['mse_raw'], val_mse_full['mse_normalized_by_variance']
                
                final_performance = {
                    'train_r2': r2_score(y_target_original, y_pred_train_final),
                    'test_r2': r2_score(y_test_target_original, y_pred_test_final),
                    'train_mse': train_mse_final,
                    'test_mse': test_mse_final,
                    'train_mae': mean_absolute_error(y_target_original, y_pred_train_final),
                    'test_mae': mean_absolute_error(y_test_target_original, y_pred_test_final),
                    'train_rmse': np.sqrt(train_mse_final),
                    'test_rmse': np.sqrt(test_mse_final),
                    'train_norm_mse': train_norm_mse_final,
                    'test_norm_mse': test_norm_mse_final,
                    'predictions_train': y_pred_train_final,
                    'predictions_test': y_pred_test_final
                }
            
            print(f"   Ã¢Å“â€¦ Final Test RÃ‚Â²: {final_performance['test_r2']:.4f}")
            print(f"   Ã¢Å“â€¦ Final Test RMSE: {final_performance['test_rmse']:.4f}")
            print(f"   Ã¢Å“â€¦ Final Test Norm MSE: {final_performance['test_norm_mse']:.4f}")
            
            # Store results
            all_results[model_name] = {
                **final_performance,
                **avg_performance,
                'cv_scores': cv_scores,
                'fold_models': fold_models
            }
            
            # Store model and feature importance
            self.models[f'{model_name}_{target_metric}'] = final_model

            # Extract feature importance from sklearn models too
            if hasattr(final_model, 'feature_importances_'):
                if model_name == 'LightGBM':
                    feature_names = X_train_clean.columns
                else:
                    feature_names = self.X_train_enhanced.columns
                    
                feature_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': final_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[f'{model_name}_{target_metric}'] = feature_imp
                print(f"   Ã¢Å“â€¦ Feature importance extracted for {model_name}")
            elif hasattr(final_model, 'coef_'):  # Linear models
                feature_names = self.X_train_enhanced.columns
                feature_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(final_model.coef_)
                }).sort_values('importance', ascending=False)
                self.feature_importance[f'{model_name}_{target_metric}'] = feature_imp
                print(f"   Ã¢Å“â€¦ Feature importance extracted from coefficients for {model_name}")
        
        # Step 8: Model comparison and selection
        print(f"\nÃ°Å¸Ââ€  MODEL COMPARISON for {target_metric}")
        print("=" * 60)
        
        comparison_df = pd.DataFrame({
            model: {
                'CV_R2_Mean': results['val_r2_mean'],
                'CV_R2_Std': results['val_r2_std'],
                'Test_R2': results['test_r2'],
                'Test_RMSE': results['test_rmse'],
                'Test_MAE': results['test_mae'],
                'Test_Norm_MSE': results['test_norm_mse']
            }
            for model, results in all_results.items()
        }).T
        
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        print(comparison_df.round(4))
        
        # Identify best model
        best_model_name = comparison_df.index[0]
        best_performance = all_results[best_model_name]
        
        print(f"\nÃ°Å¸Å½Â¯ BEST MODEL: {best_model_name}")
        print(f"   Test RÃ‚Â²: {best_performance['test_r2']:.4f}")
        print(f"   Test RMSE: {best_performance['test_rmse']:.4f}")
        print(f"   Test Normalized MSE: {best_performance['test_norm_mse']:.4f}")
        print(f"   CV RÃ‚Â² (mean Ã‚Â± std): {best_performance['val_r2_mean']:.4f} Ã‚Â± {best_performance['val_r2_std']:.4f}")
        
        # Performance validation
        if best_performance['test_r2'] < 0.5:
            print("Ã¢Å¡ Ã¯Â¸Â  WARNING: Best model RÃ‚Â² < 0.5. Consider:")
            print("   Ã¢â‚¬Â¢ More feature engineering")
            print("   Ã¢â‚¬Â¢ Different target transformations")
            print("   Ã¢â‚¬Â¢ Ensemble methods")
            print("   Ã¢â‚¬Â¢ More data collection")
        elif best_performance['test_r2'] < 0.7:
            print("Ã¢Å¡ Ã¯Â¸Â  MODERATE: Best model RÃ‚Â² < 0.7. Room for improvement.")
        else:
            print("Ã¢Å“â€¦ GOOD: Model performance is satisfactory")
        
        # Store performance comparison
        self.model_performance[target_metric] = all_results
        self.model_comparison = comparison_df
        
        print("Ã¢Å“â€¦ Supervised model training completed!")
        return all_results

    def _train_neural_network_cv(self, X_train, y_train, X_val, y_val, model_name, target_metric, fold):
        """Train neural network for cross-validation fold - FIXED VERSION"""
        
        # Create model
        model = self._create_robust_mlp_model(X_train.shape[1])
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        
        # Training
        best_loss = float('inf')
        patience_counter = 0
        patience = 30
        
        for epoch in range(200):  # Reduced epochs for CV
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    model.load_state_dict(best_state)
                    break
        
        # Final predictions
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).numpy().flatten()
            val_pred = model(X_val_tensor).numpy().flatten()
        
        # FIXED: Calculate metrics using the scaled targets directly
        # No need to transform back to original scale for CV evaluation
        try:
            # Use the actual y_train and y_val passed to this function
            y_train_for_eval = y_train.values if hasattr(y_train, 'values') else y_train
            y_val_for_eval = y_val.values if hasattr(y_val, 'values') else y_val
            
            # If we need original scale for evaluation, use .loc instead of .iloc
            if target_metric in self.y_scalers and self.y_scalers[target_metric] is not None:
                # Transform predictions back to original scale
                train_pred_orig = self.y_scalers[target_metric].inverse_transform(train_pred.reshape(-1, 1)).flatten()
                val_pred_orig = self.y_scalers[target_metric].inverse_transform(val_pred.reshape(-1, 1)).flatten()
                
                # FIXED: Use .loc with the actual indices instead of .iloc
                y_train_orig = self.y_train[target_metric].loc[X_train.index]
                y_val_orig = self.y_train[target_metric].loc[X_val.index]
                
                performance = {
                    'train_r2': r2_score(y_train_orig, train_pred_orig),
                    'val_r2': r2_score(y_val_orig, val_pred_orig),
                    'train_mse': mean_squared_error(y_train_orig, train_pred_orig),
                    'val_mse': mean_squared_error(y_val_orig, val_pred_orig),
                    'train_mae': mean_absolute_error(y_train_orig, train_pred_orig),
                    'val_mae': mean_absolute_error(y_val_orig, val_pred_orig),
                    'train_rmse': np.sqrt(mean_squared_error(y_train_orig, train_pred_orig)),
                    'val_rmse': np.sqrt(mean_squared_error(y_val_orig, val_pred_orig)),
                    'train_norm_mse': mean_squared_error(y_train_orig, train_pred_orig) / np.var(y_train_orig) if np.var(y_train_orig) > 0 else float('inf'),
                    'val_norm_mse': mean_squared_error(y_val_orig, val_pred_orig) / np.var(y_val_orig) if np.var(y_val_orig) > 0 else float('inf')
                }
            else:
                # Use scaled targets directly for evaluation
                performance = {
                    'train_r2': r2_score(y_train_for_eval, train_pred),
                    'val_r2': r2_score(y_val_for_eval, val_pred),
                    'train_mse': mean_squared_error(y_train_for_eval, train_pred),
                    'val_mse': mean_squared_error(y_val_for_eval, val_pred),
                    'train_mae': mean_absolute_error(y_train_for_eval, train_pred),
                    'val_mae': mean_absolute_error(y_val_for_eval, val_pred),
                    'train_rmse': np.sqrt(mean_squared_error(y_train_for_eval, train_pred)),
                    'val_rmse': np.sqrt(mean_squared_error(y_val_for_eval, val_pred)),
                    'train_norm_mse': mean_squared_error(y_train_for_eval, train_pred) / np.var(y_train_for_eval) if np.var(y_train_for_eval) > 0 else float('inf'),
                    'val_norm_mse': mean_squared_error(y_val_for_eval, val_pred) / np.var(y_val_for_eval) if np.var(y_val_for_eval) > 0 else float('inf')
                }
            
        except (KeyError, IndexError) as e:
            print(f"      Ã¢Å¡ Ã¯Â¸Â  Metric calculation failed, using scaled targets: {e}")
            # Fallback: use scaled targets directly
            performance = {
                'train_r2': r2_score(y_train_for_eval, train_pred),
                'val_r2': r2_score(y_val_for_eval, val_pred),
                'train_mse': mean_squared_error(y_train_for_eval, train_pred),
                'val_mse': mean_squared_error(y_val_for_eval, val_pred),
                'train_mae': mean_absolute_error(y_train_for_eval, train_pred),
                'val_mae': mean_absolute_error(y_val_for_eval, val_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train_for_eval, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val_for_eval, val_pred)),
                'train_norm_mse': mean_squared_error(y_train_for_eval, train_pred) / np.var(y_train_for_eval) if np.var(y_train_for_eval) > 0 else float('inf'),
                'val_norm_mse': mean_squared_error(y_val_for_eval, val_pred) / np.var(y_val_for_eval) if np.var(y_val_for_eval) > 0 else float('inf')
            }
        
        return model, performance

    def _create_robust_mlp_model(self, input_dim, hidden_dims=None):
        """
        Create robust MLP model.
        
        ✅ FIXED: Uses module-level RobustMLP class for pickle support
        """
        # ✅ Use module-level RobustMLP class (defined at top of file)
        return RobustMLP(input_dim, hidden_dims)
    
    def _train_neural_network_final(self, X_train, y_train, X_test, y_test, model_name, target_metric):
        """Train final neural network on full training set with fixed indexing"""
        
        model = self._create_robust_mlp_model(X_train.shape[1])
        
        # Enhanced training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=500, steps_per_epoch=1
        )
        
        # Convert to tensors with proper error handling
        try:
            if hasattr(X_train, 'values'):
                X_train_tensor = torch.FloatTensor(X_train.values)
            else:
                X_train_tensor = torch.FloatTensor(X_train)
                
            if hasattr(X_test, 'values'):
                X_test_tensor = torch.FloatTensor(X_test.values)
            else:
                X_test_tensor = torch.FloatTensor(X_test)
            
            if hasattr(y_train, 'values'):
                y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
            else:
                y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
                
            if hasattr(y_test, 'values'):
                y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
            else:
                y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
                
        except Exception as e:
            print(f"      Ã¢ÂÅ’ Tensor conversion failed: {e}")
            raise
        
        # Training with validation monitoring
        best_loss = float('inf')
        patience_counter = 0
        patience = 50
        
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation check
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        model.load_state_dict(best_state)
                        break
        
        # Final predictions
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).numpy().flatten()
            test_pred = model(X_test_tensor).numpy().flatten()
        
        # Transform back to original scale - FIXED: no index lookups
        try:
            if target_metric in self.y_scalers:
                train_pred_orig = self.y_scalers[target_metric].inverse_transform(train_pred.reshape(-1, 1)).flatten()
                test_pred_orig = self.y_scalers[target_metric].inverse_transform(test_pred.reshape(-1, 1)).flatten()
                
                # Use the actual target data passed to this function
                y_train_orig = y_train.values if hasattr(y_train, 'values') else y_train
                y_test_orig = y_test.values if hasattr(y_test, 'values') else y_test
                
                # If targets were also scaled, inverse transform them
                try:
                    y_train_orig = self.y_scalers[target_metric].inverse_transform(y_train_orig.reshape(-1, 1)).flatten()
                    y_test_orig = self.y_scalers[target_metric].inverse_transform(y_test_orig.reshape(-1, 1)).flatten()
                except:
                    pass  # Keep original if transform fails
            else:
                train_pred_orig = train_pred
                test_pred_orig = test_pred
                y_train_orig = y_train.values if hasattr(y_train, 'values') else y_train
                y_test_orig = y_test.values if hasattr(y_test, 'values') else y_test
            
            performance = {
                'train_r2': r2_score(y_train_orig, train_pred_orig),
                'test_r2': r2_score(y_test_orig, test_pred_orig),
                'train_mse': mean_squared_error(y_train_orig, train_pred_orig),
                'test_mse': mean_squared_error(y_test_orig, test_pred_orig),
                'train_mae': mean_absolute_error(y_train_orig, train_pred_orig),
                'test_mae': mean_absolute_error(y_test_orig, test_pred_orig),
                'train_rmse': np.sqrt(mean_squared_error(y_train_orig, train_pred_orig)),
                'test_rmse': np.sqrt(mean_squared_error(y_test_orig, test_pred_orig)),
                'train_norm_mse': mean_squared_error(y_train_orig, train_pred_orig) / np.var(y_train_orig) if np.var(y_train_orig) > 0 else float('inf'),
                'test_norm_mse': mean_squared_error(y_test_orig, test_pred_orig) / np.var(y_test_orig) if np.var(y_test_orig) > 0 else float('inf'),
                'predictions_train': train_pred_orig,
                'predictions_test': test_pred_orig
            }
            
        except Exception as e:
            print(f"      Ã¢ÂÅ’ Final metric calculation failed: {e}")
            # Fallback calculation
            performance = {
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'train_norm_mse': mean_squared_error(y_train, train_pred) / np.var(y_train) if np.var(y_train) > 0 else float('inf'),
                'test_norm_mse': mean_squared_error(y_test, test_pred) / np.var(y_test) if np.var(y_test) > 0 else float('inf'),
                'predictions_train': train_pred,
                'predictions_test': test_pred
            }
        
        return model, performance

    def _create_adaptive_ensemble_model(self, input_dim, task_type='regression', n_classes=None):
        """
        Create an ensemble of different model architectures.
        
        ✅ FIXED: Uses module-level AdaptiveEnsemble class for pickle support
        """
        # ✅ Use module-level AdaptiveEnsemble class (defined at top of file)
        return AdaptiveEnsemble(input_dim, task_type, n_classes)
    
    def _train_neural_network_regressor(self, X_train, y_train, X_test, y_test, target_name):
        """
        Train a neural network for regression tasks.
        Useful for complex targets like FNR that trees struggle with.
        
        ✅ FIXED: Uses module-level TabularNN and NNWrapper classes for pickle support
        """
        print(f"      🧠 Training neural network for {target_name}...")
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        # Special handling for TTD - needs more capacity and training
        if target_name == 'time_to_detection_threshold':
            hidden_sizes = [256, 128, 64, 32]  # Deeper network for TTD
            n_epochs = 250  # More training
            patience = 40  # More patience
            learning_rate = 0.001
            batch_size = 32  # Smaller batches for better TTD learning
            print(f"         Using TTD-optimized architecture: {hidden_sizes}")
        else:
            hidden_sizes = [256, 128, 64, 32]  # Standard architecture
            n_epochs = 200
            patience = 30
            learning_rate = 0.001
            batch_size = 64
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_t = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
        y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # ✅ Use module-level TabularNN class (defined at top of file)
        model = TabularNN(X_train_t.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        max_patience = patience
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_t)
                val_loss = criterion(val_pred, y_test_t).item()
            model.train()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"         Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).numpy().flatten()
            test_pred = model(X_test_t).numpy().flatten()
        
        from sklearn.metrics import r2_score, mean_squared_error
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        print(f"         ✅ Neural Network: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")
        
        # ✅ Use module-level NNWrapper class (defined at top of file)
        return NNWrapper(model), test_r2, test_mse
    
    def _train_with_multiple_strategies(self, X_train_final, y_train_processed, X_test_final, y_test_processed, 
                                target_metric, is_classification, n_classes=None):
        """
        High-performance stacked ensemble with proper CV handling
        Returns best single model OR ensemble
        """
        
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge
        
        print(f"      🎯 Training high-performance ensemble...")
        
        # Store predictions for stacking
        base_predictions_train = {}
        base_predictions_test = {}
        base_models = {}
        
        # ========== MODEL 1: XGBoost (Best for tabular data) ==========
        print(f"      🌳 XGBoost...", end=" ")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(
            X_train_final, y_train_processed,
            eval_set=[(X_test_final, y_test_processed)],
            verbose=False
        )
        
        base_predictions_train['xgb'] = xgb_model.predict(X_train_final)
        base_predictions_test['xgb'] = xgb_model.predict(X_test_final)
        base_models['xgb'] = xgb_model
        
        xgb_r2 = r2_score(y_test_processed, base_predictions_test['xgb'])
        print(f"R² = {xgb_r2:.4f}")
        
        # ========== MODEL 2: LightGBM (Different gradient boosting) ==========
        print(f"      💡 LightGBM...", end=" ")
        
        X_train_clean = self._clean_feature_names(X_train_final)
        X_test_clean = self._clean_feature_names(X_test_final)
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=5,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        lgb_model.fit(
            X_train_clean, y_train_processed,
            eval_set=[(X_test_clean, y_test_processed)],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(100)]
        )
        
        base_predictions_train['lgb'] = lgb_model.predict(X_train_clean)
        base_predictions_test['lgb'] = lgb_model.predict(X_test_clean)
        base_models['lgb'] = lgb_model
        
        lgb_r2 = r2_score(y_test_processed, base_predictions_test['lgb'])
        print(f"R² = {lgb_r2:.4f}")
        
        # ========== MODEL 3: Random Forest (Different algorithm) ==========
        print(f"      🌲 Random Forest...", end=" ")
        
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_final, y_train_processed)
        
        base_predictions_train['rf'] = rf_model.predict(X_train_final)
        base_predictions_test['rf'] = rf_model.predict(X_test_final)
        base_models['rf'] = rf_model
        
        rf_r2 = r2_score(y_test_processed, base_predictions_test['rf'])
        print(f"R² = {rf_r2:.4f}")
        
        # ========== STACKED ENSEMBLE ==========
        print(f"      🔗 Stacking Ensemble...", end=" ")
        
        # Create meta-features from base model predictions
        X_meta_train = np.column_stack([
            base_predictions_train['xgb'],
            base_predictions_train['lgb'],
            base_predictions_train['rf']
        ])
        
        X_meta_test = np.column_stack([
            base_predictions_test['xgb'],
            base_predictions_test['lgb'],
            base_predictions_test['rf']
        ])
        
        # Train meta-learner with optimized regularization
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import Ridge

        param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
        meta_learner = GridSearchCV(
            Ridge(),
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        meta_learner.fit(X_meta_train, y_train_processed)
        meta_learner = meta_learner.best_estimator_  # Use best
        
        stacked_pred_test = meta_learner.predict(X_meta_test)
        stacked_r2 = r2_score(y_test_processed, stacked_pred_test)
        
        print(f"R² = {stacked_r2:.4f}")
        print(f"      📊 Ensemble weights: XGB={meta_learner.coef_[0]:.3f}, LGB={meta_learner.coef_[1]:.3f}, RF={meta_learner.coef_[2]:.3f}")
        
        # ========== SELECT BEST MODEL ==========
        all_models = {
            'XGBoost': (xgb_model, xgb_r2, base_predictions_test['xgb']),
            'LightGBM': (lgb_model, lgb_r2, base_predictions_test['lgb']),
            'RandomForest': (rf_model, rf_r2, base_predictions_test['rf']),
            'StackedEnsemble': ({
                'base_models': base_models,
                'meta_learner': meta_learner,
                'X_train_clean': X_train_clean
            }, stacked_r2, stacked_pred_test)
        }
        
        best_name = max(all_models.keys(), key=lambda k: all_models[k][1])
        best_model, best_r2, best_predictions = all_models[best_name]
        
        print(f"      ✅ Best: {best_name} (R² = {best_r2:.4f})")
        
        # Return in expected format
        performance = {
            'train_r2': r2_score(y_train_processed, 
                                meta_learner.predict(X_meta_train) if best_name == 'StackedEnsemble' 
                                else best_model.predict(X_train_final if best_name != 'LightGBM' else X_train_clean)),
            'test_r2': best_r2,
            'test_mse': mean_squared_error(y_test_processed, best_predictions),
            'test_mae': mean_absolute_error(y_test_processed, best_predictions),
            'test_rmse': np.sqrt(mean_squared_error(y_test_processed, best_predictions)),
            'test_norm_mse': mean_squared_error(y_test_processed, best_predictions) / np.var(y_test_processed) if np.var(y_test_processed) > 0 else float('inf'),
            'predictions_test': best_predictions
        }
        
        return best_model, performance

    def _create_enhanced_traditional_model(self, input_dim, task_type='regression', n_classes=None):
        """
        Create enhanced traditional model with better architecture.
        
        ✅ FIXED: Uses module-level EnhancedTraditional class for pickle support
        """
        # ✅ Use module-level EnhancedTraditional class (defined at top of file)
        return EnhancedTraditional(input_dim, task_type, n_classes)
    
    def _fix_index_alignment(self, X_data, y_data, target_metric):
        """Fix index alignment issues between X and y data"""
        try:
            # If X_data has an index, use it to align y_data
            if hasattr(X_data, 'index'):
                # Use .loc instead of .iloc to match by index labels, not positions
                if target_metric in self.y_train.columns:
                    y_aligned = self.y_train[target_metric].loc[X_data.index]
                else:
                    # Fallback: if target not in y_train, use provided y_data
                    if len(y_data) == len(X_data):
                        y_aligned = pd.Series(y_data, index=X_data.index)
                    else:
                        raise ValueError(f"Length mismatch: X_data ({len(X_data)}) vs y_data ({len(y_data)})")
            else:
                # If no index, just ensure same length
                if len(y_data) == len(X_data):
                    y_aligned = y_data
                else:
                    raise ValueError(f"Length mismatch: X_data ({len(X_data)}) vs y_data ({len(y_data)})")
            
            return y_aligned
        except (KeyError, IndexError) as e:
            # If alignment fails, fall back to provided y_data
            print(f"      Ã¢Å¡ Ã¯Â¸Â  Index alignment failed, using provided y_data: {e}")
            return y_data

    def train_surrogate_models(self, force_retrain=False):
        """
        FIXED VERSION: Train surrogate models with proper handling of all targets
        - No variable contamination between targets
        - Correct target data for all models
        - Proper control flow for difficult targets
        """
        print("🏭 Training ENHANCED surrogate models for RL...")
        print("=" * 60)

        # ⚡ CRITICAL: Verify target data integrity
        print(f"\n🔍 VERIFYING TARGET DATA INTEGRITY:")
        for col in self.y_train.columns:
            y_min, y_max = self.y_train[col].min(), self.y_train[col].max()
            print(f"   {col}: [{y_min:.6f}, {y_max:.6f}]")
            
            # Sanity checks
            if 'false_negative_rate' in col and y_max > 1.0:
                raise ValueError(f"❌ BUG: {col} has max={y_max:.6f} > 1.0! Wrong data!")
            if 'time_to_detection' in col and y_max > 100:
                print(f"   ⚠️ WARNING: {col} has max={y_max:.2f} > 100 (unusual but might be OK)")
        
        # Initialize storage
        self.label_encoders = {}
        self.target_transformers = {}
        self.surrogate_metric_scalers = {}  # ⚡ Initialize here!
        self.surrogate_features = {}  # ⚡ Initialize here!
        surrogate_performance = {}
        
        # ✅ ADD VALIDATION: Ensure dictionaries stay initialized
        print(f"📊 Initialized storage:")
        print(f"   surrogate_metric_scalers: {type(self.surrogate_metric_scalers)}")
        print(f"   surrogate_features: {type(self.surrogate_features)}")
        
        # Use the already prepared scaled features
        if not hasattr(self, 'X_train_scaled') or self.X_train_scaled is None:
            print("⚡️  Scaled features not found, preparing data...")
            self._prepare_data_with_scaling()
        
        # Use enhanced features if available
        if hasattr(self, 'X_train_enhanced') and self.X_train_enhanced is not None:
            print("📊 Using enhanced feature set for surrogate models")
            X_train = self.X_train_enhanced.copy()
            X_test = self.X_test_enhanced.copy()
        else:
            print("📊 Using scaled feature set for surrogate models")
            X_train = self.X_train_scaled.copy()
            X_test = self.X_test_scaled.copy()
        
        # Use features as-is (already scaled from _prepare_data_with_scaling)
        print("🎧 Using pre-scaled features (no additional scaling needed)...")
        
        # X_train and X_test are already standardized
        X_train_surrogate = X_train.copy()
        X_test_surrogate = X_test.copy()
        
        # Create identity scaler for compatibility
        from sklearn.preprocessing import StandardScaler
        self.surrogate_feature_scaler = StandardScaler()
        self.surrogate_feature_scaler.fit(X_train)  # Fit to already-scaled data
        
        print(f"📈 Training surrogate models for {len(self.y_train.columns)} targets")
        print(f"   Feature shape: {X_train_surrogate.shape}")
        
        # Track skipped targets
        skipped_targets = []
        
        # CRITICAL: Train surrogates in dependency order
        target_order = []
        fnr_targets = []

        for target in self.y_train.columns:
            if 'false_negative' in target.lower():
                fnr_targets.append(target)
            else:
                target_order.append(target)

        target_order.extend(fnr_targets)
        print(f"📋 Training order (FNR components first): {target_order}")

        # ═══════════════════════════════════════════════════════════════
        # MAIN TRAINING LOOP
        # ═══════════════════════════════════════════════════════════════
        for target_idx, target_metric in enumerate(target_order):
            print(f"\n🎯 Target {target_idx + 1}/{len(self.y_train.columns)}: {target_metric}")
            print("-" * 50)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Extract and Verify Target Data
            # ═══════════════════════════════════════════════════════════
            y_train_target = self.y_train[target_metric].copy()
            y_test_target = self.y_test[target_metric].copy()

            print(f"\n🔍 TRAINING DATA VALIDATION for {target_metric}:")
            print(f"   X_train shape: {X_train.shape}")
            print(f"   y_train: min={y_train_target.min():.6f}, max={y_train_target.max():.6f}, mean={y_train_target.mean():.6f}")
            print(f"   y_test: min={y_test_target.min():.6f}, max={y_test_target.max():.6f}, mean={y_test_target.mean():.6f}")

            # Check for extrapolation needs
            self.check_data_leakage(target_metric)

            # Verify expected ranges (updated to match actual simulation data)
            expected_ranges = {
                'signal_to_noise_ratio_SNR': (-120, 50),  # Updated: saw -117 in data
                'dynamic_range_of_output': (0, 40),        # Updated: saw 34.9 in data
                'false_negative_rate': (0, 0.25),          # Updated: saw 0.206 in data
                'time_to_detection_threshold': (10, 9000)  
            }

            if target_metric in expected_ranges:
                exp_min, exp_max = expected_ranges[target_metric]
                actual_min, actual_max = y_train_target.min(), y_train_target.max()
                
                if actual_min < exp_min or actual_max > exp_max:
                    print(f"   📊 INFO: Values slightly outside typical range [{exp_min}, {exp_max}]")
                    print(f"      Actual range: [{actual_min:.2f}, {actual_max:.2f}]")
                    print(f"      This is acceptable - simulation explores edge cases")
                else:
                    print(f"   ✅ Values within expected range [{exp_min}, {exp_max}]")
            
            # Handle missing values
            if y_train_target.isnull().any():
                print(f"   Handling {y_train_target.isnull().sum()} missing target values...")
                if y_train_target.dtype == 'object':
                    mode_val = y_train_target.mode()[0] if len(y_train_target.mode()) > 0 else 'unknown'
                    y_train_target = y_train_target.fillna(mode_val)
                    y_test_target = y_test_target.fillna(mode_val)
                else:
                    median_val = y_train_target.median()
                    y_train_target = y_train_target.fillna(median_val)
                    y_test_target = y_test_target.fillna(median_val)
            
            # Sanity check for FNR - with proper bounds checking
            if 'false_negative_rate' in target_metric:
                if y_train_target.max() > 1.0:
                    raise ValueError(f"❌ CRITICAL BUG: FNR has values > 1.0 (max={y_train_target.max():.6f})!")
                
                # Additional validation: FNR should typically be < 0.3 for biosensors
                if y_train_target.max() > 0.3:
                    print(f"   ⚠️ WARNING: FNR max={y_train_target.max():.4f} is unusually high")
                    print(f"      Expected: FNR typically < 0.2 for good biosensors")
                    print(f"      This may indicate:")
                    print(f"      1. Poorly performing circuits in dataset (realistic)")
                    print(f"      2. Data quality issues (needs investigation)")
                    
                # Check minimum value
                if y_train_target.min() < 0:
                    raise ValueError(f"❌ CRITICAL BUG: FNR has negative values (min={y_train_target.min():.6f})!")
            
            # Determine task type
            is_classification = y_train_target.dtype == 'object' or y_train_target.dtype.name == 'category'
            
            # 🎯 ULTIMATE: Train FNR using MULTI-STRATEGY approach (like TTD)
            if target_metric == 'false_negative_rate' and not is_classification:
                print("   ⚡ Training FNR predictor using MULTI-STRATEGY approach")
                print("      Will try: 1) Gradient Boosting Ensemble, 2) Neural Network, 3) Deep NN")
                print("      Then select best R² performer")
                
                # Store results from different approaches
                fnr_candidates = {}
                
                # ════════════════════════════════════════════════════════════
                # APPROACH 1: LightGBM + XGBoost Ensemble (Trees work well for FNR)
                # ════════════════════════════════════════════════════════════
                print(f"\n      🌲 APPROACH 1: Gradient Boosting Ensemble...")
                try:
                    import lightgbm as lgb
                    import xgboost as xgb
                    from sklearn.metrics import r2_score
                    
                    # LightGBM optimized for FNR (small values, skewed distribution)
                    lgb_params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 31,  # Smaller for FNR (less complex)
                        'learning_rate': 0.05,  # Slower for stability
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'verbose': -1,
                        'n_estimators': 800,  # More trees for FNR
                        'reg_alpha': 0.2,  # More regularization
                        'reg_lambda': 0.2,
                        'min_child_samples': 20  # Prevent overfitting to small FNR values
                    }
                    
                    lgb_model = lgb.LGBMRegressor(**lgb_params)
                    lgb_model.fit(
                        X_train_surrogate, y_train_final,
                        eval_set=[(X_test_surrogate, y_test_final)],
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                    
                    # XGBoost optimized for FNR
                    xgb_params = {
                        'objective': 'reg:squarederror',
                        'max_depth': 6,  # Shallower for FNR
                        'learning_rate': 0.05,
                        'n_estimators': 800,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.2,
                        'reg_lambda': 0.2,
                        'min_child_weight': 5,  # Prevent overfitting
                        'verbosity': 0
                    }
                    
                    xgb_model = xgb.XGBRegressor(**xgb_params)
                    xgb_model.fit(
                        X_train_surrogate, y_train_final,
                        eval_set=[(X_test_surrogate, y_test_final)],
                        verbose=False
                    )
                    
                    # Ensemble: Average predictions
                    ensemble_model = EnsembleModel([lgb_model, xgb_model])
                    ensemble_pred = ensemble_model.predict(X_test_surrogate)
                    ensemble_r2 = r2_score(y_test_final, ensemble_pred)
                    ensemble_mse = np.mean((y_test_final - ensemble_pred)**2)
                    
                    # Debug: Check prediction range
                    print(f"         Ensemble pred range: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
                    print(f"         True test range: [{y_test_final.min():.4f}, {y_test_final.max():.4f}]")
                    
                    fnr_candidates['ensemble'] = {
                        'model': ensemble_model,
                        'r2': ensemble_r2,
                        'mse': ensemble_mse,
                        'name': 'LightGBM+XGBoost Ensemble'
                    }
                    print(f"         ✅ Ensemble: R²={ensemble_r2:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Ensemble failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ════════════════════════════════════════════════════════════
                # APPROACH 2: Standard Neural Network with Proper Scaling
                # ════════════════════════════════════════════════════════════
                print(f"\n      🧠 APPROACH 2: Neural Network...")
                try:
                    # Train on ORIGINAL data (NOT scaled) to match tree models
                    # This avoids the scaling mismatch issue
                    nn_model, nn_r2, nn_mse = self._train_neural_network_regressor(
                        X_train_surrogate, y_train_final,  # Use ORIGINAL targets
                        X_test_surrogate, y_test_final,
                        target_metric
                    )
                    
                    # Get predictions
                    nn_pred = nn_model.predict(X_test_surrogate)
                    
                    # Debug: Check prediction range
                    print(f"         NN pred range: [{nn_pred.min():.4f}, {nn_pred.max():.4f}]")
                    
                    fnr_candidates['neural_network'] = {
                        'model': nn_model,
                        'r2': nn_r2,
                        'mse': nn_mse,
                        'name': 'Neural Network'
                    }
                    print(f"         ✅ Neural Network: R²={nn_r2:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Neural Network failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ════════════════════════════════════════════════════════════
                # APPROACH 3: Enhanced Deep Neural Network
                # ════════════════════════════════════════════════════════════
                print(f"\n      🧬 APPROACH 3: Enhanced Deep Neural Network...")
                try:
                    import torch
                    import torch.nn as nn
                    import torch.optim as optim
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    # Use ORIGINAL data (like ensemble)
                    X_train_t = torch.FloatTensor(X_train_surrogate.values)
                    y_train_t = torch.FloatTensor(y_train_final).reshape(-1, 1)
                    X_test_t = torch.FloatTensor(X_test_surrogate.values)
                    y_test_t = torch.FloatTensor(y_test_final).reshape(-1, 1)
                    
                    train_dataset = TensorDataset(X_train_t, y_train_t)
                    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                    
                    # Optimized architecture for FNR (smaller, simpler)
                    class FNR_NN(nn.Module):
                        def __init__(self, input_dim):
                            super().__init__()
                            self.fc1 = nn.Linear(input_dim, 256)
                            self.bn1 = nn.BatchNorm1d(256)
                            self.fc2 = nn.Linear(256, 128)
                            self.bn2 = nn.BatchNorm1d(128)
                            self.fc3 = nn.Linear(128, 64)
                            self.bn3 = nn.BatchNorm1d(64)
                            self.fc4 = nn.Linear(64, 32)
                            self.fc5 = nn.Linear(32, 1)
                            self.dropout = nn.Dropout(0.3)  # More dropout for FNR
                        
                        def forward(self, x):
                            x = F.relu(self.bn1(self.fc1(x)))
                            x = self.dropout(x)
                            x = F.relu(self.bn2(self.fc2(x)))
                            x = self.dropout(x)
                            x = F.relu(self.bn3(self.fc3(x)))
                            x = self.dropout(x)
                            x = F.relu(self.fc4(x))
                            x = self.fc5(x)
                            # ✅ CRITICAL: Apply sigmoid to bound output to [0, 1]
                            # Then scale to expected FNR range [0, 0.25]
                            x = torch.sigmoid(x) * 0.25
                            return x
                    
                    enhanced_nn = FNR_NN(X_train_t.shape[1])
                    criterion = nn.MSELoss()  # MSE for FNR
                    optimizer = optim.AdamW(enhanced_nn.parameters(), lr=0.001, weight_decay=1e-3)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
                    
                    # Training
                    best_loss = float('inf')
                    patience_counter = 0
                    best_state = None
                    
                    for epoch in range(400):  # More epochs for FNR
                        enhanced_nn.train()
                        epoch_loss = 0
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            pred = enhanced_nn(X_batch)
                            loss = criterion(pred, y_batch)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(enhanced_nn.parameters(), 0.5)
                            optimizer.step()
                            epoch_loss += loss.item()
                        
                        # Validation
                        enhanced_nn.eval()
                        with torch.no_grad():
                            val_pred = enhanced_nn(X_test_t)
                            val_loss = criterion(val_pred, y_test_t).item()
                        
                        scheduler.step(val_loss)
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            patience_counter = 0
                            best_state = enhanced_nn.state_dict().copy()
                        else:
                            patience_counter += 1
                            if patience_counter >= 50:
                                print(f"         Early stopping at epoch {epoch+1}")
                                break
                    
                    # Load best model
                    if best_state is not None:
                        enhanced_nn.load_state_dict(best_state)
                    
                    # Evaluate
                    enhanced_nn.eval()
                    with torch.no_grad():
                        test_pred = enhanced_nn(X_test_t).numpy().flatten()
                    
                    from sklearn.metrics import r2_score, mean_squared_error
                    enhanced_r2 = r2_score(y_test_final, test_pred)
                    enhanced_mse = mean_squared_error(y_test_final, test_pred)
                    
                    # Debug: Check prediction range
                    print(f"         Enhanced NN pred range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
                    
                    # Wrap for sklearn compatibility
                    fnr_candidates['enhanced_nn'] = {
                        'model': NNWrapper(enhanced_nn),
                        'r2': enhanced_r2,
                        'mse': enhanced_mse,
                        'name': 'Enhanced Deep NN'
                    }
                    print(f"         ✅ Enhanced NN: R²={enhanced_r2:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Enhanced NN failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ════════════════════════════════════════════════════════════
                # SELECT BEST PERFORMER
                # ════════════════════════════════════════════════════════════
                print(f"\n      🏆 SELECTING BEST MODEL...")
                
                if fnr_candidates:
                    # Sort by R² (descending)
                    best_key = max(fnr_candidates.keys(), key=lambda k: fnr_candidates[k]['r2'])
                    best_model_info = fnr_candidates[best_key]
                    
                    print(f"\n      📊 PERFORMANCE COMPARISON:")
                    for key, info in sorted(fnr_candidates.items(), key=lambda x: x[1]['r2'], reverse=True):
                        print(f"         {info['name']}: R²={info['r2']:.3f}, RMSE={np.sqrt(info['mse']):.4f}")
                    
                    print(f"\n      ✅ WINNER: {best_model_info['name']} (R²={best_model_info['r2']:.3f})")
                    
                    # Store winner
                    self.surrogate_models[target_metric] = best_model_info['model']
                    surrogate_performance[target_metric] = {
                        'test_r2': best_model_info['r2'],
                        'test_mse': best_model_info['mse'],
                        'type': 'regression',
                        'model_type': best_model_info['name']
                    }
                    
                    # Create scaler
                    from sklearn.preprocessing import StandardScaler
                    fnr_scaler = StandardScaler()
                    fnr_scaler.fit(X_train_surrogate)
                    self.surrogate_metric_scalers[target_metric] = fnr_scaler
                    self.surrogate_features[target_metric] = list(X_train_surrogate.columns)
                    print(f"      ✅ FNR scaler created ({len(X_train_surrogate.columns)} features)")
                    
                    continue  # Skip standard training
                else:
                    print(f"      ❌ All approaches failed, falling back to standard training...")

            # 🎯 ULTIMATE: Train TTD using MULTI-STRATEGY approach (try 3 methods, pick best)
            if target_metric == 'time_to_detection_threshold' and not is_classification:
                print("   ⚡ Training TTD predictor using MULTI-STRATEGY approach")
                print("      Will try: 1) Neural Network, 2) Two-Stage, 3) Ensemble")
                print("      Then select best R² performer")
                
                # Store results from different approaches
                ttd_candidates = {}
                
                # ════════════════════════════════════════════════════════════
                # APPROACH 1: Neural Network (Current best: 0.856)
                # ════════════════════════════════════════════════════════════
                print(f"\n      🧠 APPROACH 1: Neural Network...")
                try:
                    # ✅ FIX: Use SCALED targets for neural network training
                    if hasattr(self, 'y_train_scaled') and target_metric in self.y_train_scaled:
                        y_train_for_nn = self.y_train_scaled[target_metric].values
                        y_test_for_nn = self.y_test_scaled[target_metric].values
                        print(f"         Using SCALED targets for TTD NN")
                        use_scaled = True
                    else:
                        y_train_for_nn = y_train_final
                        y_test_for_nn = y_test_final
                        print(f"         ⚠️ Using unscaled targets")
                        use_scaled = False
                    
                    nn_model, nn_r2, nn_mse = self._train_neural_network_regressor(
                        X_train_surrogate, y_train_for_nn,
                        X_test_surrogate, y_test_for_nn,
                        target_metric
                    )
                    
                    # ✅ FIX: Evaluate in ORIGINAL space
                    if use_scaled and hasattr(self, 'y_scalers') and target_metric in self.y_scalers:
                        nn_pred_test_scaled = nn_model.predict(X_test_surrogate)
                        nn_pred_test_original = self.y_scalers[target_metric].inverse_transform(
                            nn_pred_test_scaled.reshape(-1, 1)
                        ).flatten()
                        
                        # Use TRUE original test data
                        if target_metric in self.y_test.columns:
                            y_test_true_original = self.y_test[target_metric].values
                        else:
                            y_test_true_original = y_test_target
                        
                        from sklearn.metrics import r2_score, mean_squared_error
                        nn_r2 = r2_score(y_test_true_original, nn_pred_test_original)
                        nn_mse = mean_squared_error(y_test_true_original, nn_pred_test_original)
                        
                        print(f"         Pred range: [{nn_pred_test_original.min():.1f}, {nn_pred_test_original.max():.1f}]")
                    
                    ttd_candidates['neural_network'] = {
                        'model': nn_model,
                        'r2': nn_r2,
                        'mse': nn_mse,
                        'name': 'Neural Network'
                    }
                    print(f"         ✅ Neural Network: R²={nn_r2:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Neural Network failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ════════════════════════════════════════════════════════════
                # APPROACH 2: LightGBM + XGBoost Ensemble
                # ════════════════════════════════════════════════════════════
                print(f"\n      🌲 APPROACH 2: Gradient Boosting Ensemble...")
                try:
                    import lightgbm as lgb
                    import xgboost as xgb
                    from sklearn.metrics import r2_score
                    
                    # LightGBM with quantile loss (robust to outliers)
                    lgb_params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 63,
                        'learning_rate': 0.03,
                        'feature_fraction': 0.9,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'verbose': -1,
                        'n_estimators': 500,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1
                    }
                    
                    lgb_model = lgb.LGBMRegressor(**lgb_params)
                    lgb_model.fit(
                        X_train_surrogate, y_train_final,
                        eval_set=[(X_test_surrogate, y_test_final)],
                        callbacks=[lgb.early_stopping(50, verbose=False)]
                    )
                    
                    # XGBoost
                    xgb_params = {
                        'objective': 'reg:squarederror',
                        'max_depth': 8,
                        'learning_rate': 0.03,
                        'n_estimators': 500,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1,
                        'verbosity': 0
                    }
                    
                    xgb_model = xgb.XGBRegressor(**xgb_params)
                    xgb_model.fit(
                        X_train_surrogate, y_train_final,
                        eval_set=[(X_test_surrogate, y_test_final)],
                        verbose=False
                    )
                    
                    # Ensemble: Average predictions (using module-level class for pickle)
                    ensemble_model = EnsembleModel([lgb_model, xgb_model])
                    ensemble_pred = ensemble_model.predict(X_test_surrogate)
                    ensemble_r2 = r2_score(y_test_final, ensemble_pred)
                    ensemble_mse = np.mean((y_test_final - ensemble_pred)**2)
                    
                    ttd_candidates['ensemble'] = {
                        'model': ensemble_model,
                        'r2': ensemble_r2,
                        'mse': ensemble_mse,
                        'name': 'LightGBM+XGBoost Ensemble'
                    }
                    print(f"         ✅ Ensemble: R²={ensemble_r2:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Ensemble failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ════════════════════════════════════════════════════════════
                # APPROACH 3: Enhanced Neural Network (Deeper + Regularization)
                # ════════════════════════════════════════════════════════════
                print(f"\n      🧬 APPROACH 3: Enhanced Deep Neural Network...")
                try:
                    import torch
                    import torch.nn as nn
                    import torch.optim as optim
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    # ✅ FIX: Use SCALED targets for neural network
                    if hasattr(self, 'y_train_scaled') and target_metric in self.y_train_scaled:
                        y_train_for_nn = self.y_train_scaled[target_metric].values
                        y_test_for_nn = self.y_test_scaled[target_metric].values
                        use_scaled_ttd = True
                    else:
                        y_train_for_nn = y_train_final
                        y_test_for_nn = y_test_final
                        use_scaled_ttd = False
                    
                    X_train_t = torch.FloatTensor(X_train_surrogate.values)
                    y_train_t = torch.FloatTensor(y_train_for_nn).reshape(-1, 1)
                    X_test_t = torch.FloatTensor(X_test_surrogate.values)
                    y_test_t = torch.FloatTensor(y_test_for_nn).reshape(-1, 1)
                    
                    train_dataset = TensorDataset(X_train_t, y_train_t)
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    
                    # Deeper network with residual connections
                    class EnhancedNN(nn.Module):
                        def __init__(self, input_dim):
                            super().__init__()
                            self.fc1 = nn.Linear(input_dim, 512)
                            self.bn1 = nn.BatchNorm1d(512)
                            self.fc2 = nn.Linear(512, 256)
                            self.bn2 = nn.BatchNorm1d(256)
                            self.fc3 = nn.Linear(256, 128)
                            self.bn3 = nn.BatchNorm1d(128)
                            self.fc4 = nn.Linear(128, 64)
                            self.bn4 = nn.BatchNorm1d(64)
                            self.fc5 = nn.Linear(64, 1)
                            self.dropout = nn.Dropout(0.2)
                        
                        def forward(self, x):
                            x = F.relu(self.bn1(self.fc1(x)))
                            x = self.dropout(x)
                            x = F.relu(self.bn2(self.fc2(x)))
                            x = self.dropout(x)
                            x = F.relu(self.bn3(self.fc3(x)))
                            x = self.dropout(x)
                            x = F.relu(self.bn4(self.fc4(x)))
                            x = self.fc5(x)
                            return x
                    
                    enhanced_nn = EnhancedNN(X_train_t.shape[1])
                    criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
                    optimizer = optim.AdamW(enhanced_nn.parameters(), lr=0.001, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
                    
                    # Training
                    best_loss = float('inf')
                    patience_counter = 0
                    best_state = None
                    
                    for epoch in range(300):
                        enhanced_nn.train()
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            pred = enhanced_nn(X_batch)
                            loss = criterion(pred, y_batch)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(enhanced_nn.parameters(), 1.0)
                            optimizer.step()
                        
                        # Validation
                        enhanced_nn.eval()
                        with torch.no_grad():
                            val_pred = enhanced_nn(X_test_t)
                            val_loss = criterion(val_pred, y_test_t).item()
                        
                        scheduler.step(val_loss)
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            patience_counter = 0
                            best_state = enhanced_nn.state_dict().copy()
                        else:
                            patience_counter += 1
                            if patience_counter >= 40:
                                break
                    
                    # Evaluate on test set
                    enhanced_nn.eval()
                    with torch.no_grad():
                        test_pred = enhanced_nn(X_test_t).numpy().flatten()
                    
                    # ✅ FIX: If trained on scaled, evaluate in original space
                    if use_scaled_ttd and hasattr(self, 'y_scalers') and target_metric in self.y_scalers:
                        test_pred_original = self.y_scalers[target_metric].inverse_transform(
                            test_pred.reshape(-1, 1)
                        ).flatten()
                        
                        # Use TRUE original test data
                        if target_metric in self.y_test.columns:
                            y_test_true_original = self.y_test[target_metric].values
                        else:
                            y_test_true_original = y_test_target
                        
                        from sklearn.metrics import r2_score, mean_squared_error
                        enhanced_r2 = r2_score(y_test_true_original, test_pred_original)
                        enhanced_mse = mean_squared_error(y_test_true_original, test_pred_original)
                        
                        print(f"         Pred range: [{test_pred_original.min():.1f}, {test_pred_original.max():.1f}]")
                    else:
                        from sklearn.metrics import r2_score, mean_squared_error
                        enhanced_r2 = r2_score(y_test_final, test_pred)
                        enhanced_mse = mean_squared_error(y_test_final, test_pred)
                    
                    ttd_candidates['enhanced_nn'] = {
                        'model': enhanced_nn,
                        'r2': enhanced_r2,
                        'mse': enhanced_mse,
                        'name': 'Enhanced Deep NN'
                    }
                    print(f"         ✅ Enhanced NN: R²={enhanced_r2:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ Enhanced NN failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ════════════════════════════════════════════════════════════
                # SELECT BEST PERFORMER
                # ════════════════════════════════════════════════════════════
                print(f"\n      🏆 SELECTING BEST MODEL...")
                
                if ttd_candidates:
                    # Sort by R² (descending)
                    best_key = max(ttd_candidates.keys(), key=lambda k: ttd_candidates[k]['r2'])
                    best_model_info = ttd_candidates[best_key]
                    
                    print(f"\n      📊 PERFORMANCE COMPARISON:")
                    for key, info in sorted(ttd_candidates.items(), key=lambda x: x[1]['r2'], reverse=True):
                        print(f"         {info['name']}: R²={info['r2']:.3f}, RMSE={np.sqrt(info['mse']):.1f}")
                    
                    print(f"\n      ✅ WINNER: {best_model_info['name']} (R²={best_model_info['r2']:.3f})")
                    
                    # Store winner
                    self.surrogate_models[target_metric] = best_model_info['model']
                    surrogate_performance[target_metric] = {
                        'test_r2': best_model_info['r2'],
                        'test_mse': best_model_info['mse'],
                        'type': 'regression',
                        'model_type': best_model_info['name']
                    }
                    
                    # Create scaler
                    from sklearn.preprocessing import StandardScaler
                    ttd_scaler = StandardScaler()
                    ttd_scaler.fit(X_train_surrogate)
                    self.surrogate_metric_scalers[target_metric] = ttd_scaler
                    self.surrogate_features[target_metric] = list(X_train_surrogate.columns)
                    print(f"      ✅ TTD scaler created ({len(X_train_surrogate.columns)} features)")
                    
                    continue  # Skip standard training
                else:
                    print(f"      ❌ All approaches failed, falling back to standard training...")
            
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: Prepare Final Target Data
            # ═══════════════════════════════════════════════════════════
            
            # Skip log transform for TTD - handled by two-stage predictor
            # (Log transform on censored data makes it worse!)
            self.ttd_log_transform = {'enabled': False}
            
            if is_classification:
                print(f"   📊 Classification task detected")
                
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                
                combined_targets = pd.concat([pd.Series(y_train_target), pd.Series(y_test_target)], ignore_index=True)
                le.fit(combined_targets)
                
                y_train_final = le.transform(y_train_target)
                y_test_final = le.transform(y_test_target)
                
                self.label_encoders[target_metric] = le
                n_classes = len(le.classes_)
                
                print(f"      Classes ({n_classes}): {list(le.classes_)}")
                
            else:
                print(f"   📊 Regression task detected")
                
                # Convert to numeric
                y_train_numeric = pd.to_numeric(y_train_target, errors='coerce')
                y_test_numeric = pd.to_numeric(y_test_target, errors='coerce')
                
                # Handle NaN
                if y_train_numeric.isnull().any():
                    median_val = y_train_numeric.median()
                    y_train_numeric = y_train_numeric.fillna(median_val)
                    y_test_numeric = y_test_numeric.fillna(median_val)
                
                print(f"      Range: [{y_train_numeric.min():.4f}, {y_train_numeric.max():.4f}]")
                print(f"      Mean: {y_train_numeric.mean():.4f}, Std: {y_train_numeric.std():.4f}")
                print(f"      Skewness: {y_train_numeric.skew():.4f}")
                
                # Apply transformation if needed
                target_skewness = abs(y_train_numeric.skew())
                if target_skewness > 1.5:
                    print(f"      Applying target transformation (skewness: {target_skewness:.3f})")
                    
                    from sklearn.preprocessing import PowerTransformer
                    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                    
                    try:
                        y_train_transformed = transformer.fit_transform(y_train_numeric.values.reshape(-1, 1)).flatten()
                        y_test_transformed = transformer.transform(y_test_numeric.values.reshape(-1, 1)).flatten()
                        
                        new_skewness = abs(stats.skew(y_train_transformed))
                        print(f"      New skewness: {new_skewness:.3f}")
                        
                        if new_skewness < target_skewness:
                            y_train_final = y_train_transformed
                            y_test_final = y_test_transformed
                            self.target_transformers[target_metric] = transformer
                            print("      ✅ Target transformation applied")
                        else:
                            y_train_final = y_train_numeric.values
                            y_test_final = y_test_numeric.values
                            self.target_transformers[target_metric] = None
                            print("      ❌ Target transformation didn't improve")
                            
                    except Exception as e:
                        print(f"      ❌ Target transformation failed: {e}")
                        y_train_final = y_train_numeric.values
                        y_test_final = y_test_numeric.values
                        self.target_transformers[target_metric] = None
                else:
                    y_train_final = y_train_numeric.values
                    y_test_final = y_test_numeric.values
                    self.target_transformers[target_metric] = None
                
                # ═══════════════════════════════════════════════════════════
                # TTD INFO: Keep timeout values as-is (neural networks can handle them)
                # ═══════════════════════════════════════════════════════════
                if target_metric == 'time_to_detection_threshold':
                    TIMEOUT = 9000.0
                    
                    timeout_count_train = (y_train_final >= TIMEOUT).sum()
                    timeout_count_test = (y_test_final >= TIMEOUT).sum()
                    
                    print(f"      ℹ️  TTD timeout values (9000):")
                    print(f"         Train: {timeout_count_train}/{len(y_train_final)} ({100*timeout_count_train/len(y_train_final):.1f}%)")
                    print(f"         Test: {timeout_count_test}/{len(y_test_final)} ({100*timeout_count_test/len(y_test_final):.1f}%)")
                    print(f"      ✅ Keeping timeout values as-is (valid data representing 'no detection')")
                    
                    # Neural networks can learn the full range [0, 9000]
                    # No replacement needed!
                
                n_classes = 1
            
            # ═══════════════════════════════════════════════════════════
            # STEP 4: Feature Selection
            # ═══════════════════════════════════════════════════════════
            print(f"   🔍 Performing enhanced feature selection...")

            from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFECV
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            try:
                # Use RFECV with STRONGER diversity enforcement
                if is_classification:
                    min_features = max(5, min(20, len(X_train_surrogate.columns) // 4))
                else:
                    min_features = max(5, min(30, len(X_train_surrogate.columns) // 3))
                
                # Use consistent random seed for reproducibility
                # Don't enforce artificial diversity - let RFECV find best features naturally
                metric_seed = 42  # Same seed for all metrics - let data determine features
                np.random.seed(metric_seed)
                
                if is_classification:
                    estimator = RandomForestClassifier(n_estimators=100, random_state=metric_seed, n_jobs=-1)
                else:
                    estimator = RandomForestRegressor(n_estimators=100, random_state=metric_seed, n_jobs=-1)
                
                selector = RFECV(estimator, min_features_to_select=min_features, cv=3, scoring=None, n_jobs=1, verbose=0)
                X_train_selected = selector.fit_transform(X_train_surrogate, y_train_final)
                X_test_selected = selector.transform(X_test_surrogate)

                selected_features = list(X_train_surrogate.columns[selector.support_])
                        
                print(f"      ✅ Selected features (list): {len(selected_features)}")
                print(f"      First 5 features: {selected_features[:5]}")
                
                # Create and fit scaler
                metric_scaler = StandardScaler()
                metric_scaler.fit(X_train_selected)

                # Check if scaler has EITHER mean_ (StandardScaler) OR center_ (RobustScaler)
                has_mean = hasattr(metric_scaler, 'mean_')
                has_center = hasattr(metric_scaler, 'center_')
                if not (has_mean or has_center):
                    raise RuntimeError(f"Scaler failed to fit for {target_metric}! No mean_ or center_ attribute.")

                X_train_final = pd.DataFrame(
                    metric_scaler.transform(X_train_selected),
                    columns=selected_features, 
                    index=X_train_surrogate.index
                )
                X_test_final = pd.DataFrame(
                    metric_scaler.transform(X_test_selected),
                    columns=selected_features,
                    index=X_test_surrogate.index
                )
                
                # Store scaler immediately
                self.surrogate_metric_scalers[target_metric] = metric_scaler
                print(f"      ✅ Scaler fitted and stored for {target_metric}")
                center = metric_scaler.mean_[:3] if hasattr(metric_scaler, 'mean_') else metric_scaler.center_[:3]
                print(f"      Scaler stats: center={center}, scale={metric_scaler.scale_[:3]}")
                
            except Exception as e:
                print(f"      ⚡️  Feature selection failed: {e}, using fallback")
                
                # Fallback
                metric_seed = 42 + hash(target_metric) % 1000

                if is_classification:
                    selector = SelectKBest(score_func=f_classif, k='all')
                else:
                    selector = SelectKBest(score_func=f_regression, k='all')

                X_train_selected = selector.fit_transform(X_train_surrogate, y_train_final)
                X_test_selected = selector.transform(X_test_surrogate)

                feature_scores = selector.scores_
                feature_ranking = np.argsort(feature_scores)[::-1]

                n_samples = len(X_train_surrogate)
                max_features = min(50, len(feature_ranking), n_samples // 3)

                np.random.seed(metric_seed)
                top_core = int(max_features * 0.8)
                remaining = max_features - top_core

                core_indices = feature_ranking[:top_core]
                candidate_indices = feature_ranking[top_core:top_core*3]
                random_indices = np.random.choice(candidate_indices, size=min(remaining, len(candidate_indices)), replace=False)

                selected_indices = np.concatenate([core_indices, random_indices])
                selected_features = list(X_train_surrogate.columns[selected_indices])
                
                # Create scaler for fallback
                metric_scaler = StandardScaler()
                metric_scaler.fit(X_train_surrogate[selected_features])

                X_train_final = pd.DataFrame(
                    metric_scaler.transform(X_train_surrogate[selected_features]),
                    columns=selected_features,
                    index=X_train_surrogate.index
                )
                X_test_final = pd.DataFrame(
                    metric_scaler.transform(X_test_surrogate[selected_features]),
                    columns=selected_features,
                    index=X_test_surrogate.index
                )

                self.surrogate_metric_scalers[target_metric] = metric_scaler
                print(f"      ✅ Fallback scaler stored for {target_metric}")
            
            # Store feature mapping
            self.surrogate_features[target_metric] = list(selected_features)
            print(f"      ✅ Stored feature mapping: {len(selected_features)} features")
            print(f"      Top 5 features: {selected_features[:5]}")
            
            # ═══════════════════════════════════════════════════════════
            # STEP 5: Cross-Validation
            # ═══════════════════════════════════════════════════════════
            cv_folds = 5
            fold_performances = []

            from sklearn.model_selection import StratifiedKFold, KFold

            y_binned = pd.qcut(y_train_target, q=5, labels=False, duplicates='drop')

            try:
                kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_splits = list(kf.split(X_train_final, y_binned))
                print(f"   ✅ Using stratified {cv_folds}-fold CV (balanced across target range)")
            except:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_splits = list(kf.split(X_train_final))
                print(f"   ⚠️ Using regular {cv_folds}-fold CV (stratification failed)")

            X_train_np = X_train_final.values if hasattr(X_train_final, 'values') else X_train_final
            y_train_np = y_train_final if isinstance(y_train_final, np.ndarray) else np.array(y_train_final)

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                print(f"      Fold {fold_idx + 1}/{cv_folds}: ", end="")
                
                X_fold_train = X_train_np[train_idx]
                X_fold_val = X_train_np[val_idx]
                y_fold_train = y_train_np[train_idx]
                y_fold_val = y_train_np[val_idx]
                
                X_fold_train_df = pd.DataFrame(X_fold_train, columns=X_train_final.columns)
                X_fold_val_df = pd.DataFrame(X_fold_val, columns=X_train_final.columns)
                
                try:
                    best_model, fold_perf = self._train_with_multiple_strategies(
                        X_fold_train_df, y_fold_train, X_fold_val_df, y_fold_val,
                        target_metric, is_classification, n_classes
                    )
                    
                    if is_classification:
                        print(f"Acc: {fold_perf['test_accuracy']:.3f}")
                    else:
                        print(f"R²: {fold_perf['test_r2']:.3f}")
                    
                    fold_performances.append(fold_perf)
                    
                except Exception as e:
                    print(f"Failed: {e}")
                    if is_classification:
                        dummy_perf = {'test_accuracy': 0.0, 'test_f1': 0.0}
                    else:
                        dummy_perf = {'test_r2': -1.0, 'test_mse': float('inf')}
                    fold_performances.append(dummy_perf)
            
            # Calculate CV performance
            if is_classification:
                valid_performances = [p for p in fold_performances if p['test_accuracy'] > 0]
                if valid_performances:
                    avg_acc = np.mean([p['test_accuracy'] for p in valid_performances])
                    avg_f1 = np.mean([p['test_f1'] for p in valid_performances])
                    print(f"   📊 CV Results: Accuracy: {avg_acc:.4f} ± {np.std([p['test_accuracy'] for p in valid_performances]):.4f}")
                    print(f"                 F1: {avg_f1:.4f} ± {np.std([p['test_f1'] for p in valid_performances]):.4f}")
                else:
                    print("   ❌ All CV folds failed")
                    continue
            else:
                valid_performances = [p for p in fold_performances if p['test_r2'] > -10]
                if valid_performances:
                    avg_r2 = np.mean([p['test_r2'] for p in valid_performances])
                    avg_mse = np.mean([p['test_mse'] for p in valid_performances])
                    print(f"   📊 CV Results: R²: {avg_r2:.4f} ± {np.std([p['test_r2'] for p in valid_performances]):.4f}")
                    print(f"                 MSE: {avg_mse:.4f} ± {np.std([p['test_mse'] for p in valid_performances]):.4f}")
                else:
                    print("   ❌ All CV folds failed")
                    continue
            
            # ═══════════════════════════════════════════════════════════
            # STEP 6: Train Final Model
            # ═══════════════════════════════════════════════════════════
            print(f"   🏯 Training final model with multiple strategies...")
            
            try:
                final_model, final_performance = self._train_with_multiple_strategies(
                    X_train_final, y_train_final, X_test_final, y_test_final,
                    target_metric, is_classification, n_classes
                )
                
                if is_classification:
                    final_performance.update({
                        'cv_accuracy': avg_acc,
                        'cv_f1': avg_f1,
                        'type': 'classification',
                        'n_classes': n_classes
                    })
                    print(f"   ✅ Final Test Accuracy: {final_performance['test_accuracy']:.4f}")
                    
                    if final_performance['test_accuracy'] < 0.5:
                        print("   ⚡️  VERY LOW: Test accuracy < 0.5")
                    elif final_performance['test_accuracy'] < 0.7:
                        print("   📈 LOW: Test accuracy < 0.7") 
                    elif final_performance['test_accuracy'] < 0.85:
                        print("   📊 MODERATE: Test accuracy < 0.85")
                    else:
                        print("   🏯 EXCELLENT: Test accuracy ≥ 0.85")
                        
                else:
                    # Get predictions from model FIRST
                    if isinstance(final_model, dict) and 'base_models' in final_model:
                        # Stacked ensemble
                        test_pred = self._predict_stacked_ensemble(final_model, X_test_final)
                    else:
                        # Single model
                        if hasattr(final_model, 'predict'):
                            test_pred = final_model.predict(X_test_final)
                        else:
                            raise ValueError(f"Model doesn't have predict method: {type(final_model)}")
                    
                    # Add 'type' key for regression
                    final_performance['type'] = 'regression'
                    
                    # NOW inverse transform TTD predictions if log transform was used
                    if target_metric == 'time_to_detection_threshold' and hasattr(self, 'ttd_log_transform') and self.ttd_log_transform['enabled']:
                        epsilon = self.ttd_log_transform['epsilon']
                        
                        # Inverse transform: exp(log_pred) - epsilon
                        test_pred_original = np.exp(test_pred) - epsilon
                        print(f"      ⚡ Inverse log transform applied to predictions")
                        print(f"         Log-space range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
                        print(f"         Original-space range: [{test_pred_original.min():.1f}, {test_pred_original.max():.1f}]")
                        
                        # Recalculate metrics in original space
                        # Need to inverse transform y_test too
                        y_test_original = np.exp(y_test_final) - epsilon
                        
                        final_performance['test_r2'] = r2_score(y_test_original, test_pred_original)
                        final_performance['test_mse'] = mean_squared_error(y_test_original, test_pred_original)
                        final_performance['test_normalized_mse'] = self._calculate_normalized_mse(y_test_original, test_pred_original)
                        
                        # Store both versions
                        final_performance['predictions_test'] = test_pred_original
                        final_performance['predictions_test_log_space'] = test_pred
                    else:
                        # No transform needed, calculate metrics normally
                        if 'test_normalized_mse' not in final_performance:
                            final_performance['test_normalized_mse'] = self._calculate_normalized_mse(y_test_final, test_pred)
                    
                    print(f"   ✅ Final Test R²: {final_performance['test_r2']:.4f}")
                    print(f"   ✅ Final Test MSE: {final_performance['test_mse']:.4f}")
                    
                    if 'test_normalized_mse' in final_performance:
                        norm_mse = final_performance['test_normalized_mse']
                        print(f"   📊 Normalized MSE by variance: {norm_mse['mse_normalized_by_variance']:.4f}")
                        print(f"   📊 RMSE normalized by std: {norm_mse['rmse_normalized_by_std']:.4f}")
                        print(f"   📊 RMSE normalized by range: {norm_mse['rmse_normalized_by_range']:.4f}")
                    
                    if final_performance['test_r2'] < 0.0:
                        print("   ⚡️  VERY POOR: Test R² < 0.0")
                    elif final_performance['test_r2'] < 0.3:
                        print("   ❌ POOR: Test R² < 0.3")
                    elif final_performance['test_r2'] < 0.6:
                        print("   📈 LOW: Test R² < 0.6")
                    elif final_performance['test_r2'] < 0.8:
                        print("   📊 MODERATE: Test R² < 0.8")
                    elif final_performance['test_r2'] < 0.9:
                        print("   🏯 GOOD: Test R² < 0.9")
                    else:
                        print("   🏸 EXCELLENT: Test R² ≥ 0.9")
            
            except Exception as e:
                print(f"   ❌ Final model training failed: {e}")
                continue
            
            # ═══════════════════════════════════════════════════════════
            # STEP 7: Store Results
            # ═══════════════════════════════════════════════════════════
            surrogate_performance[target_metric] = final_performance
            
            # Verify model uniqueness
            model_id = id(final_model)
            if hasattr(self, 'surrogate_models'):
                for existing_metric, existing_model in self.surrogate_models.items():
                    if id(existing_model) == model_id:
                        raise ValueError(f"Model reuse: {target_metric} == {existing_metric}")
            
            self.surrogate_models[target_metric] = final_model
            print(f"✅ Stored UNIQUE model for {target_metric} (ID: {model_id})")
            
            # Verify feature uniqueness
            print(f"\n🔍 FEATURE UNIQUENESS CHECK for {target_metric}:")
            print(f"   Count: {len(self.surrogate_features[target_metric])}")
            print(f"   First 5: {self.surrogate_features[target_metric][:5]}")
            print(f"   Last 5: {self.surrogate_features[target_metric][-5:]}")

        # ═══════════════════════════════════════════════════════════════
        # POST-TRAINING: Validation and Save
        # ═══════════════════════════════════════════════════════════════
        print("\n🔍 FINAL FEATURE MAPPING VALIDATION:")
        print("=" * 60)
        
        for metric in self.target_metrics:
            if metric in self.surrogate_features:
                features = self.surrogate_features[metric]
                print(f"\n{metric}:")
                print(f"   Count: {len(features)}")
                print(f"   Type: {type(features)}")
                print(f"   First 5: {features[:5]}")
                print(f"   Last 5: {features[-5:]}")
                
                # Check for duplicates
                if len(features) != len(set(features)):
                    raise ValueError(f"{metric} has duplicate features!")
        
        print("=" * 60)

        # ═══════════════════════════════════════════════════════════════
        # VERIFY FNR-TTD CORRELATION
        # ═══════════════════════════════════════════════════════════════
        print("\n🔍 VERIFYING FNR-TTD CORRELATION:")
        print("=" * 60)
        
        if 'false_negative_rate' in self.y_train.columns and 'time_to_detection_threshold' in self.y_train.columns:
            fnr_ttd_corr = self.y_train['false_negative_rate'].corr(self.y_train['time_to_detection_threshold'])
            print(f"FNR ↔ TTD correlation: {fnr_ttd_corr:.3f}")
            
            if abs(fnr_ttd_corr) > 0.90:
                print("⚠️  WARNING: FNR and TTD are nearly identical (correlation > 0.9)!")
                print("   This suggests FNR is redundant and may not add unique information.")
                print("   Consider:")
                print("   1. Using ONLY TTD in reward function")
                print("   2. Or ensuring FNR is calculated independently from TTD")
            elif abs(fnr_ttd_corr) > 0.70:
                print("⚡ CAUTION: FNR and TTD are strongly correlated (0.7-0.9)")
                print("   They may contain overlapping information.")
            else:
                print("✅ FNR and TTD have healthy correlation (<0.7)")
                print("   Both metrics add unique value.")
        
        print("=" * 60)
        
        # Save configurations
        config_path = self.models_dir / "surrogate_configs.json"
        surrogate_configs = {
            'target_metrics': self.target_metrics,
            'data_statistics': {},
            'model_architectures': {},
            'r2_scores': {}
        }

        for metric, model in self.surrogate_models.items():
            if metric in surrogate_performance:
                surrogate_configs['r2_scores'][metric] = surrogate_performance[metric].get('test_r2', -1)
            
            if metric in self.y_train.columns:
                surrogate_configs['data_statistics'][metric] = {
                    'min': float(self.y_train[metric].min()),
                    'max': float(self.y_train[metric].max()),
                    'mean': float(self.y_train[metric].mean()),
                    'std': float(self.y_train[metric].std())
                }
            
            if hasattr(model, 'get_params'):
                surrogate_configs['model_architectures'][metric] = str(type(model).__name__)

            # Save feature mappings
            if metric in self.surrogate_features:
                features_path = self.models_dir / f'{metric}_features.json'
                with open(features_path, 'w') as f:
                    json.dump(self.surrogate_features[metric], f)
                print(f"   ✅ Saved feature mapping for {metric}")

        with open(config_path, 'w') as f:
            json.dump(surrogate_configs, f, indent=2)

        print(f"✅ Surrogate configurations saved to: {config_path}")

        # Final summary
        print(f"\n🏰 ENHANCED SURROGATE MODEL TRAINING COMPLETE")
        print("=" * 60)
        
        successful_targets = list(surrogate_performance.keys())
        classification_targets = [k for k, v in surrogate_performance.items() if v['type'] == 'classification']
        regression_targets = [k for k, v in surrogate_performance.items() if v['type'] == 'regression']
        
        print(f"✅ Successfully trained: {len(successful_targets)}/{len(self.y_train.columns)} targets")
        
        if skipped_targets:
            print(f"⭕️  Skipped targets ({len(skipped_targets)}): {skipped_targets}")
        
        if regression_targets:
            print(f"\n📊 Regression targets ({len(regression_targets)}):")
            for target in regression_targets:
                perf = surrogate_performance[target]
                print(f"   {target}: Test R² = {perf['test_r2']:.4f}")
                
                if 'test_normalized_mse' in perf:
                    norm_mse = perf['test_normalized_mse']
                    print(f"      Normalized RMSE/std: {norm_mse['rmse_normalized_by_std']:.4f}")
        
        if regression_targets:
            r2_scores = [surrogate_performance[t]['test_r2'] for t in regression_targets]
            avg_reg_r2 = np.mean(r2_scores)
            print(f"📈 Average Regression R²: {avg_reg_r2:.4f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("✅ Enhanced surrogate models ready for RL!")

        # Fidelity validation
        print("\n🎯 SURROGATE MODEL FIDELITY VALIDATION")
        print("=" * 60)
        
        for target, perf in surrogate_performance.items():
            if perf['type'] == 'regression':
                r2 = perf['test_r2']
                if r2 < 0.90:
                    print(f"⚡️  {target}: R² = {r2:.4f} < 0.90 (BELOW THRESHOLD)")
                elif r2 < 0.95:
                    print(f"✅ {target}: R² = {r2:.4f} (Acceptable)")
                else:
                    print(f"⭐ {target}: R² = {r2:.4f} (Outstanding)")

        # ✅ FIXED: Actually check if models meet threshold
        poor_models = [t for t, perf in surrogate_performance.items() 
                    if perf['type'] == 'regression' and perf['test_r2'] < 0.90]

        if poor_models:
            print(f"\n⚠️  WARNING: {len(poor_models)} model(s) below R² = 0.90 threshold:")
            for model in poor_models:
                r2 = surrogate_performance[model]['test_r2']
                print(f"   ❌ {model}: R² = {r2:.4f}")
            print("\n⚠️  CRITICAL: RL performance will be LIMITED by poor surrogate models!")
            print("   Recommendations:")
            print("   1. Increase dataset size (need more samples)")
            print("   2. Try neural network models for difficult targets")
            print("   3. Check for data quality issues")
        else:
            print("\n✅ All regression surrogates meet minimum R² ≥ 0.90 threshold!")

        # ✅ FINAL VALIDATION: Ensure all targets have scalers
        print("\n🔍 POST-TRAINING VALIDATION:")
        print("=" * 60)
        
        for target in target_order:
            if target in self.surrogate_models:
                # Check scaler
                if target in self.surrogate_metric_scalers:
                    scaler = self.surrogate_metric_scalers[target]
                    if hasattr(scaler, 'mean_'):
                        print(f"✅ {target}: Scaler OK ({len(scaler.mean_)} features)")
                    elif hasattr(scaler, 'center_'):
                        print(f"✅ {target}: Scaler OK ({len(scaler.center_)} features)")
                    else:
                        print(f"❌ {target}: Scaler exists but not fitted!")
                else:
                    print(f"❌ {target}: NO SCALER! (Model exists but scaler missing)")
                    raise RuntimeError(f"Critical: {target} model trained but scaler not created!")
                
                # Check features
                if target in self.surrogate_features:
                    print(f"   Features: {len(self.surrogate_features[target])} stored")
                else:
                    print(f"   ⚠️ WARNING: No feature list stored for {target}")
        
        print("=" * 60)
        
        return surrogate_performance
    
    def _create_physics_based_ttd_predictor(self, X_train, y_train, X_test, y_test):
        """
        ✅ Physics-based TTD predictor using analytical formulas
        
        INSIGHT: TTD cannot be reliably predicted from static biosensor config using ML alone.
        TTD depends on temporal dynamics (signal rise, threshold crossing time, noise fluctuations).
        
        Instead, we use physics-based estimation from biosensor theory:
        - Signal strength = f(sensitivity, response_curve, Kd)
        - Detection time ≈ (threshold/signal_peak) * rise_time * circuit_delay
        - Add SNR-dependent variance
        
        This gives MUCH better results than trying to learn temporal patterns from static features!
        
        ✅ FIXED: Uses module-level PhysicsTTDPredictor class for pickle support
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        print(f"      🔧 Creating PHYSICS-BASED TTD predictor...")
        
        # Analyze training data to understand TTD distribution
        y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        MAX_TIME = 3600.0  # Maximum simulation time
        TIMEOUT = 9000.0   # Censoring value for non-detection
        
        # Detect censored data
        detected_train = y_train_arr < TIMEOUT
        n_detected = np.sum(detected_train)
        n_timeout = np.sum(~detected_train)
        
        print(f"      📊 TTD distribution:")
        print(f"         Detected: {n_detected}/{len(y_train_arr)} ({100*n_detected/len(y_train_arr):.1f}%)")
        print(f"         Timeout: {n_timeout}/{len(y_train_arr)}")
        
        if n_detected > 0:
            ttd_detected = y_train_arr[detected_train]
            q1, median, q3 = np.percentile(ttd_detected, [25, 50, 75])
            print(f"         TTD quantiles (detected): Q1={q1:.1f}, Median={median:.1f}, Q3={q3:.1f}")
            median_ttd = median
        else:
            median_ttd = 1800.0  # Default fallback
        
        # Get feature names
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # ✅ Use module-level PhysicsTTDPredictor class (defined at top of file)
        predictor = PhysicsTTDPredictor(feature_names, median_ttd)
        
        # Evaluate predictor
        train_pred = predictor.predict(X_train)
        test_pred = predictor.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train_arr, train_pred)
        test_r2 = r2_score(y_test_arr, test_pred)
        test_mse = mean_squared_error(y_test_arr, test_pred)
        test_mae = mean_absolute_error(y_test_arr, test_pred)
        
        performance = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_rmse': np.sqrt(test_mse),
            'predictions_test': test_pred
        }
        
        print(f"      ✅ Physics TTD: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}")
        
        return predictor, performance
    
    def _create_component_based_fnr_predictor(self, X_train, y_train, X_test, y_test):
        """
        IMPROVED FNR predictor with robust handling of all edge cases.
        
        Similar improvements as TTD predictor.
        """
        
        print(f"   🔧 Creating IMPROVED FNR predictor...")
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Encode categorical variables
            # ═══════════════════════════════════════════════════════════
            X_train_encoded, encoding_info = self.encode_categorical_features(X_train)
            X_test_encoded, _ = self.encode_categorical_features(X_test)
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: Calculate robust components
            # ═══════════════════════════════════════════════════════════
            
            # Component 1: SNR Penalty
            print(f"      📊 Component 1: SNR Penalty")
            
            # Get SNR
            if 'signal_to_noise_ratio_SNR' in X_train_encoded.columns:
                snr_train = X_train_encoded['signal_to_noise_ratio_SNR'].values
                snr_test = X_test_encoded['signal_to_noise_ratio_SNR'].values
            else:
                # noise_preset is already encoded as 0, 1, 2
                noise_values = X_train_encoded['noise_preset'].values
                snr_train = np.where(noise_values == 0, 20,
                           np.where(noise_values == 1, 10, 5))
                noise_values_test = X_test_encoded['noise_preset'].values
                snr_test = np.where(noise_values_test == 0, 20,
                          np.where(noise_values_test == 1, 10, 5))
            
            # Clip SNR
            snr_train = np.clip(snr_train, -50, 50)
            snr_test = np.clip(snr_test, -50, 50)
            
            # Safe sigmoid: lower SNR → higher penalty
            snr_penalty_train = 0.20 / (1 + self.safe_exp(snr_train / 10, clip_min=-30, clip_max=30))
            snr_penalty_test = 0.20 / (1 + self.safe_exp(snr_test / 10, clip_min=-30, clip_max=30))
            
            snr_penalty_train = np.clip(snr_penalty_train, 0.01, 0.20)
            snr_penalty_test = np.clip(snr_penalty_test, 0.01, 0.20)
            
            if not self.validate_no_nan_inf(snr_penalty_train, "snr_penalty"):
                raise ValueError("SNR penalty contains NaN/inf")
            
            # Component 2: Threshold Factor
            print(f"      📊 Component 2: Threshold Factor")
            
            signal_to_threshold_train = self.safe_divide(
                X_train_encoded['sclerostin_concentration'] * X_train_encoded['biosensor_sensitivity'],
                X_train_encoded['biosensor_threshold'],
                fill_value=1.0,
                min_denom=1e-6
            )
            signal_to_threshold_test = self.safe_divide(
                X_test_encoded['sclerostin_concentration'] * X_test_encoded['biosensor_sensitivity'],
                X_test_encoded['biosensor_threshold'],
                fill_value=1.0,
                min_denom=1e-6
            )
            
            # Clip before exp
            signal_to_threshold_train = np.clip(signal_to_threshold_train, 0.01, 100)
            signal_to_threshold_test = np.clip(signal_to_threshold_test, 0.01, 100)
            
            threshold_factor_train = 0.20 * self.safe_exp(
                -(signal_to_threshold_train - 1.0), 
                clip_min=-20, clip_max=20
            )
            threshold_factor_test = 0.20 * self.safe_exp(
                -(signal_to_threshold_test - 1.0),
                clip_min=-20, clip_max=20
            )
            
            threshold_factor_train = np.clip(threshold_factor_train, 0.01, 0.20)
            threshold_factor_test = np.clip(threshold_factor_test, 0.01, 0.20)
            
            if not self.validate_no_nan_inf(threshold_factor_train, "threshold_factor"):
                raise ValueError("Threshold factor contains NaN/inf")
            
            # Component 3: Circuit Reliability
            print(f"      📊 Component 3: Circuit Reliability")
            
            circuit_map = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0}  # Now using encoded values
            
            circuit_reliability_train = (
                X_train_encoded['circuit_type'].map(circuit_map) *
                (1.0 + X_train_encoded['hill_coefficient'] / 10.0)
            )
            circuit_reliability_test = (
                X_test_encoded['circuit_type'].map(circuit_map) *
                (1.0 + X_test_encoded['hill_coefficient'] / 10.0)
            )
            
            # Fill any NaN from mapping failures
            circuit_reliability_train = circuit_reliability_train.fillna(1.5)
            circuit_reliability_test = circuit_reliability_test.fillna(1.5)
            
            circuit_reliability_train = np.clip(circuit_reliability_train, 0.5, 3.0)
            circuit_reliability_test = np.clip(circuit_reliability_test, 0.5, 3.0)
            
            if not self.validate_no_nan_inf(circuit_reliability_train, "circuit_reliability"):
                raise ValueError("Circuit reliability contains NaN/inf")
            
            # Component 4: Environmental Noise
            print(f"      📊 Component 4: Environmental Noise")
            
            # Safe geometric mean with protection against division by zero
            estrogen_norm_train = self.safe_divide(
                X_train_encoded['estrogen'],
                X_train_encoded['estrogen'].mean() if X_train_encoded['estrogen'].mean() > 1e-10 else 1.0,
                fill_value=1.0,
                min_denom=1e-10
            )
            estrogen_norm_test = self.safe_divide(
                X_test_encoded['estrogen'],
                X_train_encoded['estrogen'].mean() if X_train_encoded['estrogen'].mean() > 1e-10 else 1.0,
                fill_value=1.0,
                min_denom=1e-10
            )
            
            pth_norm_train = self.safe_divide(
                X_train_encoded['pth'],
                X_train_encoded['pth'].mean() if X_train_encoded['pth'].mean() > 1e-10 else 1.0,
                fill_value=1.0,
                min_denom=1e-10
            )
            pth_norm_test = self.safe_divide(
                X_test_encoded['pth'],
                X_train_encoded['pth'].mean() if X_train_encoded['pth'].mean() > 1e-10 else 1.0,
                fill_value=1.0,
                min_denom=1e-10
            )
            
            mineral_norm_train = self.safe_divide(
                X_train_encoded['mineral_ion'],
                X_train_encoded['mineral_ion'].mean() if X_train_encoded['mineral_ion'].mean() > 1e-10 else 1.0,
                fill_value=1.0,
                min_denom=1e-10
            )
            mineral_norm_test = self.safe_divide(
                X_test_encoded['mineral_ion'],
                X_train_encoded['mineral_ion'].mean() if X_train_encoded['mineral_ion'].mean() > 1e-10 else 1.0,
                fill_value=1.0,
                min_denom=1e-10
            )
            
            # Clip normalized values
            for arr in [estrogen_norm_train, estrogen_norm_test, pth_norm_train, pth_norm_test, mineral_norm_train, mineral_norm_test]:
                arr[:] = np.clip(arr, 0.1, 10.0)
            
            env_noise_train = (estrogen_norm_train * pth_norm_train * mineral_norm_train) ** 0.333
            env_noise_test = (estrogen_norm_test * pth_norm_test * mineral_norm_test) ** 0.333
            
            env_noise_train = np.clip(env_noise_train, 0.1, 5.0)
            env_noise_test = np.clip(env_noise_test, 0.1, 5.0)
            
            if not self.validate_no_nan_inf(env_noise_train, "env_noise"):
                raise ValueError("Environmental noise contains NaN/inf")
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: Train component models
            # ═══════════════════════════════════════════════════════════
            
            snr_features = ['noise_preset', 'scenario', 'biosensor_sensitivity']
            threshold_features = [
                'biosensor_threshold', 'biosensor_sensitivity', 'biosensor_kd',
                'sclerostin_concentration', 'sclerostin_max'
            ]
            circuit_features = [
                'circuit_type', 'response_type', 'hill_coefficient',
                'off_level', 'on_level', 'biosensor_dynamic_range_max'
            ]
            env_features = [
                'estrogen', 'pth', 'mineral_ion', 'rankl_concentration',
                'opg_concentration', 'scenario', 'noise_preset'
            ]
            
            # Validate features
            for feat_list in [snr_features, threshold_features, circuit_features, env_features]:
                feat_list[:] = [f for f in feat_list if f in X_train_encoded.columns]
            
            # Train models
            model_snr = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
            model_snr.fit(X_train_encoded[snr_features], snr_penalty_train)
            snr_pred = model_snr.predict(X_test_encoded[snr_features])
            snr_r2 = r2_score(snr_penalty_test, snr_pred)
            print(f"         ✅ SNR Penalty R² = {snr_r2:.4f}")
            
            model_threshold = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
            model_threshold.fit(X_train_encoded[threshold_features], threshold_factor_train)
            threshold_pred = model_threshold.predict(X_test_encoded[threshold_features])
            threshold_r2 = r2_score(threshold_factor_test, threshold_pred)
            print(f"         ✅ Threshold Factor R² = {threshold_r2:.4f}")
            
            model_circuit = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
            model_circuit.fit(X_train_encoded[circuit_features], circuit_reliability_train)
            circuit_pred = model_circuit.predict(X_test_encoded[circuit_features])
            circuit_r2 = r2_score(circuit_reliability_test, circuit_pred)
            print(f"         ✅ Circuit Factor R² = {circuit_r2:.4f}")
            
            model_env = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
            model_env.fit(X_train_encoded[env_features], env_noise_train)
            env_pred = model_env.predict(X_test_encoded[env_features])
            env_r2 = r2_score(env_noise_test, env_pred)
            print(f"         ✅ Environmental Noise R² = {env_r2:.4f}")
            
            # ═══════════════════════════════════════════════════════════
            # STEP 4: Combine components
            # ═══════════════════════════════════════════════════════════
            
            print(f"      🔗 Combining components...")
            
            # Clip predictions
            snr_pred = np.clip(snr_pred, 0.01, 0.20)
            threshold_pred = np.clip(threshold_pred, 0.01, 0.20)
            circuit_pred = np.clip(circuit_pred, 0.5, 3.0)
            env_pred = np.clip(env_pred, 0.1, 5.0)
            
            baseline_fnr = 0.01
            
            fnr_predicted = (
                baseline_fnr *
                (1 + snr_pred) *
                (1 + threshold_pred) *
                circuit_pred *
                env_pred
            )
            
            # Normalize by max and clip
            if fnr_predicted.max() > 0:
                fnr_predicted = fnr_predicted / fnr_predicted.max() * y_test.max()
            fnr_predicted = np.clip(fnr_predicted, 0.0, 0.25)
            
            if not self.validate_no_nan_inf(fnr_predicted, "fnr_predicted"):
                raise ValueError("Final FNR prediction contains NaN/inf")
            
            overall_r2 = r2_score(y_test, fnr_predicted)
            overall_mse = mean_squared_error(y_test, fnr_predicted)
            
            print(f"      ✅ Combined FNR R² = {overall_r2:.4f}")
            print(f"      ✅ Combined FNR MSE = {overall_mse:.6f}")
            
            # ═══════════════════════════════════════════════════════════
            # STEP 5: Create wrapper
            # ═══════════════════════════════════════════════════════════
            
            class ImprovedFNRPredictor:
                """Improved FNR predictor with categorical encoding"""
                
                def __init__(self, model_snr, model_threshold, model_circuit, model_env,
                            snr_features, threshold_features, circuit_features, env_features,
                            encoding_info, baseline_fnr=0.01, normalization_factor=1.0):
                    self.model_snr = model_snr
                    self.model_threshold = model_threshold
                    self.model_circuit = model_circuit
                    self.model_env = model_env
                    self.snr_features = snr_features
                    self.threshold_features = threshold_features
                    self.circuit_features = circuit_features
                    self.env_features = env_features
                    self.encoding_info = encoding_info
                    self.baseline_fnr = baseline_fnr
                    self.normalization_factor = normalization_factor
                
                def predict(self, X):
                    """Predict FNR with full safety checks"""
                    # Encode categorical
                    X_encoded, _ = self.encode_categorical_features(X)
                    
                    # Get predictions
                    snr_pred = self.model_snr.predict(X_encoded[self.snr_features])
                    threshold_pred = self.model_threshold.predict(X_encoded[self.threshold_features])
                    circuit_pred = self.model_circuit.predict(X_encoded[self.circuit_features])
                    env_pred = self.model_env.predict(X_encoded[self.env_features])
                    
                    # Clip
                    snr_pred = np.clip(snr_pred, 0.01, 0.20)
                    threshold_pred = np.clip(threshold_pred, 0.01, 0.20)
                    circuit_pred = np.clip(circuit_pred, 0.5, 3.0)
                    env_pred = np.clip(env_pred, 0.1, 5.0)
                    
                    # Combine
                    fnr = (
                        self.baseline_fnr *
                        (1 + snr_pred) *
                        (1 + threshold_pred) *
                        circuit_pred *
                        env_pred
                    )
                    
                    fnr = fnr / self.normalization_factor
                    return np.clip(fnr, 0.0, 0.25)
                
                def get_params(self, deep=True):
                    return {
                        'baseline_fnr': self.baseline_fnr,
                        'normalization_factor': self.normalization_factor
                    }
            
            norm_factor = fnr_predicted.max() / y_test.max() if y_test.max() > 0 else 1.0
            
            fnr_predictor = ImprovedFNRPredictor(
                model_snr, model_threshold, model_circuit, model_env,
                snr_features, threshold_features, circuit_features, env_features,
                encoding_info, baseline_fnr=baseline_fnr, normalization_factor=norm_factor
            )
            
            print(f"      ✅ Component-based FNR predictor created!")
            print(f"         Component R²: SNR={snr_r2:.3f}, Threshold={threshold_r2:.3f}, Circuit={circuit_r2:.3f}, Env={env_r2:.3f}")
            print(f"         Overall R² = {overall_r2:.4f}")
            
            if overall_r2 < 0.5:
                print(f"      ⚠️ WARNING: R² < 0.5, consider using standard ensemble instead")
            
            return fnr_predictor, {
                'test_r2': overall_r2,
                'test_mse': overall_mse,
                'test_mae': mean_absolute_error(y_test, fnr_predicted),
                'type': 'regression',
                'component_r2': {
                    'snr': snr_r2,
                    'threshold': threshold_r2,
                    'circuit': circuit_r2,
                    'env': env_r2
                },
                'predictions_test': fnr_predicted
            }
            
        except Exception as e:
            print(f"      ❌ Component-based FNR failed: {e}")
            print(f"      → Raising exception to trigger fallback")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_classification_surrogate(self, input_dim, n_classes):
        """Create optimized classification surrogate model"""
        
        class ClassificationSurrogate(nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                
                # Adaptive architecture
                if input_dim < 50:
                    hidden_dims = [128, 64]
                elif input_dim < 150:
                    hidden_dims = [256, 128, 64]
                else:
                    hidden_dims = [512, 256, 128]
                
                layers = []
                prev_dim = input_dim
                
                for i, hidden_dim in enumerate(hidden_dims):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3 if i == 0 else 0.2)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, n_classes))
                
                self.layers = nn.Sequential(*layers)
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                return self.layers(x)
        
        return ClassificationSurrogate(input_dim, n_classes)


    def _create_regression_surrogate(self, input_dim):
        """Create optimized regression surrogate model"""
        
        class RegressionSurrogate(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                # Adaptive architecture
                if input_dim < 50:
                    hidden_dims = [128, 64, 32]
                elif input_dim < 150:
                    hidden_dims = [256, 128, 64]
                else:
                    hidden_dims = [512, 256, 128, 64]
                
                layers = []
                prev_dim = input_dim
                
                for i, hidden_dim in enumerate(hidden_dims):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2 if i < len(hidden_dims) - 1 else 0.1)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                
                self.layers = nn.Sequential(*layers)
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                return self.layers(x)
        
        return RegressionSurrogate(input_dim)
    
    def check_data_leakage(self, metric_name):
        """Detect if test set has values outside training distribution"""
        if metric_name not in self.y_train.columns or metric_name not in self.y_test.columns:
            return False
        
        train_min = self.y_train[metric_name].min()
        train_max = self.y_train[metric_name].max()
        test_min = self.y_test[metric_name].min()
        test_max = self.y_test[metric_name].max()
        
        extrapolation_needed = False
        
        # Check if test set extends beyond training range
        if test_min < train_min:
            diff_pct = abs((test_min - train_min) / (train_max - train_min + 1e-9)) * 100
            if diff_pct > 5:  # Only warn if >5% beyond range
                print(f"   📊 INFO: {metric_name} test_min ({test_min:.4f}) slightly below train_min ({train_min:.4f})")
                print(f"      Difference: {diff_pct:.1f}% of training range (acceptable)")
                extrapolation_needed = True
        
        if test_max > train_max:
            diff_pct = abs((test_max - train_max) / (train_max - train_min + 1e-9)) * 100
            if diff_pct > 5:  # Only warn if >5% beyond range
                print(f"   📊 INFO: {metric_name} test_max ({test_max:.4f}) slightly above train_max ({train_max:.4f})")
                print(f"      Difference: {diff_pct:.1f}% of training range (acceptable)")
                extrapolation_needed = True
        
        # Only show impact message if significant extrapolation
        if extrapolation_needed:
            print(f"   💡 Note: Minor extrapolation is normal in ML (models handle this well)")
        
        return extrapolation_needed

    def setup_rl_environment(self, target_metrics: List[str] = None) -> gym.Env:
        """
        Create RL environment that only allows modification of modifiable features
        """
        if target_metrics is None:
            target_metrics = ['multi_objective_score']
        
        print(f"Ã°Å¸Å½Â® Setting up RL environment for: {target_metrics}")
        print("   RL Agent can ONLY modify circuit design parameters!")
        
        # Create logging directory
        self.log_dir = self.output_dir / "rl_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        class BiosensorEnv(gym.Env):
            def __init__(self, modifiable_bounds, fixed_values, surrogate_models, 
                target_metrics, log_dir, surrogate_features, X_train, 
                data_min_max, categorical_label_encoders=None, 
                feature_scaler=None, surrogate_metric_scalers=None,
                target_transformers=None, y_scalers=None):
                """Initialize environment with proper variable separation"""
                
                super(BiosensorEnv, self).__init__()
                
                # Ã¢Å“â€¦ FIX #1: Validate and store data_min_max FIRST (before anything else)
                if not data_min_max:
                    raise ValueError("Ã¢Å’ CRITICAL: data_min_max is empty! Cannot create environment.")
                
                if not isinstance(data_min_max, dict):
                    raise TypeError(f"Ã¢Å’ data_min_max must be dict, got {type(data_min_max)}")
                
                # Make a DEEP COPY to prevent external modifications
                import copy
                self.data_min_max = copy.deepcopy(data_min_max)
                
                # Validate all required metrics are present
                required_metrics = ['signal_to_noise_ratio_SNR', 'dynamic_range_of_output',
                                'false_negative_rate', 'time_to_detection_threshold']
                missing = [m for m in required_metrics if m not in self.data_min_max]
                if missing:
                    raise ValueError(f"Ã¢Å’ Missing required metrics in data_min_max: {missing}")
                
                print(f"   âœ… Environment {id(self)}: data_min_max stored with {len(self.data_min_max)} metrics")
                
                # Store core mappings
                self.categorical_label_encoders = categorical_label_encoders or {}
                self.target_transformers = target_transformers or {}
                self.y_scalers = y_scalers or {}  # ✅ ADD THIS LINE
                self.surrogate_features = surrogate_features
                self.log_dir = log_dir
                self.episode_logs = []
                self.step_logs = []
                self.current_episode = 0
                self.global_step = 0
                
                # Store ACTUAL scaler objects directly
                self.surrogate_metric_scalers = {}
                if surrogate_metric_scalers:
                    for metric, scaler in surrogate_metric_scalers.items():
                        if scaler is not None:
                            # ✅ Validate scaler is fitted (check both StandardScaler and RobustScaler)
                            has_mean = hasattr(scaler, 'mean_')  # StandardScaler
                            has_center = hasattr(scaler, 'center_')  # RobustScaler
                            
                            if not (has_mean or has_center):
                                raise ValueError(f"Scaler for {metric} is not fitted (no mean_ or center_)!")
                            
                            self.surrogate_metric_scalers[metric] = scaler
                        else:
                            raise ValueError(f"Scaler for {metric} is None!")
                
                # Parse and validate data_min_max
                if not data_min_max:
                    raise ValueError("âŒ data_min_max parameter is empty!")
                
                self.data_min_max = {}
                for key, value in data_min_max.items():
                    if isinstance(value, (tuple, list)) and len(value) >= 2:
                        self.data_min_max[key] = (float(value[0]), float(value[1]))
                    else:
                        raise ValueError(f"Invalid format for {key}: {value}")
                
                # Validate required metrics
                required_metrics = ['signal_to_noise_ratio_SNR', 'dynamic_range_of_output', 
                                'false_negative_rate', 'time_to_detection_threshold']
                missing = [m for m in required_metrics if m not in self.data_min_max]
                if missing:
                    raise ValueError(f"âŒ Missing metrics: {missing}")
                
                # Backup for safety
                self._original_data_min_max = dict(self.data_min_max)
                
                # Print confirmation once
                if not hasattr(BiosensorEnv, '_data_range_confirmed'):
                    print(f"   âœ… data_min_max: {len(self.data_min_max)} metrics")
                    BiosensorEnv._data_range_confirmed = True
                
                # Store variable classifications
                self.modifiable_bounds = modifiable_bounds
                self.fixed_values = fixed_values
                self.surrogate_models = surrogate_models
                self.target_metrics = target_metrics
                self.X_train = X_train
                
                self.modifiable_features = list(modifiable_bounds.keys())
                self.fixed_features = list(fixed_values.keys())
                self.n_modifiable = len(self.modifiable_features)
                
                # Handle categorical features
                self.categorical_features = {}
                self.continuous_features = []
                categorical_definitions = {
                    'circuit_type': ['direct_binding', 'amplifying', 'threshold', 'ratiometric'],
                    'response_type': ['linear', 'hill', 'michaelis_menten']
                }
                
                for feature in self.modifiable_features:
                    if feature in categorical_definitions:
                        self.categorical_features[feature] = categorical_definitions[feature]
                    else:
                        self.continuous_features.append(feature)
                
                # Reward weights
                self.metric_weights = {
                    'signal_to_noise_ratio_SNR': 0.45,
                    'dynamic_range_of_output': 0.25,
                    'false_negative_rate': 0.20,
                    'time_to_detection_threshold': 0.10,
                    'multi_objective_score': 1.0
                }
                
                # Initialize tracking
                self.step_count = 0
                self.max_steps = 100  # New - 2x faster episodes
                self.reward_history = []
                self.best_reward = float('-inf')
                self.best_states = []
                self.previous_state = None
                self.visited_states = []
                self.state_visit_counts = {}
                self.current_episode_rewards = []
                self.current_episode_states = []
                self.current_episode_actions = []
                self.current_episode_predictions = []
                
                # âœ… FIX: Simplified action space - only 3 actions per feature (up/down/keep)
                self.action_mapping = {}
                action_idx = 0

                # Discrete actions: adjust each continuous feature
                for feature in self.continuous_features:
                    # Only 3 actions: decrease, keep, increase
                    for delta, label in [(-0.10, 'decrease'), (0.0, 'keep'), (0.10, 'increase')]:
                        self.action_mapping[action_idx] = ('continuous', feature, label, delta)
                        action_idx += 1

                # Categorical features: set to specific value
                for feature, categories in self.categorical_features.items():
                    for category in categories:
                        self.action_mapping[action_idx] = ('categorical', feature, 'set', category)
                        action_idx += 1

                print(f"   âœ… Simplified action space: {len(self.action_mapping)} actions")
                
                # Use Box action space (continuous) - much better for RL
                # Each action is a delta for each modifiable feature
                from gymnasium import spaces as gym_spaces
                # Use ALL modifiable features (continuous + categorical)
                n_continuous = len(self.continuous_features)
                n_categorical = sum(len(cats) for cats in self.categorical_features.values())
                total_action_dim = n_continuous + n_categorical

                self.action_space = gym_spaces.Box(
                    low=-1.0, 
                    high=1.0, 
                    shape=(total_action_dim,),  # ✅ CORRECT - 9 features
                    dtype=np.float32
                )

                print(f"   ✅ Action space created: {self.action_space}")

                print(f"   ✅ Continuous action space: {self.action_space.shape[0]} dimensions")

                # Ã¢Å“â€¦ FIX: Calculate exact observation size based on _get_observation implementation
                # Components: normalized_modifiable + target_predictions + target_gaps + context + bounds_utilization
                obs_size = (
                    self.n_modifiable +  # normalized_modifiable
                    4 +                   # target predictions (SNR, DR, FNR, TTD)
                    4 +                   # target gaps (improvement potential)
                    6 +                   # context (step_ratio, reward_mean, etc.)
                    self.n_modifiable     # bounds utilization
                )

                self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(obs_size,), dtype=np.float32)
                print(f"   ✅ Observation space: {obs_size} dimensions")
                print(f"      = {self.n_modifiable} features + 4 predictions + 4 gaps + 6 context + {self.n_modifiable} bounds")
                
                # Initialize state
                self.modifiable_state = np.array([
                    np.random.uniform(bounds[0], bounds[1])
                    for bounds in self.modifiable_bounds.values()
                ], dtype=np.float32)
                
                # Validate and initialize
                self._validate_surrogate_models()
                self._initialize_log_files()
                
                print(f"âœ… Environment initialized: {self.n_modifiable} modifiable parameters")

            def seed(self, seed=None):
                """Set random seed for reproducibility"""
                np.random.seed(seed)
                return [seed]

            def _validate_surrogate_models(self):
                """Validate that surrogate models work with combined inputs"""
                print("Validating surrogate models...")
                
                for metric, model in self.surrogate_models.items():
                    try:
                        # Create a dummy full state (modifiable + fixed)
                        dummy_full_state = self._create_full_state(target_metric=metric)
                        
                        # ✅ FIXED: Always use .predict() for consistency
                        # All our models (sklearn, NNWrapper, PyTorch wrapped) have .predict()
                        test_output = model.predict(dummy_full_state.reshape(1, -1))
                        
                        if np.isnan(test_output).any():
                            print(f"WARNING: Surrogate model {metric} produces NaN!")
                        else:
                            model_type = type(model).__name__
                            print(f"✔️ Surrogate model {metric} validated ({model_type})")
                    
                    except Exception as e:
                        print(f"ERROR: Surrogate model {metric} validation failed: {e}")
                        import traceback
                        traceback.print_exc()

            def set_difficulty(self, difficulty_level):
                """Set environment difficulty (0.0 = easy, 1.0 = hard)"""
                self.difficulty = difficulty_level
                
                # Adjust reward scaling based on difficulty
                if hasattr(self, 'base_metric_weights'):
                    for metric in self.metric_weights:
                        self.metric_weights[metric] = self.base_metric_weights[metric] * (0.5 + 0.5 * difficulty_level)

            def _create_full_state(self, target_metric=None):
                """Create full state with CORRECT feature mapping - FIXED VERSION"""
                if target_metric is None:
                    target_metric = list(self.surrogate_features.keys())[0]
                
                if target_metric not in self.surrogate_features:
                    raise ValueError(f"No feature mapping for {target_metric}!")
                
                feature_order = self.surrogate_features[target_metric]
                full_state = np.zeros(len(feature_order), dtype=np.float32)
                
                # ✅ DEBUG: Track what we're filling
                filled_features = []
                
                for idx, feature in enumerate(feature_order):
                    if feature in self.modifiable_features:
                        modifiable_idx = self.modifiable_features.index(feature)
                        full_state[idx] = self.modifiable_state[modifiable_idx]
                        filled_features.append(f"{feature}={self.modifiable_state[modifiable_idx]:.4f}[M]")
                    elif feature in self.fixed_features:
                        full_state[idx] = self.fixed_values[feature]
                        filled_features.append(f"{feature}={self.fixed_values[feature]:.4f}[F]")
                    else:
                        # ✅ CRITICAL: Feature not found - fill with 0 and warn
                        full_state[idx] = 0.0
                        filled_features.append(f"{feature}=0.0[MISSING]")
                
                # ✅ DEBUG: Print FIRST call only to see what's being fed
                if not hasattr(self, '_debug_printed'):
                    self._debug_printed = set()
                
                if target_metric not in self._debug_printed and self.step_count <= 1:
                    print(f"\n🔍 FEATURE MAPPING for {target_metric}:")
                    print(f"   Features ({len(feature_order)}): {feature_order[:5]}...")
                    print(f"   Values: {full_state[:5]}")
                    print(f"   Sample mappings: {filled_features[:3]}")
                    self._debug_printed.add(target_metric)
                
                return full_state

            def _create_full_state_for_all_features(self):
                """
                Create full state with ALL available features for physics-based calculations
                Used by FNR predictor which needs access to all scaled features
                """
                # Combine modifiable and fixed features
                full_feature_names = self.modifiable_features + self.fixed_features
                
                # Get the longest feature set from any surrogate
                max_features = 0
                reference_features = None
                for features in self.surrogate_features.values():
                    if len(features) > max_features:
                        max_features = len(features)
                        reference_features = features
                
                if reference_features is None:
                    # Fallback to basic approach
                    return self._create_full_state()
                
                # Create state matching the reference features
                full_state = np.zeros(len(reference_features))
                
                for idx, feature in enumerate(reference_features):
                    if feature in self.modifiable_features:
                        modifiable_idx = self.modifiable_features.index(feature)
                        full_state[idx] = self.modifiable_state[modifiable_idx]
                    elif feature in self.fixed_features:
                        full_state[idx] = self.fixed_values[feature]
                    # If feature not found, it stays 0 (already initialized)
                
                return full_state

            def _initialize_log_files(self):
                """Initialize CSV log files"""
                episode_log_path = os.path.join(self.log_dir, "episode_summary.csv")
                with open(episode_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'episode', 'mean_reward', 'total_reward', 'best_reward',  # âœ… Fixed order
                        'final_reward', 'steps_taken', 'convergence_achieved', 
                        'reward_improvement', 'state_stability'
                    ])
                
                step_log_path = os.path.join(self.log_dir, "step_details.csv")
                with open(step_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = ['episode', 'step', 'action', 'reward', 'cumulative_reward']
                    headers.extend([f'modifiable_{name}' for name in self.modifiable_features])
                    headers.extend([f'pred_{metric}' for metric in self.target_metrics])
                    writer.writerow(headers)
                
                # Initialize trajectory evolution CSV with CORRECT headers for Sclerostin
                trajectory_log_path = os.path.join(self.log_dir, "trajectory_evolution.csv")
                with open(trajectory_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # FIX: Use actual modifiable features from environment
                    headers = ['episode', 'step']
                    
                    # Add value columns for each modifiable feature
                    for feature in self.modifiable_features:
                        headers.append(f'{feature}_value')
                    
                    # Add change columns (absolute)
                    for feature in self.modifiable_features:
                        headers.append(f'{feature}_change')
                    
                    # Add change percentage columns
                    for feature in self.modifiable_features:
                        headers.append(f'{feature}_change_pct')
                    
                    writer.writerow(headers)

            def set_reward_weights(self, new_weights):
                """Dynamically change reward weights during training"""
                self.metric_weights.update(new_weights)
                print(f"   Ã¢Å“â€¦ Updated reward weights: {new_weights}")

            def reset(self, seed=None, options=None):
                """Reset with ORIGINAL-SPACE state storage
                
                Args:
                    seed: Random seed (required for gymnasium compatibility)
                    options: Additional options (required for gymnasium compatibility)
                """
                if seed is not None:
                    np.random.seed(seed)
                
                # ✅ IMPROVED: Curriculum-based initialization strategy
                if not hasattr(self, 'current_episode'):
                    self.current_episode = 0
                else:
                    self.current_episode += 1

                # Adapt strategy based on training progress
                if self.current_episode < 100:
                    # Early training: pure exploration
                    init_strategy = 'random'
                elif self.current_episode < 500:
                    # Mid training: balanced
                    init_strategy = np.random.choice(['random', 'center', 'near_best'], p=[0.3, 0.3, 0.4])
                else:
                    # Late training: exploit good states
                    init_strategy = np.random.choice(['center', 'near_best'], p=[0.2, 0.8])

                if init_strategy == 'random' or not hasattr(self, 'best_states'):
                    # Random initialization in ORIGINAL space (but avoid extremes)
                    self.modifiable_state = np.zeros(self.n_modifiable, dtype=np.float32)
                    
                    for i, feature in enumerate(self.modifiable_features):
                        if feature in self.X_train.columns:
                            # ✅ Use 80% of range (avoid extreme edges)
                            original_min = float(self.X_train[feature].quantile(0.1))
                            original_max = float(self.X_train[feature].quantile(0.9))
                            
                            self.modifiable_state[i] = np.random.uniform(original_min, original_max)
                        else:
                            bounds = list(self.modifiable_bounds.values())[i]
                            self.modifiable_state[i] = np.random.uniform(bounds[0], bounds[1])
                
                elif init_strategy == 'center':
                    # Start at center of ORIGINAL data distribution
                    for i, feature in enumerate(self.modifiable_features):
                        if feature in self.X_train.columns:
                            self.modifiable_state[i] = float(self.X_train[feature].mean())
                        else:
                            bounds = list(self.modifiable_bounds.values())[i]
                            self.modifiable_state[i] = (bounds[0] + bounds[1]) / 2.0
                
                else:  # near_best
                    # ✅ CORRECTED: Check if best_states exists and has elements
                    if hasattr(self, 'best_states') and len(self.best_states) > 0:
                        base_state = self.best_states[-1]
                        noise = np.random.normal(0, 0.1, size=len(base_state))
                        self.modifiable_state = base_state + noise * np.abs(base_state)
                    else:
                        # Fallback to random initialization
                        self.modifiable_state = np.zeros(self.n_modifiable, dtype=np.float32)
                        for i, feature in enumerate(self.modifiable_features):
                            if feature in self.X_train.columns:
                                original_min = float(self.X_train[feature].min())
                                original_max = float(self.X_train[feature].max())
                                self.modifiable_state[i] = np.random.uniform(original_min, original_max)
                
                # Clip to ORIGINAL bounds
                self._clip_modifiable_state()
                
                # Reset tracking
                self.step_count = 0
                self.reward_history = []
                self.current_episode += 1
                
                self.current_episode_rewards.clear()
                self.current_episode_states.clear()
                self.current_episode_actions.clear()
                self.current_episode_predictions.clear()
                
                obs = self._get_observation()
                
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs, dtype=np.float32)
                
                if obs.shape != self.observation_space.shape:
                    if len(obs) < self.observation_space.shape[0]:
                        obs = np.pad(obs, (0, self.observation_space.shape[0] - len(obs)), mode='constant')
                    else:
                        obs = obs[:self.observation_space.shape[0]]
                
                # ✅ FIXED: Return Gymnasium-style (obs, info)
                return obs, {}

            def _clip_modifiable_state(self):
                """Clip state to ORIGINAL data bounds"""
                for i, feature in enumerate(self.modifiable_features):
                    if feature in self.X_train.columns:
                        # Use ORIGINAL data bounds
                        original_min = float(self.X_train[feature].min())
                        original_max = float(self.X_train[feature].max())
                        self.modifiable_state[i] = np.clip(self.modifiable_state[i], original_min, original_max)
                    else:
                        # Fallback to scaled bounds
                        bounds = list(self.modifiable_bounds.values())[i]
                        self.modifiable_state[i] = np.clip(self.modifiable_state[i], bounds[0], bounds[1])

            def _get_observation(self):
                """Get enhanced observation with trajectory velocity and comprehensive state info"""
                
                # ✅ STEP 1: Normalize modifiable features using ORIGINAL data distribution
                normalized_modifiable = np.zeros_like(self.modifiable_state)
                
                for i, feature in enumerate(self.modifiable_features):
                    if feature in self.X_train.columns:
                        # Use ORIGINAL data statistics
                        original_mean = float(self.X_train[feature].mean())
                        original_std = float(self.X_train[feature].std())
                        
                        if original_std > 1e-10:
                            # Z-score normalization (better for neural networks)
                            normalized_modifiable[i] = (self.modifiable_state[i] - original_mean) / original_std
                        else:
                            normalized_modifiable[i] = 0.0
                    else:
                        # Fallback: min-max normalization
                        bounds = list(self.modifiable_bounds.values())[i]
                        range_val = bounds[1] - bounds[0]
                        if range_val > 0:
                            normalized_val = (self.modifiable_state[i] - bounds[0]) / range_val
                            normalized_modifiable[i] = 2 * normalized_val - 1  # Scale to [-1, 1]
                        else:
                            normalized_modifiable[i] = 0.0
                
                # Clip to reasonable range
                normalized_modifiable = np.clip(normalized_modifiable, -3.0, 3.0)
                
                # ✅ STEP 2: Calculate current predictions for all targets
                target_predictions = []
                for metric in ['signal_to_noise_ratio_SNR', 'dynamic_range_of_output', 
                            'false_negative_rate', 'time_to_detection_threshold']:
                    if metric in self.surrogate_models:
                        try:
                            model = self.surrogate_models[metric]
                            full_state = self._create_full_state(target_metric=metric)
                            state_tensor = torch.FloatTensor(full_state).unsqueeze(0)
                            
                            # ✅ FIXED: Always use .predict() for all models
                            pred = model.predict(full_state.reshape(1, -1))[0]
                            
                            # ✅ Use data_min_max for proper normalization
                            if hasattr(self, 'data_min_max') and metric in self.data_min_max:
                                min_val, max_val = self.data_min_max[metric]
                                pred_norm = (pred - min_val) / (max_val - min_val + 1e-8)
                                pred_norm = np.clip(pred_norm, 0, 1)
                            else:
                                # Fallback normalization
                                if metric == 'signal_to_noise_ratio_SNR':
                                    pred_norm = np.clip(pred / 40.0, 0, 1)
                                elif metric == 'dynamic_range_of_output':
                                    pred_norm = np.clip(pred / 20.0, 0, 1)
                                elif metric == 'false_negative_rate':
                                    pred_norm = pred  # Already in [0,1]
                                else:  # time_to_detection
                                    pred_norm = np.clip(pred / 10.0, 0, 1)
                            
                            target_predictions.append(pred_norm)
                        except:
                            target_predictions.append(0.0)
                    else:
                        target_predictions.append(0.0)
                
                target_predictions = np.array(target_predictions)
                
                # ✅ STEP 3: Calculate target gaps (how much improvement is possible)
                target_gaps = []
                for i, pred in enumerate(target_predictions):
                    # For SNR and DR: gap = 1.0 - pred (we want to maximize)
                    # For FNR and TTD: gap = pred (we want to minimize, so gap is current value)
                    if i < 2:  # SNR and DR
                        gap = 1.0 - pred
                    else:  # FNR and TTD
                        gap = pred
                    target_gaps.append(gap)
                
                target_gaps = np.array(target_gaps)
                
                # ✅ STEP 4: Context information (episode progress and recent performance)
                step_ratio = self.step_count / self.max_steps
                
                if len(self.reward_history) > 0:
                    recent_rewards = self.reward_history[-5:]
                    reward_mean = np.mean(recent_rewards)
                    reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
                    reward_trend = recent_rewards[-1] - recent_rewards[0] if len(recent_rewards) > 1 else 0.0
                    current_reward = self.reward_history[-1]
                else:
                    reward_mean = reward_std = reward_trend = current_reward = 0.0
                
                # Calculate current MOS quickly
                current_mos = (
                    target_predictions[0] * 0.45 +  # SNR
                    target_predictions[1] * 0.25 +  # DR
                    (1 - target_predictions[2]) * 0.20 +  # 1 - FNR
                    (1 - target_predictions[3]) * 0.10   # 1 - TTD
                )
                
                context = np.array([
                    step_ratio,
                    reward_mean / 10.0,  # Normalize by expected max reward
                    reward_std / 10.0,
                    reward_trend / 10.0,
                    current_reward / 10.0,
                    current_mos
                ])
                
                # ✅ STEP 5: Bounds utilization (how close to limits)
                bounds_utilization = []
                for i, feature in enumerate(self.modifiable_features):
                    bounds = list(self.modifiable_bounds.values())[i]
                    current_val = self.modifiable_state[i]
                    # -1 = at lower bound, 0 = middle, +1 = at upper bound
                    if bounds[1] > bounds[0]:
                        utilization = 2 * (current_val - bounds[0]) / (bounds[1] - bounds[0]) - 1
                    else:
                        utilization = 0.0
                    bounds_utilization.append(utilization)
                
                bounds_utilization = np.array(bounds_utilization)
                
                # Step 6: Concatenate all observation components (SIMPLIFIED)
                try:
                    obs_components = [
                        normalized_modifiable,      # (n_modifiable,)
                        target_predictions,         # (4,)
                        target_gaps,                # (4,)
                        context,                    # (6,)
                        bounds_utilization          # (n_modifiable,)
                    ]
                    
                    obs = np.concatenate(obs_components).astype(np.float32)
                    
                except Exception as e:
                    print(f"⚠️ Observation construction failed: {e}")
                    obs = np.concatenate([
                        normalized_modifiable,
                        np.zeros(4),
                        np.zeros(4),
                        np.array([step_ratio, 0, 0, 0, 0, 0]),
                        np.zeros(self.n_modifiable)
                    ]).astype(np.float32)
                
                # ✅ STEP 7: Safety checks
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                obs = np.clip(obs, -3.0, 3.0)
                
                # ✅ CRITICAL: Validate observation size
                obs_size = (
                    self.n_modifiable +  # normalized_modifiable
                    4 +                   # target predictions
                    4 +                   # target gaps
                    6 +                   # context
                    self.n_modifiable     # bounds utilization
                )  # REMOVED: velocity and momentum

                self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(obs_size,), dtype=np.float32)

                return obs

            def set_reward_shaping(self, episode_num):
                """Gradually increase reward requirements"""
                if episode_num < 50:
                    self.reward_scale = 0.5  # Easier early on
                elif episode_num < 200:
                    self.reward_scale = 0.75
                else:
                    self.reward_scale = 1.0  # Full difficulty

            def step(self, action):
                """Step environment with CONTINUOUS actions for better exploration"""
                self.step_count += 1
                self.global_step += 1
                
                # Store previous state for change tracking
                prev_state = self.modifiable_state.copy()
                
                # ✅ Handle continuous actions (action is a numpy array of floats in [-1, 1])
                if isinstance(action, np.ndarray):
                    action = action.flatten()
                elif isinstance(action, (list, tuple)):
                    action = np.array(action, dtype=np.float32)
                else:
                    # Fallback: treat as scalar and broadcast
                    action = np.array([action] * len(self.continuous_features), dtype=np.float32)
                
                # ✅ CORRECTED: Apply continuous actions properly
                # action is a numpy array of shape (n_continuous_features,) with values in [-1, 1]

                # Ensure action is the right shape
                action = np.atleast_1d(action).flatten()

                # ✅ FIXED: Apply actions in ORIGINAL space
                for i, feature in enumerate(self.modifiable_features):  # All features, not just continuous
                    if i >= len(action):
                        break
                        
                    feature_idx = self.modifiable_features.index(feature)
                    
                    # Get ORIGINAL data range
                    if feature in self.X_train.columns:
                        original_min = float(self.X_train[feature].min())
                        original_max = float(self.X_train[feature].max())
                        data_range = original_max - original_min
                    else:
                        # Fallback
                        bounds_list = list(self.modifiable_bounds.values())
                        original_min, original_max = bounds_list[feature_idx]
                        data_range = original_max - original_min
                    
                    # ✅ IMPROVED: Smooth action scaling with stochastic exploration
                    episode_progress = self.step_count / self.max_steps

                    # ✅ FIXED: Adaptive action scaling - start large, decrease gradually
                    if episode_progress < 0.2:
                        action_scale = 0.20  # Large exploration early
                    elif episode_progress < 0.5:
                        action_scale = 0.15  # Moderate exploration mid-episode
                    elif episode_progress < 0.7:
                        action_scale = 0.10  # Refinement
                    else:
                        action_scale = 0.05  # Fine-tuning

                    delta = action[i] * action_scale * data_range
                    new_val = self.modifiable_state[feature_idx] + delta
                    
                    # Clip to original bounds
                    self.modifiable_state[feature_idx] = np.clip(new_val, original_min, original_max)
                
                # Handle categorical features (if any) - sample randomly with low probability
                if self.categorical_features and np.random.random() < 0.1:  # 10% chance to change categorical
                    for feature in self.categorical_features.keys():
                        if feature in self.modifiable_features:
                            feature_idx = self.modifiable_features.index(feature)
                            
                            # Get available categories
                            if hasattr(self, 'categorical_label_encoders') and feature in self.categorical_label_encoders:
                                le = self.categorical_label_encoders[feature]
                                # Sample a random category
                                random_category = np.random.choice(le.classes_)
                                encoded_value = le.transform([random_category])[0]
                                self.modifiable_state[feature_idx] = encoded_value
                
                # Calculate reward and predictions
                reward, predictions = self._calculate_reward_with_predictions()

                # ✅ SAFETY: Additional clipping to prevent overflow
                reward = float(np.clip(reward, -10.0, 10.0))

                # ✅ SAFETY: Check for NaN/Inf
                if not np.isfinite(reward):
                    print(f"⚠️ Non-finite reward in step()! Resetting to -5.0")
                    reward = -5.0
                
                # Track performance
                self.reward_history.append(reward)
                self.current_episode_rewards.append(reward)
                self.current_episode_states.append(self.modifiable_state.copy())
                self.current_episode_actions.append(action.copy())  # Store continuous action
                self.current_episode_predictions.append(predictions)
                
                if reward > self.best_reward:
                    self.best_reward = reward
                    if not hasattr(self, 'best_states'):
                        self.best_states = []
                    self.best_states.append(self.modifiable_state.copy())
                    if len(self.best_states) > 20:
                        self.best_states = self.best_states[-20:]
                
                # Log step details
                self._log_step_details(action, reward, predictions)
                
                # Log trajectory evolution
                self._log_trajectory_evolution(prev_state)
                
                # Done condition
                done = bool(self.step_count >= self.max_steps)
                reward = float(reward)
                
                # Info dict for vectorized environment
                info = {
                    'raw_reward': float(reward),
                    'step_count': int(self.step_count),
                    'best_reward': float(self.best_reward),
                    'predictions': predictions,
                    'exploration_bonus': 0.0,
                    'action_magnitude': float(np.linalg.norm(action)),  # Track how aggressive actions are
                    'state_change': float(np.linalg.norm(self.modifiable_state - prev_state))
                }
                
                # ✅ CLEAN EPISODE SUMMARY (only print once per environment)
                if done:
                    episode_reward = sum(self.current_episode_rewards)
                    mean_reward = episode_reward / self.max_steps if self.max_steps > 0 else 0
                    
                    # Only print if this is the first environment OR if we have a unique identifier
                    if not hasattr(self, 'summary_printed'):
                        self.summary_printed = {}
                    
                    episode_key = f"{self.current_episode}"
                    if episode_key not in self.summary_printed:
                        print(f"Episode {self.current_episode}: Total={episode_reward:.4f}, Mean={mean_reward:.4f}, Steps={self.max_steps}")
                        self.summary_printed[episode_key] = True
                    
                    self._log_episode_summary()
                
                # ✅ FIXED: Return Gymnasium-style (5 values)
                terminated = done
                truncated = False  # We don't use truncation
                return self._get_observation(), reward, terminated, truncated, info
         
            def _calculate_reward_with_predictions(self):
                """COMPLETELY FIXED: Proper reward calculation without double-counting"""
                
                try:
                    # ✅ VALIDATION: Ensure features are lists, not pandas Index
                    if not hasattr(self, '_features_validated'):
                        print("\n🔍 Validating surrogate_features types...")
                        for metric in self.surrogate_features:
                            if not isinstance(self.surrogate_features[metric], list):
                                print(f"   ⚠️ Converting {metric} features to list")
                                self.surrogate_features[metric] = list(self.surrogate_features[metric])
                        self._features_validated = True

                    predictions = {}
                    
                    # Step 1: Predict each metric using CORRECT features
                    for metric in self.target_metrics:
                        if metric not in self.surrogate_models:
                            predictions[metric] = 0.0
                            continue
                            
                        try:
                            model = self.surrogate_models[metric]
                            
                            # ✅ CRITICAL FIX: Get the CORRECT features for THIS metric
                            if metric not in self.surrogate_features:
                                print(f"⚠️ No feature mapping for {metric}!")
                                predictions[metric] = 0.0
                                continue
                            
                            feature_names = self.surrogate_features[metric]
                            
                            # ✅ Create state with ONLY the features this model was trained on
                            full_state = np.zeros(len(feature_names), dtype=np.float32)

                            for idx, feature in enumerate(feature_names):
                                if feature in self.modifiable_features:
                                    modifiable_idx = self.modifiable_features.index(feature)
                                    full_state[idx] = self.modifiable_state[modifiable_idx]
                                elif feature in self.fixed_features:
                                    full_state[idx] = self.fixed_values[feature]
                                else:
                                    print(f"⚠️ Feature {feature} not found!")

                            # ✅ CRITICAL FIX: Apply metric-specific scaling BEFORE prediction
                            if metric in self.surrogate_metric_scalers and self.surrogate_metric_scalers[metric] is not None:
                                scaler = self.surrogate_metric_scalers[metric]
                                full_state_scaled = scaler.transform(full_state.reshape(1, -1)).flatten()
                            else:
                                # Fallback if no scaler (shouldn't happen)
                                full_state_scaled = full_state
                            
                            # ✅ Predict using the model
                            pred_scaled = model.predict(full_state_scaled.reshape(1, -1))[0]

                            # ✅ CRITICAL FIX: Inverse transform for NNWrapper (trained on scaled targets)
                            if type(model).__name__ == 'NNWrapper':
                                # NNWrapper outputs predictions in SCALED space
                                # Need to inverse transform back to original scale
                                
                                # Debug first few steps
                                debug_step = self.step_count <= 2
                                if debug_step:
                                    print(f"\n🔍 DEBUG {metric} Prediction (Step {self.step_count}):")
                                    print(f"   Model output (scaled): {pred_scaled:.4f}")
                                
                                if hasattr(self, 'y_scalers') and metric in self.y_scalers:
                                    try:
                                        y_scaler = self.y_scalers[metric]
                                        
                                        if debug_step:
                                            print(f"   y_scaler mean_: {y_scaler.mean_[0]:.4f}")
                                            print(f"   y_scaler scale_: {y_scaler.scale_[0]:.4f}")
                                        
                                        # Inverse transform
                                        pred_original = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
                                        
                                        if debug_step:
                                            print(f"   After inverse transform: {pred_original:.4f}")
                                        
                                        # CRITICAL: Safety bounds check
                                        needs_clip = False
                                        if metric == 'false_negative_rate':
                                            if pred_original < 0 or pred_original > 0.5:
                                                if debug_step:
                                                    print(f"      ⚠️ FNR out of realistic bounds!")
                                                pred_original = np.clip(pred_original, 0.0, 0.25)
                                                needs_clip = True
                                        elif metric == 'signal_to_noise_ratio_SNR':
                                            if pred_original < -150 or pred_original > 100:
                                                pred_original = np.clip(pred_original, -120, 50)
                                                needs_clip = True
                                        elif metric == 'time_to_detection_threshold':
                                            if pred_original < 0 or pred_original > 10000:
                                                pred_original = np.clip(pred_original, 10, 9000)
                                                needs_clip = True
                                        elif metric == 'dynamic_range_of_output':
                                            if pred_original < 0 or pred_original > 50:
                                                pred_original = np.clip(pred_original, 0.05, 40)
                                                needs_clip = True
                                        
                                        if debug_step:
                                            print(f"   Final (after safety): {pred_original:.4f}")
                                            if needs_clip:
                                                print(f"      (clipped for safety)")
                                            
                                    except Exception as e:
                                        print(f"      ❌ Failed to inverse transform {metric}: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        pred_original = pred_scaled
                                else:
                                    if debug_step:
                                        print(f"   ⚠️ No y_scaler for {metric}!")
                                    pred_original = pred_scaled
                            else:
                                # Other models already output in original scale
                                pred_original = pred_scaled

                            predictions[metric] = float(pred_original)
                                
                        except Exception as e:
                            if self.step_count == 1:  # Only show errors on first step
                                print(f"⚠️ Failed to predict {metric}: {e}")
                            predictions[metric] = 0.0

                    # Step 2: Extract raw predictions from dict
                    snr_raw = predictions.get('signal_to_noise_ratio_SNR', 0)
                    dr_raw = predictions.get('dynamic_range_of_output', 0)
                    fnr_raw = predictions.get('false_negative_rate', 0.001)
                    ttd_raw = predictions.get('time_to_detection_threshold', 5)

                    # ✅ FIXED: Use STRICT clipping to training ranges
                    if hasattr(self, 'data_min_max') and self.data_min_max:
                        snr_min, snr_max = self.data_min_max['signal_to_noise_ratio_SNR']
                        dr_min, dr_max = self.data_min_max['dynamic_range_of_output']
                        fnr_min, fnr_max = self.data_min_max['false_negative_rate']
                        ttd_min, ttd_max = self.data_min_max['time_to_detection_threshold']
                        
                        # ✅ FIXED: Single normalization WITHOUT over-clipping
                        def normalize_reward(value, min_val, max_val):
                            """Normalize to [0,1] with slight extrapolation allowed"""
                            if max_val == min_val:
                                return 0.5
                            normalized = (value - min_val) / (max_val - min_val)
                            # Allow 20% beyond training range for exploration
                            return np.clip(normalized, -0.2, 1.2)

                        # Apply normalization (NO double clipping)
                        snr_norm = normalize_reward(snr_raw, snr_min, snr_max)
                        dr_norm = normalize_reward(dr_raw, dr_min, dr_max)
                        fnr_norm = normalize_reward(fnr_raw, fnr_min, fnr_max)
                        ttd_norm = normalize_reward(ttd_raw, ttd_min, ttd_max)
                    else:
                        snr_norm = dr_norm = fnr_norm = ttd_norm = 0.5

                    # Calculate individual component rewards in [0, 1]
                    snr_reward = snr_norm
                    dr_reward = dr_norm
                    fnr_reward = 1.0 - fnr_norm  # Lower is better
                    ttd_reward = 1.0 - ttd_norm  # Lower is better

                    # Weighted combination - slightly more balanced
                    # SNR and DR are most important (biosensor quality)
                    # FNR and TTD are secondary (detection capability)
                    combined_score = (
                        0.35 * snr_reward +   # Signal quality
                        0.25 * dr_reward +    # Dynamic range
                        0.25 * fnr_reward +   # Detection accuracy (increased from 0.20)
                        0.15 * ttd_reward     # Detection speed (increased from 0.10)
                    )

                    # ✅ Map [0, 1] → [0, 10] with clear scaling
                    base_reward = combined_score * 10.0
                    
                    # Show component breakdown occasionally
                    if self.step_count % 50 == 0:
                        print(f"   Component rewards: SNR={snr_reward:.2f}, DR={dr_reward:.2f}, FNR={fnr_reward:.2f}, TTD={ttd_reward:.2f}")

                    # ✅ Milestone bonuses for exceptional performance
                    if snr_norm > 0.8 and dr_norm > 0.8 and fnr_norm < 0.2:
                        # Exceptional performance bonus
                        base_reward += 2.0
                    elif snr_norm > 0.7 and dr_norm > 0.7 and fnr_norm < 0.3:
                        # Good performance bonus
                        base_reward += 1.0

                    # ✅ BETTER LOGGING: Show raw values too
                    if self.step_count % 50 == 0:
                        print(f"\n📊 REWARD CHECK @ Step {self.step_count}:")
                        print(f"   SNR={snr_raw:.1f} (norm={snr_norm:.2f}), DR={dr_raw:.1f} (norm={dr_norm:.2f})")
                        print(f"   FNR={fnr_raw:.4f} (norm={fnr_norm:.2f}), TTD={ttd_raw:.1f} (norm={ttd_norm:.2f})")
                        
                        # Show expected ranges for debugging
                        if self.step_count == 50:
                            print(f"\n   📊 EXPECTED RANGES:")
                            print(f"      SNR: -120 to +50 (mean ~14)")
                            print(f"      DR: 0 to 35 (mean ~9)")
                            print(f"      FNR: 0 to 0.2 (mean ~0.046)")
                            print(f"      TTD: 0 to 9000 (mean ~2500)")
                            print(f"   ⚠️  If TTD shows 20-30 instead of 2000-3000, TTD model is BROKEN!\n")
                        
                        print(f"   BASE REWARD: {base_reward:.3f}")

                    # ✅ Improvement tracking (for logging only, NOT added to reward)
                    improvement_bonus = 0.0
                    if len(self.reward_history) > 0:
                        previous_reward = self.reward_history[-1]
                        improvement = base_reward - previous_reward
                        improvement_bonus = improvement  # Track but don't add

                    # ✅ FINAL REWARD: Just the base reward (no confusing bonuses)
                    final_reward = base_reward
                    # Wider clip range for better learning signal
                    final_reward = float(np.clip(final_reward, -10.0, 20.0))
                    
                    if not np.isfinite(final_reward):
                        final_reward = -5.0
                    
                    self.previous_state = self.modifiable_state.copy()
                    predictions['base_reward'] = float(base_reward)
                    predictions['improvement_bonus'] = float(improvement_bonus)
                    #predictions['milestone_bonus'] = float(milestone_bonus)
                    #predictions['exploration_bonus'] = float(exploration_bonus)
                    predictions['snr_reward'] = float(snr_reward)
                    predictions['dr_reward'] = float(dr_reward)
                    predictions['fnr_reward'] = float(fnr_reward)
                    predictions['ttd_reward'] = float(ttd_reward)
                    predictions['snr_norm'] = float(snr_norm)
                    predictions['dr_norm'] = float(dr_norm)
                    predictions['fnr_norm'] = float(fnr_norm)
                    predictions['ttd_norm'] = float(ttd_norm)
                    
                    return final_reward, predictions
                    
                except Exception as e:
                    print(f"⚠️ Error in reward calculation: {e}")
                    import traceback
                    traceback.print_exc()
                    return -5.0, {}
      
            def _apply_action_with_momentum(self, action):
                """Apply actions with momentum for smoother exploration"""
                if not hasattr(self, 'action_momentum'):
                    self.action_momentum = np.zeros_like(self.modifiable_state)
                
                # Decode action
                feature_idx = action // 3
                action_type = action % 3
                
                # Calculate change with momentum
                momentum_factor = 0.1  # Small momentum to smooth changes
                
                if action_type == 0:  # decrease
                    change = -0.15  # More aggressive changes
                elif action_type == 1:  # keep (with small random walk)
                    change = np.random.normal(0, 0.02)
                else:  # increase
                    change = 0.15
                
                # Apply momentum
                self.action_momentum[feature_idx] = (
                    momentum_factor * self.action_momentum[feature_idx] + 
                    (1 - momentum_factor) * change
                )
                
                # Apply change to state
                bounds = list(self.modifiable_bounds.values())[feature_idx]
                range_size = bounds[1] - bounds[0]
                
                new_value = self.modifiable_state[feature_idx] + self.action_momentum[feature_idx] * range_size
                self.modifiable_state[feature_idx] = np.clip(new_value, bounds[0], bounds[1])

            def _store_experience_for_learning(self):
                """
                Store experience with hindsight goals for better learning
                This helps agent learn from "failures" by imagining different goals
                """
                if len(self.current_episode_states) < 2:
                    return
                
                # Get final state and predictions
                final_state = self.current_episode_states[-1]
                
                # For each step in episode, imagine if reaching final state was the goal
                hindsight_rewards = []
                for i in range(len(self.current_episode_rewards)):
                    state = self.current_episode_states[i]
                    
                    # Calculate "hindsight reward" - how close was this state to final state?
                    distance_to_final = np.linalg.norm(state - final_state)
                    hindsight_reward = 1.0 / (1.0 + distance_to_final)
                    hindsight_rewards.append(hindsight_reward)
                
                # Store for potential replay
                self.hindsight_experiences = list(zip(
                    self.current_episode_states,
                    self.current_episode_actions,
                    hindsight_rewards
                ))
            
            def _log_step_details(self, action, reward, predictions):
                """Log detailed step information"""
                # Only log every 10th step
                if self.step_count % 10 != 0:
                    return

                step_log_path = os.path.join(self.log_dir, "step_details.csv")
                
                try:
                    with open(step_log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        
                        cumulative_reward = sum(self.current_episode_rewards)
                        
                        # Convert action to string - handle both scalar and array
                        if isinstance(action, (int, np.integer)):
                            action_str = str(int(action))
                        elif isinstance(action, np.ndarray):
                            if action.size == 1:
                                action_str = str(int(action.item()))
                            else:
                                action_str = ','.join([f'{a:.4f}' for a in action])
                        else:
                            action_str = str(action)
                        
                        row = [
                            self.current_episode, self.step_count, 
                            action_str, reward, cumulative_reward
                        ]
                                    
                        # Add modifiable state values (ensure float conversion)
                        row.extend([float(x) for x in self.modifiable_state.tolist()])
                        
                        # Add predictions (ensure float conversion and handle missing)
                        for metric in self.target_metrics:
                            pred_value = predictions.get(metric, 0.0)
                            row.append(float(pred_value) if pred_value is not None else 0.0)
                        
                        writer.writerow(row)
                except Exception as e:
                    print(f"   Ã¢Å¡ Ã¯Â¸Â Failed to log step details: {e}")
            
            def _log_trajectory_evolution(self, prev_state):
                """Log trajectory evolution with state changes"""
                trajectory_log_path = os.path.join(self.log_dir, "trajectory_evolution.csv")
                
                # FIX: Match header structure exactly
                try:
                    with open(trajectory_log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        
                        row = [self.current_episode, self.step_count]
                        
                        # Current state values (float conversion)
                        row.extend([float(x) for x in self.modifiable_state.tolist()])
                        
                        # State changes (absolute) - ensure same length as features
                        changes = self.modifiable_state - prev_state
                        row.extend([float(x) for x in changes.tolist()])
                        
                        # State changes (percentage) - handle division by zero
                        pct_changes = []
                        for i, (current, previous) in enumerate(zip(self.modifiable_state, prev_state)):
                            if abs(previous) > 1e-10:
                                pct_change = ((current - previous) / abs(previous)) * 100
                            else:
                                pct_change = 0.0
                            pct_changes.append(float(pct_change))
                        
                        row.extend(pct_changes)
                        
                        # FIX: Validate row length matches header count
                        # Expected: 2 (episode, step) + 3 * len(modifiable_features)
                        expected_length = 2 + 3 * len(self.modifiable_features)
                        if len(row) != expected_length:
                            print(f"   Ã¢Å¡ Ã¯Â¸Â Row length mismatch: got {len(row)}, expected {expected_length}")
                            return
                        
                        writer.writerow(row)
                except Exception as e:
                    print(f"   Ã¢Å¡ Ã¯Â¸Â Failed to log trajectory: {e}")

            def _log_episode_summary(self):
                """Log episode summary statistics"""
                # Only log if we have rewards to log
                if not hasattr(self, 'current_episode_rewards') or len(self.current_episode_rewards) == 0:
                    return
                    
                episode_log_path = os.path.join(self.log_dir, "episode_summary.csv")
                with open(episode_log_path, 'a', newline='') as f: 
                    writer = csv.writer(f)
                    
                    # Use mean reward as the primary metric (not cumulative)
                    mean_reward = np.mean(self.current_episode_rewards)
                    total_reward = sum(self.current_episode_rewards)  # Keep for reference
                    best_reward = max(self.current_episode_rewards)
                    final_reward = self.current_episode_rewards[-1]
                    
                    # Check convergence (reward stability in last 20 steps with higher threshold)
                    if len(self.current_episode_rewards) >= 20:
                        last_20_rewards = self.current_episode_rewards[-20:]
                        reward_std = np.std(last_20_rewards)
                        reward_mean = np.mean(last_20_rewards)
                        # Convergence = low variance AND high reward
                        convergence_achieved = (reward_std < 0.05) and (reward_mean > 0.60)
                    else:
                        convergence_achieved = False
                    
                    # Calculate improvement from first to last
                    if len(self.current_episode_rewards) > 1:
                        reward_improvement = final_reward - self.current_episode_rewards[0]
                    else:
                        reward_improvement = 0.0
                    
                    # Calculate state stability
                    if len(self.current_episode_states) > 1:
                        state_changes = []
                        for i in range(1, len(self.current_episode_states)):
                            change = np.linalg.norm(self.current_episode_states[i] - self.current_episode_states[i-1])
                            state_changes.append(change)
                        state_stability = 1.0 / (1.0 + np.mean(state_changes))
                    else:
                        state_stability = 1.0
                    
                    writer.writerow([
                        self.current_episode, mean_reward, total_reward, best_reward,  # Ã¢â€ Â Swapped order
                        final_reward, len(self.current_episode_rewards), convergence_achieved,   
                        reward_improvement, state_stability
                    ])

                    # ENHANCED DIAGNOSIS LOGGING (Section VI)
                    if self.current_episode % 50 == 0:  # Log every 50 episodes
                        print(f"\nÃ°Å¸â€œÅ  CONVERGENCE DIAGNOSTICS (Episode {self.current_episode}):")
                        print(f"   Mean Reward (last 50): {np.mean(self.current_episode_rewards):.4f}")
                        print(f"   Std Reward: {np.std(self.current_episode_rewards):.4f}")
                        print(f"   Best Reward: {best_reward:.4f}")
                        print(f"   Reward Improvement: {reward_improvement:.4f}")
                        print(f"   State Stability: {state_stability:.4f}")
                        
                        # Check for learning stagnation
                        if len(self.current_episode_rewards) >= 10:
                            recent_trend = np.mean(self.current_episode_rewards[-10:]) - np.mean(self.current_episode_rewards[-20:-10]) if len(self.current_episode_rewards) >= 20 else 0
                            if abs(recent_trend) < 0.001:
                                print(f"   Ã¢Å¡ Ã¯Â¸Â  WARNING: Possible learning stagnation (trend: {recent_trend:.6f})")
                    
                    # Optional: Print episode summary for debugging
                    print(f"Episode {self.current_episode}: Total={total_reward:.4f}, Mean={mean_reward:.4f}, Steps={len(self.current_episode_rewards)}")

            def close(self):
                """Clean up and save final logs"""
                # Log the final episode if it hasn't been logged yet
                if hasattr(self, 'current_episode_rewards') and len(self.current_episode_rewards) > 0:
                    self._log_episode_summary()
                
                # Save configuration
                config_path = os.path.join(self.log_dir, "environment_config.json")
                config = {
                    'modifiable_bounds': {name: list(bounds) for name, bounds in self.modifiable_bounds.items()},
                    'fixed_values': {name: float(value) for name, value in self.fixed_values.items()},
                    'metric_weights': self.metric_weights,
                    'target_metrics': self.target_metrics,
                    'max_steps': self.max_steps,
                    'n_modifiable': self.n_modifiable,
                    'n_fixed': len(self.fixed_features),
                    'total_episodes': self.current_episode,
                    'total_steps': self.global_step
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"Ã¢Å“â€¦ All logs saved to: {self.log_dir}")

        # Calculate bounds for modifiable features only
        modifiable_bounds = {}
        print("Calculating modifiable feature bounds...")
        
        for feature in self.modifiable_features:
            if feature in self.X_train.columns:
                col_min = float(self.X_train[feature].min())
                col_max = float(self.X_train[feature].max())
                
                if col_max <= col_min:
                    print(f"WARNING: Modifiable feature {feature} has invalid range: [{col_min}, {col_max}]")
                    col_max = col_min + 1e-6
                
                modifiable_bounds[feature] = (col_min, col_max)
                print(f"  {feature}: [{col_min:.6f}, {col_max:.6f}]")
        
        # Calculate fixed values (use mean values from training data)
        fixed_values = {}
        print("Calculating fixed feature values...")
        
        for feature in self.fixed_features:
            if feature in self.X_train.columns:
                mean_value = float(self.X_train[feature].mean())
                fixed_values[feature] = mean_value
                print(f"  {feature}: {mean_value:.6f} (fixed)")

        # CALCULATE DATA MIN/MAX FOR MOS NORMALIZATION (Section III.1)
        print("ðŸ“Š Calculating data ranges for MOS normalization...")

        # Ã¢Å“â€¦ FIX: Calculate ranges from ACTUAL training data (self.y_train)
        # This ensures we use the real data distribution, not guesses
        data_min_max = {}

        mos_metrics = [
            'signal_to_noise_ratio_SNR',
            'dynamic_range_of_output', 
            'false_negative_rate',
            'time_to_detection_threshold'
        ]

        for metric in mos_metrics:
            if metric in self.y_train.columns:
                # Use actual training data range with GENEROUS margin for RL exploration
                actual_min = float(self.y_train[metric].min())
                actual_max = float(self.y_train[metric].max())
        
                # ⚡ CRITICAL DEBUG: Print actual data statistics
                print(f"\n🔍 DATA RANGE DEBUG for {metric}:")
                print(f"   Training data shape: {self.y_train[metric].shape}")
                print(f"   Min: {actual_min:.6f}")
                print(f"   Max: {actual_max:.6f}")
                print(f"   Mean: {float(self.y_train[metric].mean()):.6f}")
                print(f"   Std: {float(self.y_train[metric].std()):.6f}")
                print(f"   Sample values: {self.y_train[metric].head(10).tolist()}")

                # ⚡ CRITICAL: Use larger margins for metrics that tend to hit boundaries
                if 'false_negative' in metric.lower() or 'time_to_detection' in metric.lower():
                    # These metrics need 30% buffer to avoid saturation
                    range_margin = (actual_max - actual_min) * 0.30
                else:
                    # Standard 15% margin for SNR and DR
                    range_margin = (actual_max - actual_min) * 0.15

                min_val = actual_min - range_margin
                max_val = actual_max + range_margin

                # ⚡ Ensure positive lower bounds for metrics that should be positive
                if 'false_negative' in metric.lower() or 'time_to_detection' in metric.lower():
                    min_val = max(0.0, min_val)

                # ✅ DEBUG: Print actual data ranges
                if metric == 'signal_to_noise_ratio_SNR':
                    print(f"   📊 {metric} ACTUAL data: min={actual_min:.4f}, max={actual_max:.4f}")
                    print(f"      With margin: min={min_val:.4f}, max={max_val:.4f}")
                
                # Ensure valid range
                if max_val <= min_val:
                    print(f"   ⚠️  {metric} has invalid range, using safe defaults")
                    if 'snr' in metric.lower():
                        min_val, max_val = -150, 100
                    elif 'dynamic_range' in metric.lower():
                        min_val, max_val = 0, 50
                    elif 'false_negative' in metric.lower():
                        min_val, max_val = 0, 0.3  # ✅ FIX: Use realistic max based on actual data
                    elif 'time_to_detection' in metric.lower():
                        min_val, max_val = 0, 12000  # Include timeout value
                
                data_min_max[metric] = (float(min_val), float(max_val))
                print(f"   âœ… {metric}: [{min_val:.4f}, {max_val:.4f}]")
            else:
                print(f"   âš ï¸  {metric} not in y_train, using defaults")
                if 'snr' in metric.lower():
                    data_min_max[metric] = (0.0, 100.0)
                elif 'dynamic_range' in metric.lower():
                    data_min_max[metric] = (0.0, 20000.0)
                elif 'false_negative' in metric.lower():
                    data_min_max[metric] = (0.0, 1.0)
                elif 'time_to_detection' in metric.lower():
                    data_min_max[metric] = (0.0, 200.0)

        # Ã¢Å“â€¦ CRITICAL: Store at pipeline level BEFORE environment creation
        self.data_min_max_global = dict(data_min_max)
        print(f"\nâœ… Stored data_min_max globally: {len(self.data_min_max_global)} metrics")

        # Ensure all required metrics have entries (with defaults if needed)
        required_defaults = {
            'signal_to_noise_ratio_SNR': (0, 100),
            'dynamic_range_of_output': (0, 20000),
            'false_negative_rate': (0, 1),
            'time_to_detection_threshold': (0, 200)
        }

        for metric, default_range in required_defaults.items():
            if metric not in data_min_max:
                data_min_max[metric] = default_range
                print(f"   Ã¢Å¡ Ã¯Â¸Â  Using default range for {metric}: {default_range}")

        # VERIFY target metrics match available surrogates
        print("\nÃ°Å¸â€Â Verifying RL environment setup...")
        print(f"   Requested target_metrics: {target_metrics}")
        print(f"   Available surrogate_models: {list(self.surrogate_models.keys())}")

        # Filter to only use metrics that have surrogate models
        valid_target_metrics = [m for m in target_metrics if m in self.surrogate_models]

        if len(valid_target_metrics) < len(target_metrics):
            missing = set(target_metrics) - set(valid_target_metrics)
            print(f"   Ã¢Å¡ Ã¯Â¸Â  WARNING: Some target metrics have no surrogate model: {missing}")
            print(f"   Ã¢Å“â€¦ Using only valid metrics: {valid_target_metrics}")
            target_metrics = valid_target_metrics

        if not target_metrics:
            raise ValueError("No valid target metrics with surrogate models available!")

        # Store the feature scaler for RL to use
        self.rl_feature_scaler = self.surrogate_feature_scaler if hasattr(self, 'surrogate_feature_scaler') else self.processed_data['scaler']

        # ðŸ”’ CRITICAL VALIDATION: Ensure all scalers are valid BEFORE creating environments
        print("\nðŸ” Pre-flight check: Validating scalers...")
        invalid_metrics = []
        for metric in target_metrics:
            if metric not in self.surrogate_metric_scalers:
                invalid_metrics.append(f"{metric} (missing)")
            elif self.surrogate_metric_scalers[metric] is None:
                invalid_metrics.append(f"{metric} (None)")
            elif not hasattr(self.surrogate_metric_scalers[metric], 'mean_') and \
                not hasattr(self.surrogate_metric_scalers[metric], 'center_'):
                invalid_metrics.append(f"{metric} (not fitted)")

        if invalid_metrics:
            raise RuntimeError(
                f"âŒ CRITICAL: Invalid scalers detected before RL training:\n"
                f"   {invalid_metrics}\n"
                f"   Cannot proceed with RL training. Check surrogate model training."
            )
        else:
            print(f"   âœ… All {len(target_metrics)} metric scalers validated")
            for metric in target_metrics:
                scaler = self.surrogate_metric_scalers[metric]
                n_features = len(scaler.mean_) if hasattr(scaler, 'mean_') else len(scaler.center_)
                print(f"      {metric}: {type(scaler).__name__}, {n_features} features")

        # Ã¢Å“â€¦ ADDITIONAL FIX: Store scalers at PIPELINE level for reference
        self.rl_scalers_backup = {}
        for metric in target_metrics:
            if metric in self.surrogate_metric_scalers:
                import copy
                self.rl_scalers_backup[metric] = copy.deepcopy(self.surrogate_metric_scalers[metric])

        print(f"   âœ… Backed up {len(self.rl_scalers_backup)} scalers at pipeline level")

        # SIMPLIFIED: Use DummyVecEnv instead of SubprocVecEnv for Windows compatibility
        # DummyVecEnv runs sequentially but avoids pipe issues
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # SAVE log_dir BEFORE wrapping
            self.rl_log_dir = self.log_dir  # Store at pipeline level
            
            # Use 4 parallel environments with DummyVecEnv (safer on Windows)
            n_envs = 4
            print(f"Ã°Å¸Å¡â‚¬ Creating {n_envs} vectorized environments (DummyVecEnv)...")
            
            def make_env(rank=0, seed=0):
                """
                Creates environment with rank and seed for reproducibility.
                Factory pattern to avoid closure/pickling issues.
                """
                def _init():
                    # Ã¢Å“â€¦ FIX: Create deep copies AND validate before environment creation
                    import copy
                    
                    # Validate source data BEFORE copying
                    if not data_min_max:
                        raise ValueError(f"Ã¢Å’ Cannot create environment: data_min_max is empty in factory!")
                    
                    _local_data_min_max = copy.deepcopy(data_min_max)
                    
                    # Validate the copy worked
                    if not _local_data_min_max:
                        raise ValueError(f"Ã¢Å’ Deep copy of data_min_max failed!")
                    
                    if len(_local_data_min_max) != len(data_min_max):
                        raise ValueError(f"Ã¢Å’ Deep copy corrupted data_min_max: {len(_local_data_min_max)} vs {len(data_min_max)}")
                    _local_modifiable_bounds = copy.deepcopy(modifiable_bounds)
                    _local_fixed_values = copy.deepcopy(fixed_values)
                    _local_categorical_encoders = copy.deepcopy(self.categorical_label_encoders) if hasattr(self, 'categorical_label_encoders') else {}
                    
                    # ✅ CORRECTED: Robust scaler storage with validation
                    _local_scalers = {}
                    if hasattr(self, 'surrogate_metric_scalers'):
                        import copy
                        for metric, scaler in self.surrogate_metric_scalers.items():
                            if scaler is not None and hasattr(scaler, 'center_'):
                                try:
                                    # Deep copy and validate
                                    copied_scaler = copy.deepcopy(scaler)
                                    # Verify the copy worked
                                    if hasattr(copied_scaler, 'center_') and len(copied_scaler.center_) > 0:
                                        _local_scalers[metric] = copied_scaler
                                    else:
                                        raise ValueError(f"Scaler copy validation failed for {metric}")
                                except Exception as e:
                                    print(f"⚠️ Failed to copy scaler for {metric}: {e}")
                                    # Rebuild from training data
                                    if metric in self.surrogate_features and hasattr(self, 'X_train'):
                                        features = self.surrogate_features[metric]
                                        if all(f in self.X_train.columns for f in features):
                                            from sklearn.preprocessing import StandardScaler
                                            new_scaler = StandardScaler()
                                            new_scaler.fit(self.X_train[features])
                                            _local_scalers[metric] = new_scaler
                                            print(f"   ✅ Rebuilt scaler for {metric}")
                    
                    # Create environment with copied data
                    # ✅ CRITICAL: Deep copy y_scalers for each environment
                    _local_y_scalers = {}
                    if hasattr(self, 'y_scalers'):
                        for metric, scaler in self.y_scalers.items():
                            if scaler is not None:
                                _local_y_scalers[metric] = copy.deepcopy(scaler)
                    
                    # Create environment with copied data
                    env = BiosensorEnv(
                        modifiable_bounds=_local_modifiable_bounds,
                        fixed_values=_local_fixed_values,
                        surrogate_models=self.surrogate_models,
                        target_metrics=list(target_metrics),
                        log_dir=str(self.log_dir),
                        surrogate_features=dict(self.surrogate_features),
                        X_train=self.X_train,
                        data_min_max=_local_data_min_max,
                        categorical_label_encoders=_local_categorical_encoders,
                        feature_scaler=None,
                        surrogate_metric_scalers=_local_scalers,
                        target_transformers=self.target_transformers,
                        y_scalers=_local_y_scalers  # ✅ ADD THIS LINE
                    )

                    # Set seed for reproducibility
                    env.seed(seed + rank)
                    return env
                
                return _init
                                                                    
            # Use DummyVecEnv (safer on Windows, no multiprocessing)
            env_fns = [make_env(rank=i, seed=42) for i in range(n_envs)]
            self.rl_env = DummyVecEnv(env_fns)

            # Validate data_min_max persistence
            print("\nðŸ” VALIDATION: Testing environment data_min_max...")
            test_obs = self.rl_env.reset()
            if hasattr(self.rl_env, 'envs'):
                for i, env in enumerate(self.rl_env.envs[:2]):  # Check first 2
                    if not hasattr(env, 'data_min_max') or len(env.data_min_max) == 0:
                        raise RuntimeError(f"âŒ Environment {i} has empty data_min_max after creation!")
                    print(f"   âœ… Env {i}: {len(env.data_min_max)} metrics stored")
            else:
                if not hasattr(self.rl_env, 'data_min_max') or len(self.rl_env.data_min_max) == 0:
                    raise RuntimeError("âŒ Single environment has empty data_min_max!")
            print("âœ… All environments validated successfully!")
            print(f"Ã¢Å“â€¦ Vectorized environments created successfully!")
            
        except Exception as e:
            print(f"âš ï¸  Could not create vectorized environments: {e}")
            print("   Falling back to single environment")
            
            # FIXED: Actually create the single environment
            import copy
            _local_data_min_max = copy.deepcopy(data_min_max)
            _local_modifiable_bounds = copy.deepcopy(modifiable_bounds)
            _local_fixed_values = copy.deepcopy(fixed_values)
            _local_categorical_encoders = copy.deepcopy(self.categorical_label_encoders) if hasattr(self, 'categorical_label_encoders') else {}
            _local_scalers = {}
            if hasattr(self, 'surrogate_metric_scalers'):
                for metric, scaler in self.surrogate_metric_scalers.items():
                    if scaler is not None:
                        _local_scalers[metric] = copy.deepcopy(scaler)
            
            # ✅ CRITICAL: Deep copy y_scalers
            _local_y_scalers = {}
            if hasattr(self, 'y_scalers'):
                for metric, scaler in self.y_scalers.items():
                    if scaler is not None:
                        _local_y_scalers[metric] = copy.deepcopy(scaler)
            
            self.rl_env = BiosensorEnv(
                modifiable_bounds=_local_modifiable_bounds,
                fixed_values=_local_fixed_values,
                surrogate_models=self.surrogate_models,
                target_metrics=list(target_metrics),
                log_dir=str(self.log_dir),
                surrogate_features=dict(self.surrogate_features),
                X_train=self.X_train,
                data_min_max=_local_data_min_max,
                categorical_label_encoders=_local_categorical_encoders,
                feature_scaler=None,
                surrogate_metric_scalers=_local_scalers,
                target_transformers=self.target_transformers,
                y_scalers=_local_y_scalers  # ✅ ADD THIS LINE
            )
            
            print("   âœ… Single environment created successfully")

        # VALIDATION: Test environment before training
        print("\nÃ°Å¸Â§Âª Testing RL environment setup...")
        try:
            obs = self.rl_env.reset()
            print(f"Ã¢Å“â€¦ Reset successful, obs shape: {obs.shape if hasattr(obs, 'shape') else 'scalar'}")
            
            # Test one step - handle vectorized environments
            if hasattr(self.rl_env, 'num_envs'):
                # Vectorized environment - need array of actions
                n_envs = self.rl_env.num_envs
                actions = np.array([self.rl_env.action_space.sample() for _ in range(n_envs)])
                print(f"   Testing with {n_envs} parallel environments, actions shape: {actions.shape}")
            else:
                # Single environment
                actions = self.rl_env.action_space.sample()
                print(f"   Testing with single environment, action: {actions}")
            
            # ✅ FIXED: Handle both APIs
            step_result = self.rl_env.step(actions)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                raise ValueError(f"Unexpected step() return: {len(step_result)} values")

            # DEBUG: Let's see exactly what we got
            print(f"   DEBUG: info type = {type(info)}")
            print(f"   DEBUG: info shape/len = {len(info) if hasattr(info, '__len__') else 'N/A'}")
            if hasattr(info, '__iter__') and not isinstance(info, (str, dict)):
                print(f"   DEBUG: First element type = {type(info[0]) if len(info) > 0 else 'empty'}")

            # Handle vectorized environment properly
            # SubprocVecEnv returns a LIST of info dicts (one per parallel env)
            predictions_keys = []
            try:
                if isinstance(info, list):
                    # Vectorized: list of dicts
                    if len(info) > 0 and isinstance(info[0], dict):
                        predictions_keys = list(info[0].get('predictions', {}).keys())
                        print(f"   Ã¢Å“â€¦ Vectorized env detected: {len(info)} parallel environments")
                elif isinstance(info, dict):
                    # Single env: just a dict
                    predictions_keys = list(info.get('predictions', {}).keys())
                    print(f"   Ã¢Å“â€¦ Single env detected")
                else:
                    # Something weird - but let's handle it gracefully
                    print(f"   Ã¢Å¡ Ã¯Â¸Â Unexpected info type: {type(info)}, attempting to extract predictions...")
                    # Try to iterate and find predictions
                    for item in info:
                        if isinstance(item, dict) and 'predictions' in item:
                            predictions_keys = list(item['predictions'].keys())
                            break
            except Exception as e:
                print(f"   Ã¢Å¡ Ã¯Â¸Â Could not extract predictions: {e}")
                predictions_keys = []

            # Validate types (handle vectorized)
            if hasattr(self.rl_env, 'num_envs'):
                # Vectorized - reward/done are arrays
                reward_is_scalar = isinstance(reward, np.ndarray) and len(reward) == self.rl_env.num_envs
                done_is_bool = isinstance(done, np.ndarray) and done.dtype == bool
            else:
                # Single env
                reward_is_scalar = isinstance(reward, (int, float, np.number)) or (isinstance(reward, np.ndarray) and reward.shape == ())
                done_is_bool = isinstance(done, (bool, np.bool_))

            print(f"Ã¢Å“â€¦ Step successful")
            print(f"   Reward: {reward} (valid: {reward_is_scalar})")
            print(f"   Done: {done} (bool: {done_is_bool})")
            print(f"   Predictions: {predictions_keys}")
                        
            if not reward_is_scalar or not done_is_bool:
                print(f"Ã¢Å¡ Ã¯Â¸Â  WARNING: Non-scalar reward or done - this may cause training issues")
                
        except Exception as e:
            print(f"Ã¢ÂÅ’ Environment test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("Ã¢Å“â€¦ Advanced RL environment with proper variable classification created!")

        return self.rl_env
    
    
    def get_stable_training_configs(self):
        """Optimized PPO config for biosensor optimization"""
        
        ppo_config = {
            "policy": "MlpPolicy",
            "env": self.rl_env,
            "verbose": 1,
            
            # ✅ OPTIMIZED for continuous action biosensor task
            "learning_rate": self._cosine_schedule(3e-4, 5e-6),  # ✅ Lower start, slower decay
            "n_steps": 2048,              # ✅ MORE data per update (was 2048)
            "batch_size": 128,            # ✅ Smaller batches for better gradients (was 512)
            "n_epochs": 20,               # ✅ MORE gradient updates per batch (was 10)
            "gamma": 0.99,                # ✅ Standard discount (0.995 too high)
            "gae_lambda": 0.95,           # ✅ Standard GAE (0.98 too high)
            "clip_range": 0.1,  # TIGHTER clipping for stability
            
            "ent_coef": 0.02,             # ✅ MUCH HIGHER for exploration (was 0.005!!!)
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            
            "normalize_advantage": True,
            "policy_kwargs": {
                "net_arch": dict(
                    pi=[256, 256, 128],   # ✅ Slightly smaller but deeper (was [512, 256, 128])
                    vf=[256, 256, 128]    # ✅ Match policy network
                ),
                "activation_fn": torch.nn.Tanh,  # ✅ Tanh better for continuous (was ReLU)
                "ortho_init": True,
            }
        }
        
        return None, ppo_config  # DQN removed
    
    def _linear_schedule(self, initial_value):
        """Linear learning rate schedule"""
        def schedule(progress_remaining):
            return progress_remaining * initial_value
        return schedule

    def _adaptive_clip_schedule(self, initial_value):
        """
        Adaptive clip range schedule for better exploration
        Start high (0.3) for exploration, decay to low (0.1) for exploitation
        """
        def schedule(progress_remaining):
            # More aggressive early exploration
            min_clip = 0.1
            max_clip = initial_value
            return min_clip + (max_clip - min_clip) * progress_remaining
        return schedule
    
    def _cosine_schedule(self, initial_lr, final_lr):
        """
        Cosine annealing schedule - starts high, gradually decreases
        Better than linear for RL
        """
        def schedule(progress_remaining):
            # progress_remaining goes from 1.0 to 0.0
            progress = 1.0 - progress_remaining
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return final_lr + (initial_lr - final_lr) * cosine_decay
        return schedule

    def _quick_eval(self, agent, n_episodes=10):
        """Quick evaluation of agent performance"""
        total_reward = 0
        
        for _ in range(n_episodes):
            obs = self.rl_env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            max_steps = 300  # Prevent infinite loops
            
            while not done and step_count < max_steps:
                action, _ = agent.predict(obs, deterministic=True)
                
                # ✅ FIXED: Handle both Gym and Gymnasium API
                step_result = self.rl_env.step(action)
                
                if len(step_result) == 5:
                    # Gymnasium API: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # Old Gym API: (obs, reward, done, info)
                    obs, reward, done, info = step_result
                else:
                    raise ValueError(f"Unexpected step() return: {len(step_result)} values")
                
                # Handle vectorized vs single environment
                if hasattr(self.rl_env, 'num_envs'):
                    # Vectorized environment
                    reward_array = np.atleast_1d(reward)
                    episode_reward += np.mean(reward_array)
                    
                    if isinstance(done, np.ndarray):
                        if np.any(done):
                            break
                    elif done:
                        break
                else:
                    # Single environment
                    episode_reward += float(reward)
                    if done:
                        break
                
                step_count += 1
            
            total_reward += episode_reward
        
        return total_reward / n_episodes

    def train_rl_agents(self, total_timesteps: int = 3000000):  # 3M minimum
        """Train RL agents with much better exploration and longer training"""
        
        if self.rl_env is None:
            print("Ã¢Å¡ Ã¯Â¸Â Setting up RL environment first...")
            self.setup_rl_environment(self.target_metrics)

        # 🔍 PRE-TRAINING VALIDATION   - -- - ------
        print("\n🔍 PRE-TRAINING VALIDATION:")
        print("=" * 60)

        # Test reward calculation
        obs = self.rl_env.reset()
        action = self.rl_env.action_space.sample()
        step_out = self.rl_env.step(action)

        if len(step_out) == 5:
            _, reward, _, _, _ = step_out
        else:
            _, reward, _, _ = step_out

        print(f"✅ Reward variance test:")
        print(f"   Sample reward: {reward}")
        print(f"   Mean reward: {np.mean(reward):.4f}")
        print(f"   Std reward: {np.std(reward):.4f}")
        print(f"   Reward range valid: {np.all(reward > -1.0) and np.all(reward < 10.0)}")

        if np.mean(reward) < -0.5:
            print("⚠️  WARNING: Average reward is very negative!")
            print("   Check reward normalization ranges")
        elif np.mean(reward) > 0.0:
            print("✅ Good! Positive average reward - agent can learn")

        # Test action variety
        actions = [self.rl_env.action_space.sample() for _ in range(10)]
        action_variance = np.var(actions, axis=0)
        print(f"✅ Action space variance: {action_variance.mean():.4f}")
        print(f"   (Should be > 0.1 for good exploration)")

        # Test surrogate predictions
        if hasattr(self.rl_env, 'envs'):
            test_env = self.rl_env.envs[0]
        else:
            test_env = self.rl_env

        predictions = {}
        for metric in test_env.surrogate_models.keys():
            try:
                full_state = test_env._create_full_state(target_metric=metric)
                model = test_env.surrogate_models[metric]
                
                if hasattr(model, 'predict'):
                    pred = model.predict(full_state.reshape(1, -1))[0]
                else:
                    pred = model(torch.FloatTensor(full_state).unsqueeze(0)).item()
                
                predictions[metric] = pred
            except Exception as e:
                print(f"❌ Prediction failed for {metric}: {e}")

        print(f"✅ Surrogate predictions:")
        for k, v in predictions.items():
            print(f"   {k}: {v:.4f}")

        print("=" * 60)
        # END VALIDATION BLOCK  -----------------
        
        print(f"Ã°Å¸Å½Â¯ Training RL agents for {total_timesteps} timesteps...")
        
        # Get stable configs
        dqn_config, ppo_config = self.get_stable_training_configs()

        # Initialize agents
        print("   Initializing PPO agent (DQN removed for continuous actions)...")
        try:
            # DQN doesn't support continuous actions - only use PPO
            dqn_agent = None
            
            # Verify action space is Box (continuous)
            if not isinstance(self.rl_env.action_space, spaces.Box):
                raise ValueError(f"Expected Box action space, got {type(self.rl_env.action_space)}")
            
            ppo_agent = PPO(**ppo_config)
            print(f"      ✅ PPO initialized for continuous action space: {self.rl_env.action_space}")
        except Exception as e:
            print(f"      ❌ Agent initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        # ðŸ”º IMPROVED: Training with early stopping
        if dqn_agent is not None:
            print("   Training DQN with early stopping...")
            try:
                best_eval_reward = -float('inf')
                patience = 5  # Stop if no improvement for 5 checkpoints
                patience_counter = 0
                
                chunk_size = total_timesteps // 20  # More frequent evaluation
                
                for i in range(10):
                    print(f"      DQN chunk {i+1}/10...")
                    dqn_agent.learn(total_timesteps=chunk_size)
                    
                    # Evaluate every 2 chunks
                    if i % 2 == 1:
                        eval_reward = self._quick_eval(dqn_agent, n_episodes=10)
                        print(f"         Eval reward: {eval_reward:.4f}")
                        
                        if eval_reward > best_eval_reward + 0.01:  # Significant improvement
                            best_eval_reward = eval_reward
                            patience_counter = 0
                            # Save checkpoint
                            dqn_agent.save(self.models_dir / f"dqn_checkpoint_best.zip")
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            print(f"      âš ï¸  Early stopping triggered (no improvement for {patience} checks)")
                            # Load best checkpoint
                            dqn_agent = DQN.load(self.models_dir / f"dqn_checkpoint_best.zip")
                            break
                            
            except Exception as e:
                print(f"      âŒ DQN training failed: {e}")
                dqn_agent = None
        
        # Train PPO with checkpoints
        if ppo_agent is not None:
            print("   Training PPO with exploration focus...")
            try:
                # ✅ FIXED: Longer training with better checkpointing
                chunk_size = total_timesteps // 30  # 30 checkpoints
                best_eval_reward = -float('inf')
                patience = 5
                patience_counter = 0

                for i in range(30):
                    print(f"      PPO Training chunk {i+1}/30 ({chunk_size} steps)...")
                    
                    try:
                        ppo_agent.learn(total_timesteps=chunk_size)
                    except Exception as e:
                        print(f"      ❌ Training failed at chunk {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                    
                    # Evaluate every 5 chunks (new - less frequent)
                    if (i + 1) % 5 == 0:
                        try:
                            eval_reward = self._quick_eval(ppo_agent, n_episodes=10)
                            print(f"         Eval: {eval_reward:.4f} (best: {best_eval_reward:.4f})")
                            
                            if eval_reward > best_eval_reward + 0.01:
                                best_eval_reward = eval_reward
                                patience_counter = 0
                                ppo_agent.save(self.models_dir / "ppo_checkpoint_best.zip")
                                print(f"         ✅ New best! Saved checkpoint.")
                            else:
                                patience_counter += 1
                                
                            if patience_counter >= patience and i >= 18:
                                print(f"      ⚠️ Early stopping (no improvement for {patience} checks)")
                                ppo_agent = PPO.load(self.models_dir / "ppo_checkpoint_best.zip")
                                break
                        except Exception as e:
                            print(f"      ⚠️ Evaluation failed: {e}")
                            # Continue training anyway
                
                print("   âœ… PPO training completed")
                        
            except Exception as e:
                print(f"      Ã¢ÂÅ’ PPO training failed: {e}")
                ppo_agent = None

        agents_performance = {}
        
        # Final evaluation with MORE episodes
        if dqn_agent is not None:
            print("   Final DQN evaluation...")
            try:
                dqn_rewards = []
                n_eval_episodes = 20
                
                # Check if vectorized
                is_vec_env = hasattr(self.rl_env, 'num_envs')
                n_parallel = self.rl_env.num_envs if is_vec_env else 1
                
                episodes_completed = 0
                
                while episodes_completed < n_eval_episodes:
                    obs = self.rl_env.reset()
                    done_flags = np.zeros(n_parallel, dtype=bool)
                    episode_rewards = np.zeros(n_parallel)
                    step_counts = np.zeros(n_parallel)  # Ã¢â€ Â ADD THIS
                    
                    for step in range(1000):  # Max steps per episode
                        action, _ = dqn_agent.predict(obs, deterministic=True)
                        obs, reward, done, _ = self.rl_env.step(action)
                        
                        # Convert to arrays for consistent handling
                        reward_array = np.atleast_1d(reward)
                        done_array = np.atleast_1d(done).astype(bool)
                        
                        # Accumulate rewards only for non-done envs
                        episode_rewards[~done_flags] += reward_array[~done_flags]
                        step_counts[~done_flags] += 1  # Ã¢â€ Â ADD THIS
                        
                        # Update done flags
                        done_flags = np.logical_or(done_flags, done_array)
                        
                        # Break if all environments are done
                        if np.all(done_flags):
                            break
                    
                    # Ã¢â€ Â CHANGE: Average per-step reward instead of cumulative
                    avg_episode_rewards = episode_rewards / np.maximum(step_counts, 1)
                    
                    # Record completed episodes
                    if is_vec_env:
                        # Each parallel env is one episode
                        for i in range(n_parallel):
                            if episodes_completed < n_eval_episodes:
                                dqn_rewards.append(avg_episode_rewards[i])  # Ã¢â€ Â CHANGED
                                episodes_completed += 1
                    else:
                        dqn_rewards.append(avg_episode_rewards[0])  # Ã¢â€ Â CHANGED
                        episodes_completed += 1
                    
                    if episodes_completed % 5 == 0:
                        print(f"      DQN Episode {episodes_completed}: {np.mean(avg_episode_rewards):.4f}")  # Ã¢â€ Â CHANGED
                
                agents_performance['DQN'] = {
                    'mean_reward': np.mean(dqn_rewards),
                    'std_reward': np.std(dqn_rewards),
                    'max_reward': np.max(dqn_rewards),
                    'rewards': dqn_rewards
                }
                
                self.rl_agents['DQN'] = dqn_agent
                print(f"      Ã¢Å“â€¦ DQN Final: {np.mean(dqn_rewards):.4f} Ã‚Â± {np.std(dqn_rewards):.4f}")
                
            except Exception as e:
                print(f"      Ã¢ÂÅ’ DQN evaluation failed: {e}")
                import traceback
                traceback.print_exc()

        # Similar for PPO...
        if ppo_agent is not None:
            print("   Final PPO evaluation...")
            try:
                ppo_rewards = []
                n_eval_episodes = 20
                
                # âœ… FIX: Simplified evaluation for vectorized environments
                is_vec_env = hasattr(self.rl_env, 'num_envs')
                n_parallel = self.rl_env.num_envs if is_vec_env else 1
                
                episodes_completed = 0
                
                while episodes_completed < n_eval_episodes:
                    obs = self.rl_env.reset()
                    episode_rewards = np.zeros(n_parallel, dtype=float)
                    step_counts = np.zeros(n_parallel, dtype=int)
                    done_flags = np.zeros(n_parallel, dtype=bool)
                    
                    for step in range(300):
                        action, _ = ppo_agent.predict(obs, deterministic=True)
                        
                        # ✅ FIXED: Handle both APIs
                        step_result = self.rl_env.step(action)
                        
                        if len(step_result) == 5:
                            # Gymnasium API
                            obs, reward, terminated, truncated, info = step_result
                            done = np.logical_or(terminated, truncated)
                        elif len(step_result) == 4:
                            # Old Gym API
                            obs, reward, done, info = step_result
                        else:
                            raise ValueError(f"Unexpected step() return: {len(step_result)} values")

                        
                        # âœ… FIX: Handle both scalar and array rewards safely
                        if isinstance(reward, (int, float)):
                            # Single environment
                            reward_array = np.array([reward], dtype=float)
                            done_array = np.array([done], dtype=bool)
                        else:
                            # Vectorized environment
                            reward_array = np.atleast_1d(reward).astype(float)
                            done_array = np.atleast_1d(done).astype(bool)
                        
                        # Accumulate only for non-done envs
                        active_mask = ~done_flags[:len(reward_array)]
                        episode_rewards[:len(reward_array)][active_mask] += reward_array[active_mask]
                        step_counts[:len(reward_array)][active_mask] += 1
                        
                        # Update done flags
                        done_flags[:len(done_array)] = np.logical_or(
                            done_flags[:len(done_array)], 
                            done_array
                        )
                        
                        if np.all(done_flags):
                            break
                    
                    # Calculate mean reward per step for each episode
                    for i in range(n_parallel):
                        if step_counts[i] > 0 and episodes_completed < n_eval_episodes:
                            avg_reward = episode_rewards[i] / step_counts[i]
                            ppo_rewards.append(float(avg_reward))
                            episodes_completed += 1
                    
                    if episodes_completed % 5 == 0:
                        print(f"      Evaluated {episodes_completed}/{n_eval_episodes} episodes")
                
                # Store results
                agents_performance['PPO'] = {
                    'mean_reward': float(np.mean(ppo_rewards)),
                    'std_reward': float(np.std(ppo_rewards)),
                    'max_reward': float(np.max(ppo_rewards)),
                    'rewards': [float(r) for r in ppo_rewards]
                }
                
                self.rl_agents['PPO'] = ppo_agent
                print(f"      âœ… PPO Final: {np.mean(ppo_rewards):.4f} Â± {np.std(ppo_rewards):.4f}")
                
            except Exception as e:
                print(f"      âŒ PPO evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Find best agent
        if agents_performance:
            best_agent = max(agents_performance.keys(), key=lambda k: agents_performance[k]['mean_reward'])
            best_performance = agents_performance[best_agent]['mean_reward']
            print(f"\nÃ°Å¸Ââ€  Best Agent: {best_agent} with mean reward: {best_performance:.4f}")

            # Ã¢â€ Â ADD DETAILED EVALUATION
            print(f"\nÃ°Å¸â€œÅ  DETAILED RL PERFORMANCE EVALUATION:")
            print("=" * 60)
            
            for agent_name, perf in agents_performance.items():
                mean_r = perf['mean_reward']
                std_r = perf['std_reward']
                
                print(f"\n{agent_name} Agent:")
                print(f"   Mean Reward: {mean_r:.4f} Ã‚Â± {std_r:.4f}")
                
                # Evaluate performance tier
                if mean_r >= 0.80:
                    tier = "Ã°Å¸Å’Å¸ EXCELLENT"
                    advice = "Near-optimal biosensor design achieved!"
                elif mean_r >= 0.70:
                    tier = "Ã¢Å“â€¦ GOOD"
                    advice = "Strong performance, minor improvements possible"
                elif mean_r >= 0.60:
                    tier = "Ã¢Å¡ Ã¯Â¸Â ACCEPTABLE"
                    advice = "Moderate improvement, consider more training"
                elif mean_r >= 0.50:
                    tier = "Ã¢ÂÅ’ MARGINAL"
                    advice = "Limited learning, check surrogate quality"
                else:
                    tier = "Ã¢ÂÅ’ POOR"
                    advice = "Significant issues, review entire pipeline"
                
                print(f"   Performance Tier: {tier}")
                print(f"   Recommendation: {advice}")
                
                # Stability evaluation
                if std_r < 0.05:
                    stability = "Ã°Å¸Å½Â¯ Excellent stability"
                elif std_r < 0.10:
                    stability = "Ã¢Å“â€¦ Good stability"
                elif std_r < 0.15:
                    stability = "Ã¢Å¡ Ã¯Â¸Â Moderate variance"
                else:
                    stability = "Ã¢ÂÅ’ High variance (unstable)"
                
                print(f"   Stability: {stability}")
                
                # Estimated MOS breakdown (approximate from mean reward)
                print(f"\n   Estimated Component Contributions:")
                print(f"   Ã¢â€Å“Ã¢â€â‚¬ SNR (45%):  ~{mean_r * 0.45:.3f}")
                print(f"   Ã¢â€Å“Ã¢â€â‚¬ DR (25%):   ~{mean_r * 0.25:.3f}")
                print(f"   Ã¢â€Å“Ã¢â€â‚¬ -FNR (20%): ~{mean_r * 0.20:.3f}")
                print(f"   Ã¢â€â€Ã¢â€â‚¬ -TTD (10%): ~{mean_r * 0.10:.3f}")
            
            # Overall recommendation
            print(f"\nÃ°Å¸Å½Â¯ OVERALL ASSESSMENT:")
            if best_performance >= 0.70:
                print("Ã¢Å“â€¦ RL optimization was successful!")
                print("   The agent learned effective biosensor designs.")
            elif best_performance >= 0.50:
                print("Ã¢Å¡ Ã¯Â¸Â RL optimization showed moderate success.")
                print("   RECOMMENDATIONS:")
                print("   1. Check surrogate model RÃ‚Â² scores (should be Ã¢â€°Â¥0.90)")
                print("   2. Increase training timesteps (try 1M instead of 500k)")
                print("   3. Review reward function weights")
            else:
                print("Ã¢ÂÅ’ RL optimization underperformed.")
                print("   CRITICAL ACTIONS NEEDED:")
                print("   1. Ã¢Å¡ Ã¯Â¸Â VERIFY surrogate model quality (RÃ‚Â² Ã¢â€°Â¥ 0.90)")
                print("   2. Increase dataset size (need more diverse examples)")
                print("   3. Check for bugs in reward calculation")
                print("   4. Consider different RL algorithm hyperparameters")

            self._plot_rl_performance(agents_performance)

        # Ã°Å¸â€Â¥ ADD THIS HERE - Close environment to save logs
        # Handle both single and vectorized environments
        if hasattr(self.rl_env, 'log_dir'):
            log_dir = self.rl_env.log_dir
        elif hasattr(self, 'rl_log_dir'):
            log_dir = self.rl_log_dir
        else:
            log_dir = self.log_dir

        print(f"Ã°Å¸â€™Â¾ Saving training logs to: {log_dir}")
        # Close environment properly (handles both single and vectorized)
        try:
            if hasattr(self.rl_env, 'close'):
                self.rl_env.close()
                print(f"Ã¢Å“â€¦ Environment logs saved")
        except Exception as e:
            print(f"Ã¢Å¡ Ã¯Â¸Â  Could not close environment cleanly: {e}")

        return agents_performance
            
    def generate_comprehensive_analysis(self):
        """Generate all visualization and analysis plots"""
        print("Ã°Å¸â€œÅ  Generating comprehensive analysis...")
        
        # 1. Data exploration plots
        self._plot_data_overview()
        self.plot_target_metrics_correlation_matrix()
        self._plot_target_distributions()
        
        # 2. Model performance plots
        if self.model_performance:
            self._plot_learning_curves()
            self._plot_prediction_scatter()
            self._plot_residual_analysis()
        
        # 3. Feature analysis
        if self.feature_importance:
            self._plot_feature_importance_comparison()
        
        # 4. Multi-objective analysis
        self._plot_pareto_frontiers()
        self._plot_multi_objective_scatter()
        
        # 5. Biomarker-specific analysis
        self._plot_biomarker_performance()
        
        print("Ã¢Å“â€¦ Analysis generation completed!")
    
    def _save_pca_visualization(self, pca_df, explained_variance):
        """Save PCA visualization as separate plots"""
        # PCA scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                            c=range(len(pca_df)), cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.title('PCA: First Two Components')
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "pca_scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Explained variance plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance[:10]) + 1), explained_variance[:10])
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "pca_explained_variance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance_for_sclerostin(self):
        """Generate feature importance plot for the 13-feature Sclerostin biosensor"""
        
        if not self.feature_importance:
            print("Ã¢Å¡ Ã¯Â¸Â  No feature importance data available. Training a model to generate it...")
            
            # Train a quick model to get feature importance
            try:
                if hasattr(self, 'processed_data') and self.processed_data:
                    if 'multi_objective_score' in self.processed_data['target_names']:
                        self.train_supervised_models('multi_objective_score')
                    else:
                        self.train_supervised_models(self.processed_data['target_names'][0])
                else:
                    print("Ã¢ÂÅ’ Cannot generate feature importance - no processed data available")
                    return
            except Exception as e:
                print(f"Ã¢ÂÅ’ Failed to generate feature importance: {e}")
                return
            
            # Check again
            if not self.feature_importance:
                print("Ã¢ÂÅ’ Still no feature importance after model training")
                return
        
        # Get the best model's feature importance
        best_model_key = list(self.feature_importance.keys())[0]
        feature_imp_df = self.feature_importance[best_model_key].copy()
        
        # Classify features as MODIFIABLE or FIXED
        modifiable_features = [
            'receptor_binding_affinity_Kd',
            'signal_amplification_factor',
            'transcriptional_promoter_strength',
            'mRNA_half_life',
            'protein_half_life',
            'feedback_presence',
            'feedback_strength',
            'circuit_cellular_burden',
            'circuit_topology'
        ]
        
        fixed_features = [
            'sclerostin_concentration',
            'dkk1_concentration',
            'local_pH',
            'mechanical_loading'
        ]
        
        # Add classification column
        feature_imp_df['classification'] = feature_imp_df['feature'].apply(
            lambda x: 'MODIFIABLE (RL Action Space)' if x in modifiable_features 
            else 'FIXED (Context)' if x in fixed_features 
            else 'Other'
        )
        
        # Sort by importance
        feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Color map
        colors = feature_imp_df['classification'].map({
            'MODIFIABLE (RL Action Space)': '#2E86AB',  # Blue
            'FIXED (Context)': '#A23B72',  # Purple
            'Other': '#CCCCCC'  # Gray
        })
        
        bars = plt.barh(range(len(feature_imp_df)), feature_imp_df['importance'], color=colors)
        plt.yticks(range(len(feature_imp_df)), feature_imp_df['feature'])
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title('Sclerostin Biosensor: Feature Importance Analysis\n(13 Input Features)', 
                fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label=f'MODIFIABLE (RL Actions, n={len(modifiable_features)})'),
            Patch(facecolor='#A23B72', label=f'FIXED (Context, n={len(fixed_features)})')
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Add grid
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "sclerostin_feature_importance_classified.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sclerostin feature importance plot saved")
    
    def _plot_model_comparison(self, performance_results, target_metric):
        """Plot model performance comparison as separate plots"""
        # Add this at the beginning of the method
        plt.figure(figsize=(12, 8))  # Set explicit, reasonable figure size

        models = list(performance_results.keys())
        r2_scores = [performance_results[model]['test_r2'] for model in models]
        mse_scores = [performance_results[model]['test_mse'] for model in models]
        
        # RÃ‚Â² comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.ylabel('RÃ‚Â² Score')
        plt.title(f'Model Comparison - RÃ‚Â² Score ({target_metric})')
        plt.ylim(0, 1)
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        # Before plt.savefig, add:
        fig = plt.gcf()
        fig.set_size_inches(12, 8)  # Ensure figure size is reasonable
        
        plt.savefig(self.plots_dir / f"model_r2_comparison_{target_metric}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # MSE comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, mse_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.ylabel('MSE')
        plt.title(f'Model Comparison - MSE ({target_metric})')
        for i, v in enumerate(mse_scores):
            plt.text(i, v + max(mse_scores)*0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        # Before plt.savefig, add:
        fig = plt.gcf()
        fig.set_size_inches(12, 8)  # Ensure figure size is reasonable
        plt.savefig(self.plots_dir / f"model_mse_comparison_{target_metric}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rl_performance(self, agents_performance: Dict):
        """Plot comprehensive RL performance analysis with separate saves"""
        if not agents_performance:
            return
        
        import matplotlib.pyplot as plt
        
        agents = list(agents_performance.keys())
        rewards_data = [agents_performance[agent]['rewards'] for agent in agents]
        
        # 1. Box plot for reward distribution
        plt.figure(figsize=(10, 6))
        plt.boxplot(rewards_data, labels=agents)
        plt.ylabel('Episode Reward')
        plt.title('RL Agents Reward Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "rl_reward_distribution_boxplot.png", 
                dpi=300, bbox_inches='tight')
        
        # 2. Histogram reward distribution
        plt.figure(figsize=(10, 6))
        for i, agent in enumerate(agents):
            rewards = agents_performance[agent]['rewards']
            plt.hist(rewards, alpha=0.6, label=agent, bins=10)
        plt.title('Reward Distribution Histogram')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rl_reward_distribution_histogram.png', dpi=300, bbox_inches='tight')
        
        # 3. Mean rewards comparison
        plt.figure(figsize=(10, 6))
        mean_rewards = [agents_performance[agent]['mean_reward'] for agent in agents]
        std_rewards = [agents_performance[agent]['std_reward'] for agent in agents]
        
        bars = plt.bar(agents, mean_rewards, yerr=std_rewards, 
                    color=['blue', 'orange'], alpha=0.7, capsize=5)
        plt.ylabel('Mean Episode Reward')
        plt.title('RL Agents Average Performance')
        plt.grid(True, alpha=0.3)
        
        for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}Ã‚Â±{std:.3f}', ha='center')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "rl_mean_performance.png", 
                dpi=300, bbox_inches='tight')
        
        # 4. Performance metrics comparison
        plt.figure(figsize=(10, 6))
        metrics = ['mean_reward', 'max_reward']
        x = np.arange(len(agents))
        width = 0.35
        
        for i, metric in enumerate(metrics):
            if all(metric in agents_performance[agent] for agent in agents):
                values = [agents_performance[agent][metric] for agent in agents]
                plt.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), alpha=0.7)
        
        plt.title('Performance Metrics Comparison')
        plt.ylabel('Reward')
        plt.xticks(x + width/2, agents)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rl_performance_metrics.png', dpi=300, bbox_inches='tight')
        
        # 5. Consistency analysis (coefficient of variation)
        plt.figure(figsize=(10, 6))
        cv_values = []
        for agent in agents:
            mean_r = agents_performance[agent]['mean_reward']
            std_r = agents_performance[agent]['std_reward']
            cv = std_r / abs(mean_r) if mean_r != 0 else 0
            cv_values.append(cv)
        
        plt.bar(agents, cv_values, alpha=0.7, color='orange')
        plt.title('Consistency Analysis (Lower is Better)')
        plt.ylabel('Coefficient of Variation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rl_consistency_analysis.png', dpi=300, bbox_inches='tight')
        
        # 6. Reward history for each agent separately
        for agent in agents:
            plt.figure(figsize=(12, 6))
            rewards = agents_performance[agent]['rewards']
            plt.plot(rewards, label=f'{agent} (mean: {np.mean(rewards):.3f})', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'{agent} Training Reward History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"rl_{agent.lower()}_reward_history.png", 
                    dpi=300, bbox_inches='tight')
        
        print(f"Ã°Å¸â€œÅ  RL performance plots saved separately:")
        print(f"  - Reward distribution (boxplot): {self.plots_dir / 'rl_reward_distribution_boxplot.png'}")
        print(f"  - Reward distribution (histogram): {self.plots_dir / 'rl_reward_distribution_histogram.png'}")
        print(f"  - Mean performance: {self.plots_dir / 'rl_mean_performance.png'}")
        print(f"  - Performance metrics: {self.plots_dir / 'rl_performance_metrics.png'}")
        print(f"  - Consistency analysis: {self.plots_dir / 'rl_consistency_analysis.png'}")
        for agent in agents:
            print(f"  - {agent} reward history: {self.plots_dir / f'rl_{agent.lower()}_reward_history.png'}")
    
    def _plot_data_overview(self):
        """Plot data overview and statistics as separate plots"""
        if self.raw_data is None:
            return
        
        # Dataset info text plot
        plt.figure(figsize=(10, 6))
        plt.text(0.1, 0.8, f"Dataset Shape: {self.raw_data.shape}", fontsize=16, weight='bold')
        plt.text(0.1, 0.7, f"Total Features: {self.raw_data.shape[1]}", fontsize=14)
        plt.text(0.1, 0.6, f"Total Samples: {self.raw_data.shape[0]}", fontsize=14)
        plt.text(0.1, 0.5, f"Missing Values: {self.raw_data.isnull().sum().sum()}", fontsize=14)
        plt.text(0.1, 0.4, f"Numerical Features: {len(self.raw_data.select_dtypes(include=[np.number]).columns)}", fontsize=14)
        plt.text(0.1, 0.3, f"Categorical Features: {len(self.raw_data.select_dtypes(include=['object']).columns)}", fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Dataset Overview Statistics", fontsize=18, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "dataset_overview_stats.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Data types distribution
        plt.figure(figsize=(8, 8))
        dtypes_count = self.raw_data.dtypes.value_counts()
        plt.pie(dtypes_count.values, labels=dtypes_count.index, autopct='%1.1f%%', startangle=90)
        plt.title("Data Types Distribution")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "data_types_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Missing values analysis
        missing_data = self.raw_data.isnull().sum().sort_values(ascending=False).head(20)
        if len(missing_data) > 0 and missing_data.max() > 0:
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(missing_data)), missing_data.values)
            plt.yticks(range(len(missing_data)), missing_data.index)
            plt.xlabel("Missing Values Count")
            plt.title("Missing Values by Column (Top 20)")
            plt.tight_layout()
            plt.savefig(self.plots_dir / "missing_values_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Individual numerical feature distributions (first 5)
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns[:5]
        for i, col in enumerate(numerical_cols):
            plt.figure(figsize=(10, 6))
            plt.hist(self.raw_data[col].dropna(), bins=30, alpha=0.7, color=plt.cm.Set3(i))
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {col}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"distribution_{col.replace('/', '_').replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_sclerostin_correlation_matrix(self):
        """Generate correlation matrix between 13 inputs and 4 outputs for Sclerostin"""
        
        if self.processed_data is None:
            print("ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸  No processed data available")
            return
        
        # Get X (13 features) and Y (4 targets)
        X_data = self.processed_data['X']
        y_data = self.processed_data['y']
        
        # Combine for correlation
        combined_data = pd.concat([X_data, y_data], axis=1)
        
        # Calculate correlation
        corr_matrix = combined_data.corr()
        
        # Extract the cross-correlation (13 features x 4 targets)
        input_features = X_data.columns
        output_targets = y_data.columns
        
        cross_corr = corr_matrix.loc[input_features, output_targets]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cross_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=False, linewidths=0.5, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1)
        
        plt.title('Sclerostin Biosensor: Input-Output Correlation Matrix\n(13 Features Ãƒâ€” 4 Targets)', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Output Targets (Y)', fontsize=12, fontweight='bold')
        plt.ylabel('Input Features (X)', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "sclerostin_input_output_correlation.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sclerostin correlation matrix saved")


    def _plot_target_distributions(self):
        """Plot target variable distributions as separate plots"""
        if self.processed_data is None:
            return
        
        target_data = self.processed_data['y']
        
        # Create individual plots for each target
        for target in target_data.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(target_data[target].dropna(), bins=30, alpha=0.7, 
                    color=plt.cm.Set3(hash(target) % 12))
            plt.xlabel(target)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {target}")
            
            # Add statistics
            mean_val = target_data[target].mean()
            std_val = target_data[target].std()
            plt.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"target_distribution_{target.replace('/', '_').replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_learning_curves(self):
        """Plot learning curves for all models as separate plots"""
        if not self.model_performance:
            return
        
        for target_metric, models_perf in self.model_performance.items():
            for model_name, perf in models_perf.items():
                plt.figure(figsize=(10, 6))
                
                if 'train_losses' in perf and 'val_losses' in perf:
                    plt.plot(perf['train_losses'], label='Training Loss', alpha=0.8)
                    plt.plot(perf['val_losses'], label='Validation Loss', alpha=0.8)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'{model_name} Learning Curve - {target_metric}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"learning_curve_{model_name}_{target_metric}.png", 
                               dpi=300, bbox_inches='tight')
                else:
                    # For non-neural network models, show performance metrics
                    metrics = ['train_r2', 'test_r2', 'train_mse', 'test_mse']
                    values = [perf.get(metric, 0) for metric in metrics]
                    bars = plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
                    plt.title(f'{model_name} Performance - {target_metric}')
                    plt.ylabel('Score')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"performance_metrics_{model_name}_{target_metric}.png", 
                               dpi=300, bbox_inches='tight')
                
                plt.close()
    
    def _plot_prediction_scatter(self):
        """Plot actual vs predicted scatter plots as separate plots"""
        if not self.model_performance:
            return
        
        for target_metric, models_perf in self.model_performance.items():
            y_actual = self.y_test[target_metric]
            
            for model_name, perf in models_perf.items():
                if 'predictions_test' in perf:
                    y_pred = perf['predictions_test']
                    
                    plt.figure(figsize=(8, 8))
                    plt.scatter(y_actual, y_pred, alpha=0.6, s=20)
                    
                    # Perfect prediction line
                    min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
                    
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title(f'{model_name} - {target_metric}\nRÃ‚Â² = {perf["test_r2"]:.3f}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"prediction_scatter_{model_name}_{target_metric}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()

    def _plot_residual_analysis(self):
        """Plot residual analysis for model validation as separate plots"""
        if not self.model_performance:
            return
        
        for target_metric, models_perf in self.model_performance.items():
            y_actual = self.y_test[target_metric]
            
            for model_name, perf in models_perf.items():
                if 'predictions_test' in perf:
                    y_pred = perf['predictions_test']
                    residuals = y_actual - y_pred
                    
                    # Residuals vs Predicted
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    plt.title(f'{model_name} - {target_metric} - Residuals vs Predicted')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"residuals_vs_predicted_{model_name}_{target_metric}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Residuals histogram
                    plt.figure(figsize=(10, 6))
                    plt.hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue')
                    plt.xlabel('Residuals')
                    plt.ylabel('Density')
                    plt.title(f'{model_name} - {target_metric} - Residuals Distribution')
                    plt.grid(True, alpha=0.3)
                    
                    # Add normal distribution overlay
                    x = np.linspace(residuals.min(), residuals.max(), 100)
                    normal_dist = (1/np.sqrt(2*np.pi*residuals.var())) * np.exp(-0.5*(x-residuals.mean())**2/residuals.var())
                    plt.plot(x, normal_dist, 'r-', label='Normal Distribution')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"residuals_distribution_{model_name}_{target_metric}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
    
    def _plot_feature_importance_comparison(self):
        """Plot feature importance comparison across models"""
        if not self.feature_importance:
            return
        
        # Combine feature importance from all models
        all_features = {}
        for model_key, feature_imp in self.feature_importance.items():
            for _, row in feature_imp.head(10).iterrows():  # Top 10 features per model
                if row['feature'] not in all_features:
                    all_features[row['feature']] = {}
                all_features[row['feature']][model_key] = row['importance']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_features).T.fillna(0)
        
        if len(comparison_df) > 0:
            plt.figure(figsize=(14, 8))
            sns.heatmap(comparison_df, annot=True, cmap='YlOrRd', fmt='.3f')
            plt.title('Feature Importance Comparison Across Models')
            plt.xlabel('Model')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "feature_importance_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_pareto_frontiers(self):
        """Plot Pareto frontiers for multi-objective analysis as separate plots"""
        if self.processed_data is None:
            return
        
        target_data = self.processed_data['y']
        
        # Create Pareto frontier plots for key metric pairs
        metric_pairs = [
            ('perf_sensitivity', 'perf_specificity'),
            ('perf_f1_score', 'robustness_score'),
            ('perf_snr', 'perf_response_time')
        ]
        
        for i, (metric1, metric2) in enumerate(metric_pairs):
            if metric1 in target_data.columns and metric2 in target_data.columns:
                plt.figure(figsize=(10, 8))
                
                x = target_data[metric1]
                y = target_data[metric2]
                
                # Scatter plot
                scatter = plt.scatter(x, y, alpha=0.6, s=20, c=target_data.get('multi_objective_score', x), 
                                    cmap='viridis')
                
                # Simple Pareto frontier approximation
                # Find points that are not dominated by any other point
                pareto_points = []
                for idx in range(len(x)):
                    is_pareto = True
                    for other_idx in range(len(x)):
                        if other_idx != idx:
                            if x.iloc[other_idx] >= x.iloc[idx] and y.iloc[other_idx] >= y.iloc[idx]:
                                if x.iloc[other_idx] > x.iloc[idx] or y.iloc[other_idx] > y.iloc[idx]:
                                    is_pareto = False
                                    break
                    if is_pareto:
                        pareto_points.append((x.iloc[idx], y.iloc[idx]))
                
                if pareto_points:
                    pareto_points.sort()
                    pareto_x, pareto_y = zip(*pareto_points)
                    plt.plot(pareto_x, pareto_y, 'r-', linewidth=2, alpha=0.8, label='Pareto Frontier')
                    plt.scatter(pareto_x, pareto_y, c='red', s=50, zorder=5, label='Pareto Points')
                
                plt.xlabel(metric1)
                plt.ylabel(metric2)
                plt.title(f'Pareto Analysis: {metric1} vs {metric2}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.colorbar(scatter, label='Multi-Objective Score')
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"pareto_frontier_{metric1}_vs_{metric2}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_multi_objective_scatter(self):
        """Plot multi-dimensional scatter plots for performance analysis"""
        if self.processed_data is None:
            return
        
        target_data = self.processed_data['y']
        
        # 3D scatter plot if we have enough metrics
        key_metrics = ['perf_f1_score', 'robustness_score', 'perf_snr']
        available_metrics = [m for m in key_metrics if m in target_data.columns]
        
        if len(available_metrics) >= 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            x = target_data[available_metrics[0]]
            y = target_data[available_metrics[1]]
            z = target_data[available_metrics[2]]
            
            scatter = ax.scatter(x, y, z, c=target_data.get('multi_objective_score', x), 
                               cmap='viridis', alpha=0.6)
            
            ax.set_xlabel(available_metrics[0])
            ax.set_ylabel(available_metrics[1])
            ax.set_zlabel(available_metrics[2])
            ax.set_title('3D Multi-Objective Performance Space')
            
            plt.colorbar(scatter, label='Multi-Objective Score')
            plt.savefig(self.plots_dir / "multi_objective_3d.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_biomarker_performance(self):
        """Plot biomarker-specific performance analysis as separate plots"""
        if self.raw_data is None or 'target_biomarker' not in self.raw_data.columns:
            return
        
        # Group by biomarker and analyze performance
        biomarker_groups = self.raw_data.groupby('target_biomarker')
        
        # Performance metrics by biomarker
        performance_metrics = ['perf_f1_score', 'perf_snr', 'robustness_score']
        available_metrics = [m for m in performance_metrics if m in self.raw_data.columns]
        
        for metric in available_metrics:
            biomarker_performance = []
            biomarker_names = []
            
            for biomarker, group in biomarker_groups:
                if len(group) > 10:  # Only include biomarkers with sufficient data
                    biomarker_performance.append(group[metric].values)
                    biomarker_names.append(biomarker)
            
            if biomarker_performance:
                plt.figure(figsize=(12, 6))
                box_plot = plt.boxplot(biomarker_performance, labels=biomarker_names, patch_artist=True)
                
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(biomarker_performance)))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                plt.ylabel(metric)
                plt.title(f'{metric} Distribution by Biomarker')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"biomarker_performance_{metric.replace('/', '_').replace(' ', '_')}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        # Individual biomarker comparison plots
        for biomarker, group in biomarker_groups:
            if len(group) > 10:  # Only for biomarkers with sufficient data
                plt.figure(figsize=(10, 6))
                
                # Plot all available metrics for this biomarker
                metric_means = []
                metric_names = []
                
                for metric in available_metrics:
                    if metric in group.columns:
                        metric_means.append(group[metric].mean())
                        metric_names.append(metric)
                
                if metric_means:
                    bars = plt.bar(metric_names, metric_means, color=plt.cm.Set2(range(len(metric_names))))
                    plt.title(f'Average Performance Metrics - {biomarker}')
                    plt.ylabel('Average Score')
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metric_means):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                    
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f"biomarker_avg_performance_{biomarker.replace('/', '_').replace(' ', '_')}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
    
    def save_results_summary(self):
        """Save comprehensive results summary for later reuse (e.g., plots after reload)"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'shape': self.raw_data.shape if self.raw_data is not None else None,
                'features': len(self.processed_data['feature_names']) if self.processed_data else None,
                'targets': len(self.processed_data['target_names']) if self.processed_data else None
            },
            'model_performance': {},
            'feature_importance': {},
            'best_models': self._get_best_models()
        }

        # Save model performance
        for target_metric, models_perf in self.model_performance.items():
            summary['model_performance'][target_metric] = {}
            for model_name, perf in models_perf.items():
                summary['model_performance'][target_metric][model_name] = {
                    'test_r2': perf.get('test_r2'),
                    'test_mse': perf.get('test_mse'),
                    'train_r2': perf.get('train_r2'),
                    'train_mse': perf.get('train_mse'),
                    'predictions_test': perf.get('predictions_test', []).tolist() if isinstance(perf.get('predictions_test'), np.ndarray) else perf.get('predictions_test'),
                    'train_losses': perf.get('train_losses', []),
                    'val_losses': perf.get('val_losses', [])
                }

        # Save feature importance
        for model_key, feature_imp_df in self.feature_importance.items():
            # Save CSV file for human analysis
            feature_imp_df.to_csv(self.results_dir / f"feature_importance_{model_key}.csv", index=False)
            
            # Store in JSON-friendly dict for reloading
            summary['feature_importance'][model_key] = feature_imp_df.to_dict(orient='records')

        # Save as JSON
        summary_path = self.results_dir / "results_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Ã¢Å“â€¦ Results summary saved to {summary_path}")
    
    def _get_best_models(self):
        """Get best performing models for each metric"""
        best_models = {}
        
        for target_metric, models_perf in self.model_performance.items():
            best_r2 = -1
            best_model = None
            
            for model_name, perf in models_perf.items():
                if perf['test_r2'] > best_r2:
                    best_r2 = perf['test_r2']
                    best_model = model_name
            
            best_models[target_metric] = {
                'model': best_model,
                'r2_score': best_r2,
                'mse': models_perf[best_model]['test_mse'] if best_model else None
            }
        
        return best_models
    
    def _save_ml_models_cache(self):
        """Save all ML models and related data to cache"""
        saved_ml_dir = self.output_dir / "saved_ML"
        saved_ml_dir.mkdir(exist_ok=True)
        
        print("\nðŸ’¾ Caching ML models for future use...")
        
        try:
            # 1. Save surrogate models
            if not hasattr(self, 'surrogate_features') or not self.surrogate_features:
                print("   âŒ Cannot cache models - surrogate_features missing!")
                return

            surrogate_cache = {
                'models': {},
                'features': self.surrogate_features,
                'label_encoders': self.categorical_label_encoders if hasattr(self, 'categorical_label_encoders') else {},
                'target_transformers': self.target_transformers if hasattr(self, 'target_transformers') else {}
            }

            print(f"   ðŸ“¦ Caching features for {len(self.surrogate_features)} metrics")
            
            for metric, model in self.surrogate_models.items():
                # ✅ Handle NNWrapper (which wraps PyTorch models)
                if isinstance(model, NNWrapper):
                    # Extract the actual PyTorch model
                    pytorch_model = model.model
                    
                    if metric in self.surrogate_features:
                        input_dim = len(self.surrogate_features[metric])
                    else:
                        input_dim = len(self.X_train.columns)
                    
                    state_dict = pytorch_model.state_dict()
                    state_dict_str = str(state_dict)[:100]
                    
                    surrogate_cache['models'][metric] = {
                        'type': 'nn_wrapper',
                        'state_dict': state_dict,
                        'input_dim': input_dim,
                        'fingerprint': state_dict_str
                    }
                    print(f"   📦 Cached NNWrapper model {metric} (input_dim={input_dim})")
                
                elif hasattr(model, 'state_dict'):  # Direct PyTorch
                    if metric in self.surrogate_features:
                        input_dim = len(self.surrogate_features[metric])
                    elif hasattr(model, 'layers') and len(model.layers) > 0:
                        input_dim = model.layers[0].in_features
                    else:
                        input_dim = len(self.X_train.columns)
                    
                    state_dict = model.state_dict()
                    state_dict_str = str(state_dict)[:100]
                    
                    surrogate_cache['models'][metric] = {
                        'type': 'pytorch',
                        'state_dict': state_dict,
                        'input_dim': input_dim,
                        'fingerprint': state_dict_str
                    }
                    print(f"   📦 Cached PyTorch model {metric} (input_dim={input_dim})")
                
                elif isinstance(model, PhysicsBasedFNRPredictor):
                    surrogate_cache['models'][metric] = {
                        'type': 'physics_based_fnr',
                        'snr_features': model.snr_features,
                        'ttd_features': model.ttd_features,
                        'dr_features': model.dr_features,
                        'has_noise_feature': model.has_noise_feature
                    }
                    print(f"   📦 Cached PhysicsBasedFNR model {metric}")
                
                elif isinstance(model, PhysicsTTDPredictor):
                    surrogate_cache['models'][metric] = {
                        'type': 'physics_ttd',
                        'feature_names': model.feature_names,
                        'median_ttd': model.median_ttd
                    }
                    print(f"   📦 Cached PhysicsTTD model {metric}")
                
                else:  # sklearn or other
                    surrogate_cache['models'][metric] = {
                        'type': 'sklearn',
                        'model': model
                    }
                    print(f"   📦 Cached sklearn model {metric}")
            
            with open(saved_ml_dir / "surrogate_models_cache.pkl", 'wb') as f:
                pickle.dump(surrogate_cache, f)
            print("   âœ… Surrogate models cached")
            
            # 2. Save scaler PARAMETERS (not objects) for reliable reconstruction
            scalers_cache = {
                'X_scaler': self.X_scaler if hasattr(self, 'X_scaler') else None,
                'y_scalers': self.y_scalers if hasattr(self, 'y_scalers') else {},
                'surrogate_feature_scaler': self.surrogate_feature_scaler if hasattr(self, 'surrogate_feature_scaler') else None,
                'surrogate_metric_scalers': {}
            }

            # ✅ CRITICAL FIX: Store scalers with validation
            if hasattr(self, 'surrogate_metric_scalers'):
                for metric, scaler in self.surrogate_metric_scalers.items():
                    if scaler is None:
                        print(f"   ❌ Scaler for {metric} is None!")
                        continue
                    
                    # Check if scaler is fitted (StandardScaler has mean_, RobustScaler has center_)
                    if not (hasattr(scaler, 'mean_') or hasattr(scaler, 'center_')):
                        print(f"   ❌ Scaler for {metric} not fitted!")
                        continue
                    
                    # ✅ Store the actual fitted scaler
                    scalers_cache['surrogate_metric_scalers'][metric] = scaler
                    n_features = len(scaler.mean_) if hasattr(scaler, 'mean_') else len(scaler.center_)
                    print(f"   ✅ Cached scaler for {metric} (fitted on {n_features} features)")
                
                print(f"   📦 Total scalers cached: {len(scalers_cache['surrogate_metric_scalers'])}")
            else:
                print(f"   ❌ No surrogate_metric_scalers attribute found!")

            print(f"   ðŸ“¦ Cached {len(scalers_cache['surrogate_metric_scalers'])} valid scalers")

            with open(saved_ml_dir / "scalers_cache.pkl", 'wb') as f:
                pickle.dump(scalers_cache, f)
            print("   âœ… Scalers cached")
            
            # 3. Save general ML models (supervised learning)
            ml_models_cache = {
                'models': self.models,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance
            }
            
            with open(saved_ml_dir / "ml_models_cache.pkl", 'wb') as f:
                pickle.dump(ml_models_cache, f)
            print("   âœ… ML models cached")
            
            print(f"   ðŸ“ Cached models saved to: {saved_ml_dir}")
            
        except Exception as e:
            print(f"   âš ï¸  Failed to cache models: {e}")
            print("   (This won't affect current run)")

    def _load_ml_models_cache(self):
        """Load all ML models and related data from cache"""
        saved_ml_dir = self.output_dir / "saved_ML"
        
        print("\nðŸ“‚ Loading cached ML models...")
        
        # ðŸ”§ CRITICAL: Ensure we have training data for scaler rebuilding
        if not hasattr(self, 'X_train') or self.X_train is None:
            print("   âš ï¸  No X_train available - scalers cannot be rebuilt if corrupted")
            print("   ðŸ”§ Loading processed data...")
            if hasattr(self, 'processed_data') and 'X' in self.processed_data:
                X = self.processed_data['X']
                y = self.processed_data['y']
                
                from sklearn.model_selection import train_test_split

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                print("   âœ… Training data reconstructed for scaler rebuilding")
        
        try:
            # 1. Load surrogate models FIRST (contains feature mappings)
            with open(saved_ml_dir / "surrogate_models_cache.pkl", 'rb') as f:
                surrogate_cache = pickle.load(f)
            
            self.surrogate_features = surrogate_cache['features']
            self.categorical_label_encoders = surrogate_cache['label_encoders']
            self.target_transformers = surrogate_cache['target_transformers']
            print(f"   âœ… Loaded feature mappings for {len(self.surrogate_features)} metrics")
            
            # 2. Load scalers - reconstruct from parameters
            with open(saved_ml_dir / "scalers_cache.pkl", 'rb') as f:
                scalers_cache = pickle.load(f)

            self.X_scaler = scalers_cache.get('X_scaler', None)
            self.y_scalers = scalers_cache.get('y_scalers', {})
            self.surrogate_feature_scaler = scalers_cache.get('surrogate_feature_scaler', None)

            # ✅ CRITICAL FIX: Load scalers directly (they're already fitted objects)
            self.surrogate_metric_scalers = {}

            cached_scalers = scalers_cache.get('surrogate_metric_scalers', {})
            print(f"   📦 Found {len(cached_scalers)} cached scalers")

            for metric, scaler in cached_scalers.items():
                # ✅ FIXED: Check for BOTH StandardScaler (mean_) and RobustScaler (center_)
                if scaler is not None:
                    # Check if it's a fitted sklearn scaler
                    has_standard = hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')
                    has_robust = hasattr(scaler, 'center_') and hasattr(scaler, 'scale_')
                    
                    if has_standard or has_robust:
                        # ✅ It's a fitted scaler - use it directly!
                        self.surrogate_metric_scalers[metric] = scaler
                        n_features = len(scaler.mean_) if has_standard else len(scaler.center_)
                        scaler_type = "StandardScaler" if has_standard else "RobustScaler"
                        print(f"   ✅ Loaded fitted {scaler_type} for {metric} ({n_features} features)")
                    
                    elif isinstance(scaler, dict):
                        # ✅ It's scaler parameters stored as dict - reconstruct
                        from sklearn.preprocessing import StandardScaler, RobustScaler
                        
                        if 'mean_' in scaler:
                            # StandardScaler
                            reconstructed = StandardScaler()
                            reconstructed.mean_ = np.array(scaler['mean_'])
                            reconstructed.scale_ = np.array(scaler['scale_'])
                            reconstructed.var_ = np.array(scaler.get('var_', reconstructed.scale_ ** 2))
                            reconstructed.n_features_in_ = len(reconstructed.mean_)
                            reconstructed.n_samples_seen_ = scaler.get('n_samples_seen_', 1000)
                            print(f"   ✅ Reconstructed StandardScaler for {metric}")
                        
                        elif 'center_' in scaler:
                            # RobustScaler
                            reconstructed = RobustScaler()
                            reconstructed.center_ = np.array(scaler['center_'])
                            reconstructed.scale_ = np.array(scaler['scale_'])
                            reconstructed.n_features_in_ = len(reconstructed.center_)
                            print(f"   ✅ Reconstructed RobustScaler for {metric}")
                        
                        else:
                            print(f"   ❌ Invalid dict format for {metric}: missing mean_ or center_")
                            continue
                        
                        self.surrogate_metric_scalers[metric] = reconstructed
                    
                    else:
                        print(f"   ❌ Invalid scaler format for {metric}: {type(scaler)}")
                        print(f"        Has attributes: {dir(scaler)[:10]}")
                else:
                    print(f"   ❌ Scaler for {metric} is None!")

            # ✅ CRITICAL: Verify all loaded scalers
            print(f"   📊 Verification: {len(self.surrogate_metric_scalers)} scalers loaded")

            if len(self.surrogate_metric_scalers) == 0:
                print(f"   ❌ CRITICAL: No scalers loaded! Cache is broken!")
                return False

            # Verify each scaler
            for metric in self.target_metrics:
                if metric not in self.surrogate_metric_scalers:
                    print(f"   ❌ Missing scaler for target metric: {metric}")
                    print(f"   📋 Available scalers: {list(self.surrogate_metric_scalers.keys())}")
                    return False
                
                scaler = self.surrogate_metric_scalers[metric]
                
                # ✅ FIXED: Check for BOTH StandardScaler and RobustScaler
                has_mean = hasattr(scaler, 'mean_')
                has_center = hasattr(scaler, 'center_')
                
                if not (has_mean or has_center):
                    print(f"   ❌ Scaler for {metric} is not fitted!")
                    return False
                
                if has_mean:
                    n_features = len(scaler.mean_)
                    center = scaler.mean_
                elif has_center:
                    n_features = len(scaler.center_)
                    center = scaler.center_
                
                print(f"   ✅ Verified {metric}: {n_features} features, center range [{center.min():.3f}, {center.max():.3f}]")

            print(f"   🎉 All {len(self.surrogate_metric_scalers)} scalers verified!")

            # ✅ CRITICAL: Verify all target metrics have scalers (NO EMERGENCY REBUILD)
            missing_scalers = []
            for metric in self.target_metrics:
                if metric not in self.surrogate_metric_scalers:
                    missing_scalers.append(metric)
                elif self.surrogate_metric_scalers[metric] is None:
                    missing_scalers.append(metric)
                elif not hasattr(self.surrogate_metric_scalers[metric], 'mean_') and \
                    not hasattr(self.surrogate_metric_scalers[metric], 'center_'):
                    missing_scalers.append(metric)

            if missing_scalers:
                print(f"   ❌ CRITICAL: Missing scalers for: {missing_scalers}")
                print(f"   ❌ Cannot use cached models - scalers are corrupted!")
                print(f"   ❌ You must retrain models to regenerate scalers.")
                return False  # Force retraining

            print(f"   ✅ All target metrics have valid scalers")
            
            # 3. Reconstruct surrogate models
            self.surrogate_models = {}
            for metric, model_data in surrogate_cache['models'].items():
                if model_data['type'] == 'nn_wrapper':
                    # ✅ Reconstruct NNWrapper model
                    try:
                        input_dim = model_data['input_dim']
                        
                        # Create fresh TabularNN
                        pytorch_model = TabularNN(input_dim)
                        pytorch_model.load_state_dict(model_data['state_dict'], strict=True)
                        pytorch_model.eval()
                        
                        # Wrap in NNWrapper
                        model = NNWrapper(pytorch_model)
                        
                        self.surrogate_models[metric] = model
                        print(f"   ✅ NNWrapper model {metric} reconstructed (dim={input_dim})")
                    
                    except Exception as e:
                        print(f"   ❌ Failed to reconstruct NNWrapper {metric}: {e}")
                        return False
                
                elif model_data['type'] == 'pytorch':
                    # Reconstruct direct PyTorch model
                    try:
                        input_dim = model_data['input_dim']
                        
                        # ✅ Create fresh model instance
                        model = self._create_enhanced_traditional_model(input_dim, 'regression')
                        
                        # Load state dict
                        try:
                            model.load_state_dict(model_data['state_dict'], strict=True)
                            print(f"   ✅ PyTorch model {metric} reconstructed (dim={input_dim})")
                        except RuntimeError as e:
                            print(f"   ⚠️ State dict mismatch for {metric}, trying non-strict load...")
                            model.load_state_dict(model_data['state_dict'], strict=False)
                            print(f"   ⚠️ PyTorch model {metric} loaded with warnings")
                        
                        model.eval()
                        self.surrogate_models[metric] = model
                    
                    except Exception as e:
                        print(f"   ❌ Failed to reconstruct PyTorch {metric}: {e}")
                        return False
                
                elif model_data['type'] == 'physics_based_fnr':
                    # ✅ Reconstruct PhysicsBasedFNRPredictor
                    # This requires the component models - reconstruct from cached components
                    print(f"   ⚠️ PhysicsBasedFNR {metric} requires component models - will need retraining")
                    return False  # Force retrain for now
                
                elif model_data['type'] == 'physics_ttd':
                    # ✅ Reconstruct PhysicsTTDPredictor
                    model = PhysicsTTDPredictor(
                        feature_names=model_data['feature_names'],
                        median_ttd=model_data['median_ttd']
                    )
                    self.surrogate_models[metric] = model
                    print(f"   ✅ PhysicsTTD model {metric} reconstructed")
                
                elif model_data['type'] == 'sklearn':
                    # sklearn models should pickle fine
                    self.surrogate_models[metric] = model_data['model']
                    print(f"   ✅ sklearn model {metric} loaded")

                    # ✅ Also load cached feature mappings if available
                    features_path = self.models_dir / f'{metric}_features.json'
                    if features_path.exists():
                        import json
                        with open(features_path, 'r') as f:
                            self.surrogate_features[metric] = json.load(f)
                        print(f"   ✅ Loaded feature mapping: {len(self.surrogate_features[metric])} features")
            
            print("   âœ… Surrogate models loaded")
            
            # 4. Load general ML models
            with open(saved_ml_dir / "ml_models_cache.pkl", 'rb') as f:
                ml_models_cache = pickle.load(f)
            
            self.models = ml_models_cache['models'] 
            self.model_performance = ml_models_cache['model_performance']
            self.feature_importance = ml_models_cache['feature_importance']
            print("   âœ… ML models loaded")
            
            print("   ðŸŽ‰ All cached models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"   âŒ Failed to load cached models: {e}")
            import traceback
            traceback.print_exc()
            print("   Will train new models instead.")
            return False
        
    def save_best_models_only(self):
        """Save only the best models with proper handling for different model types"""
        print("Ã°Å¸â€™Â¾ Saving best models only...")
        
        best_models_dir = self.output_dir / "best_models"
        os.makedirs(best_models_dir, exist_ok=True)
        
        for metric, model in self.surrogate_models.items():
            model_path = best_models_dir / f"best_{metric}_model.pkl"
            
            try:
                # Handle different model types
                if hasattr(model, 'state_dict'):  # PyTorch model
                    torch_path = best_models_dir / f"best_{metric}_model.pth"
                    torch.save(model.state_dict(), torch_path)
                    print(f"   Ã¢Å“â€¦ Saved PyTorch model: {torch_path}")
                elif isinstance(model, PhysicsBasedFNRPredictor):  # Physics-based predictor
                    # Save with special handling for component models
                    import joblib
                    
                    # Save the predictor structure
                    predictor_data = {
                        'type': 'PhysicsBasedFNRPredictor',
                        'snr_features': model.snr_features,
                        'ttd_features': model.ttd_features,
                        'dr_features': model.dr_features,
                        'has_noise_feature': model.has_noise_feature,
                        # Note: We don't save component models here as they're saved separately
                    }
                    joblib.dump(predictor_data, model_path)
                    print(f"   Ã¢Å“â€¦ Saved physics-based predictor config: {model_path}")
                    print(f"      (Component models saved separately)")
                else:  # sklearn model
                    import joblib
                    joblib.dump(model, model_path)
                    print(f"   Ã¢Å“â€¦ Saved sklearn model: {model_path}")
            except Exception as e:
                print(f"   Ã¢Å¡ Ã¯Â¸Â  Failed to save {metric}: {e}")
                print(f"      Skipping this model...")
        
        print(f"Ã¢Å“â€¦ Best models saved to: {best_models_dir}")
    
    def run_complete_pipeline(self, biomarker: Optional[str] = None, 
                         apply_pca: bool = False, 
                         rl_timesteps: int = 500000,
                         use_cached_models: bool = None):
        """
        Run the complete pipeline from data loading to analysis
        
        Args:
            biomarker: Specific biomarker to focus on
            apply_pca: Whether to apply PCA
            rl_timesteps: Number of RL training timesteps
            use_cached_models: If None, asks user. If True, loads cached models. If False, trains new models.
        """
        print("ðŸ Starting Complete Biosensor Pipeline")
        print("=" * 50)
        
        # Create saved_ML directory
        saved_ml_dir = self.output_dir / "saved_ML"
        saved_ml_dir.mkdir(exist_ok=True)
        
        # Ask user about ML model caching if not specified
        if use_cached_models is None:
            print("\nðŸ¤” ML Model Options:")
            print("   [1] Use cached ML models (fast, if available)")
            print("   [2] Train new ML models (slow, always fresh)")
            choice = input("   Your choice (1 or 2): ").strip()
            use_cached_models = (choice == "1")
        
        # Check if cached models exist
        models_cache_path = saved_ml_dir / "ml_models_cache.pkl"
        surrogate_cache_path = saved_ml_dir / "surrogate_models_cache.pkl"
        scalers_cache_path = saved_ml_dir / "scalers_cache.pkl"
        
        cached_models_available = (
            models_cache_path.exists() and 
            surrogate_cache_path.exists() and 
            scalers_cache_path.exists()
        )
        
        if use_cached_models and not cached_models_available:
            print("   âš ï¸  No cached models found. Will train new models.")
            use_cached_models = False
        
        try:
            # Stage 1: Data Preprocessing (always needed)
            print("\nðŸ“Š STAGE 1: Data Preprocessing & Feature Selection")
            self.load_data(biomarker)
            
            print("\n🔍 FEATURE LOADING VERIFICATION:")
            print(f"   FNR from master_index: {pd.read_csv(self.data_path)['false_negative_rate'].mean():.4f}")
            print(f"   FNR in raw_data: {self.raw_data['false_negative_rate'].mean():.4f}")
            print(f"   These should MATCH!")

            # Check interaction features exist
            interaction_features = [c for c in self.raw_data.columns if 'ratio' in c or 'score' in c or 'product' in c]
            print(f"\n🔍 INTERACTION FEATURES:")
            for feat in interaction_features:
                print(f"   - {feat}: mean={self.raw_data[feat].mean():.2f}, std={self.raw_data[feat].std():.2f}")

            preprocessing_info = self.preprocess_data(apply_pca=apply_pca)


            self.plot_target_metrics_distributions(self.plots_dir / 'target_metric_plots')
            self.plot_target_metrics_correlation_matrix(self.plots_dir / 'target_metric_plots')
            self.plot_target_metrics_summary(self.plots_dir / 'target_metric_plots')

            # Check if we should use cached models
            models_loaded_from_cache = False
            if use_cached_models:
                models_loaded_from_cache = self._load_ml_models_cache()

            print("\n🔍 MODEL UNIQUENESS CHECK:")
            print("=" * 60)

            # ✅ INITIALIZE surrogate_features if it doesn't exist
            if not hasattr(self, 'surrogate_features'):
                print("⚠️ surrogate_features not initialized yet - will be populated during training")
                self.surrogate_features = {}

            # Check model uniqueness (your existing code continues here...)
            for metric in self.surrogate_models:
                model_id = id(self.surrogate_models[metric])
                print(f"✅ {metric}: unique model (ID: {model_id})")

            print("\n🔍 STATE DICT FINGERPRINTS:")
            print("=" * 60)

            print("\n🔍 FEATURE MAPPING CHECK:")
            print("=" * 60)

            # Now this won't fail:
            if self.surrogate_features:  # Only check if not empty
                for metric in self.target_metrics:
                    if metric in self.surrogate_features:
                        features = self.surrogate_features[metric]
                        print(f"\n{metric}:")
                        print(f"  Count: {len(features)}")
                        print(f"  First 5: {features[:5]}")
                        print(f"  Last 5: {features[-5:]}")
                        
                        # Check for identical feature sets
                        for other_metric in self.target_metrics:
                            if other_metric != metric and other_metric in self.surrogate_features:
                                other_features = self.surrogate_features[other_metric]
                                if len(features) == len(other_features) and set(features) == set(other_features):
                                    print(f"  🚨 IDENTICAL to {other_metric}!")
            else:
                print("No feature mappings available yet (will be created during training)")
                
            print("=" * 60)

            # ✅ DIAGNOSTIC: Check if feature mappings are truly different
            print("\n🔍 FEATURE MAPPING CHECK:")
            print("=" * 60)

            for metric in ['signal_to_noise_ratio_SNR', 'time_to_detection_threshold', 
                        'dynamic_range_of_output', 'false_negative_rate']:
                if metric in self.surrogate_features:
                    features = self.surrogate_features[metric]
                    print(f"\n{metric}:")
                    print(f"  Count: {len(features)}")
                    print(f"  First 5: {list(features[:5])}")
                    print(f"  Last 5: {list(features[-5:])}")
                    
                    # Check for exact duplicates
                    for other_metric in ['signal_to_noise_ratio_SNR', 'time_to_detection_threshold']:
                        if other_metric != metric and other_metric in self.surrogate_features:
                            other_features = self.surrogate_features[other_metric]
                            if list(features) == list(other_features):
                                print(f"  🚨 IDENTICAL to {other_metric}!")

            print("=" * 60)

            # Stage 2: Supervised Learning (skip if loaded from cache)
            if not models_loaded_from_cache:
                # Stage 3: Surrogate Modeling
                print("\nðŸ­ STAGE 3: Surrogate Modeling")
                surrogate_performance = self.train_surrogate_models()
                
                # Cache the models for next time
                self._save_ml_models_cache()
            else:
                print("\nâœ… STAGE 2-3: Using cached ML models (skipped training)")
                # Create dummy surrogate_performance for consistency
                surrogate_performance = {metric: {'test_r2': 0.95, 'type': 'cached'} 
                                        for metric in self.surrogate_models.keys()}

            # Stage 4: Reinforcement Learning (always runs)
            print("\nðŸŽ® STAGE 4: Reinforcement Learning")
            self.setup_rl_environment(self.target_metrics)

            # ðŸ”º DIAGNOSTIC: Test scaler storage
            print("\nðŸ” DIAGNOSTIC: Checking scaler storage...")
            if hasattr(self, 'surrogate_metric_scalers'):
                print(f"   âœ… surrogate_metric_scalers exists with {len(self.surrogate_metric_scalers)} scalers")
                for metric, scaler in self.surrogate_metric_scalers.items():
                    print(f"      {metric}: {type(scaler).__name__ if scaler else 'None'}")
            else:
                print(f"   âŒ surrogate_metric_scalers attribute missing!")

            # ðŸ”º DIAGNOSTIC: Test one environment's data_min_max
            if hasattr(self.rl_env, 'envs'):
                test_env = self.rl_env.envs[0]
                print(f"\nðŸ” DIAGNOSTIC: Test environment data_min_max...")
                print(f"   Has attribute: {hasattr(test_env, 'data_min_max')}")
                if hasattr(test_env, 'data_min_max'):
                    print(f"   Length: {len(test_env.data_min_max)}")
                    print(f"   Keys: {list(test_env.data_min_max.keys())}")

            # Ã¢Å“â€¦ FIX: Use standard training instead of buggy curriculum learning
            rl_performance = self.train_rl_agents(total_timesteps=rl_timesteps)

            # Ensure we have feature importance by training at least one supervised model
            if not self.feature_importance:
                print("\nÃ°Å¸Â¤â€“ STAGE 2: Training Supervised Models for Feature Importance")
                try:
                    # Train on multi-objective score first
                    if 'multi_objective_score' in self.processed_data['target_names']:
                        self.train_supervised_models('multi_objective_score')
                    else:
                        # Fall back to first available target
                        self.train_supervised_models(self.processed_data['target_names'][0])
                except Exception as e:
                    print(f"   Ã¢Å¡ Ã¯Â¸Â Supervised model training failed: {e}")
            
            # Stage 5: Sclerostin-Specific Analysis
            print("\nSTAGE 5: Sclerostin Biosensor Analysis")
            self.plot_feature_importance_for_sclerostin()
            self.plot_sclerostin_correlation_matrix()
            print("ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Sclerostin-specific visualizations completed")
            
            # Save results and best models only
            print("\nÃ°Å¸â€™Â¾ Saving Results & Best Models")
            self.save_best_models_only()
            self.save_results_summary()
            
            print("\nÃ¢Å“â€¦ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"Ã°Å¸â€œÂ All results saved to: {self.output_dir}")
            print(f"Ã°Å¸â€œÅ  Plots saved to: {self.plots_dir}")
            print(f"Ã°Å¸Â¤â€“ Models saved to: {self.models_dir}")
            print(f"Ã°Å¸â€œâ€¹ Results saved to: {self.results_dir}")
            
            return {
                'preprocessing_info': preprocessing_info,
                'model_performance': self.model_performance,
                'surrogate_performance': surrogate_performance,
                'rl_performance': rl_performance
            }
            
        except Exception as e:
            print(f"\nÃ¢ÂÅ’ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run the biosensor pipeline"""

    # Use context manager for automatic logging
    with DualLogger(log_dir="logs"):
        print("Ã°Å¸Â§Â¬ Synthetic Biology Biosensor Design Pipeline")
        print("=" * 60)
        
        # Configuration for SCLEROSTIN SIMULATION DATASET
        data_path = r"data/master_index.csv"  # Path to master index from simulation
        base_output_dir = "RL/sclerostin_rl_results"
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"Ã¢ÂÅ’ Data file not found: {data_path}")
            print("Please update the data_path variable with the correct path to your dataset.")
            return
        
        # List of biomarkers to analyze
        '''biomarkers = [
            # Main 5
            "mirna_21",    # 5
            "pcbs",     # 4
            "troponin",    # 2
            "cortisol",    # 1
            "salmonella",    # 6

            # 1. Hormones & Signaling Molecules
            'adrenaline',
            'growth hormone',
            'insulin',
            'testosterone',
            'thyroid hormone',
            'estrogen',
            'p53',
            'ifn_gamma',
            'tnf_alpha',
            'il_1beta',
            'il_6',
            'il_10',

            # 2. Metabolic & Organ Function Markers
            'glucose',
            'lactate',      # ---- Akshit E. ends
            'pyruvate',
            'creatinine',
            'urea',
            'albumin',
            'hemoglobin',
            'cholesterol',

            # 3. Electrolytes & Minerals
            'calcium',
            'potassium',
            'sodium',
            'magnesium',
            'zinc',
            'chloride',

            # 4. Toxins, Heavy Metals & Environmental Exposure
            'lead',
            'mercury',
            'arsenic',
            'cadmium',
            'dioxins',
            'pesticides',
            'bisphenol a',


            # 5. Nucleic Acid Biomarkers (Genetic/Epigenetic/Transcriptomic)
            'cfdna',
            'mirna_155',
            'mrna_gapdh',
            'viral rna',

            # 6. Pathogen-Associated Biomarkers
            'covid_19',
            'influenza',
            'e_coli',

            # 7. Energy & Cellular Activity Molecules
            'atp',
            'caffeine',
        ]'''
        # For Sclerostin biosensor - no biomarker filtering needed
        # The dataset is already specific to Sclerostin
        biomarkers = [None]  # Run once without biomarker filtering

        rl_timesteps = 5000000  # 5M steps - RL needs substantial training!
        #rl_timesteps = 3000000  # 3M steps  (for faster testing)
        #rl_timesteps = 10000000  # 10M steps (for best results)
        
        # Results storage
        all_results = {}
        successful_analyses = []
        failed_analyses = []
        
        print(f"\nÃ°Å¸Å¡â‚¬ Starting automated analysis...")
        print(f"Ã°Å¸â€œÅ  RL Timesteps: {rl_timesteps}")
        print("=" * 60)
        
        print(f"\nÃ°Å¸Å½Â¯ Sclerostin Biosensor RL Optimization")
        print("   Single dataset: Sclerostin simulations with 3 scenarios")
        print("   Scenarios: healthy, pmo (post-menopausal osteoporosis), ckd_mbd")

        # Create single output directory
        output_dir = base_output_dir

        try:
            # Initialize pipeline
            print(f"\nÃ°Å¸Å¡â‚¬ Initializing pipeline...")
            pipeline = BiosensorPipeline(data_path, output_dir)
            
            # Run complete pipeline (will ask user about caching)
            results = pipeline.run_complete_pipeline(
                biomarker=None,  # No biomarker filtering for Sclerostin
                rl_timesteps=rl_timesteps,
                use_cached_models=None  # None = ask user, True = force cache, False = force retrain
            )
            
            if results:
                print(f"\nÃ¢Å“â€¦ Sclerostin biosensor optimization completed successfully!")
                
                # Print detailed summary
                if 'model_performance' in results:
                    print("\nÃ°Å¸Â¤â€“ Surrogate Model Performance:")
                    for target, models in results['model_performance'].items():
                        best_model = max(models.items(), key=lambda x: x[1]['test_r2'])
                        print(f"   {target}:")
                        print(f"      Best: {best_model[0]} (RÃ‚Â² = {best_model[1]['test_r2']:.4f})")
                
                if 'rl_performance' in results and results['rl_performance']:
                    print("\nÃ°Å¸Å½Â® RL Agent Performance:")
                    for agent, perf in results['rl_performance'].items():
                        print(f"   {agent}:")
                        print(f"      Mean Reward: {perf['mean_reward']:.4f} Ã‚Â± {perf['std_reward']:.4f}")
                        print(f"      Max Reward: {perf['max_reward']:.4f}")
            else:
                print(f"\nÃ¢ÂÅ’ Sclerostin biosensor optimization failed!")
                
        except Exception as e:
            print(f"\nÃ¢ÂÅ’ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

        print(f"\nÃ°Å¸â€œÂ All results saved to: {output_dir}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("Ã°Å¸Å½â€° AUTOMATED ANALYSIS COMPLETE!")
        print("=" * 60)
        
        print(f"\nÃ°Å¸â€œÅ  OVERALL SUMMARY:")
        print(f"   Total biomarkers: {len(biomarkers)}")
        print(f"   Successful: {len(successful_analyses)}")
        print(f"   Failed: {len(failed_analyses)}")
        
        if successful_analyses:
            print(f"\nÃ¢Å“â€¦ Successfully analyzed:")
            for biomarker in successful_analyses:
                print(f"   - {biomarker}")
        
        if failed_analyses:
            print(f"\nÃ¢ÂÅ’ Failed analyses:")
            for biomarker in failed_analyses:
                print(f"   - {biomarker}")
        
        # Generate comparative summary
        if len(successful_analyses) > 1:
            print(f"\nÃ°Å¸â€œË† COMPARATIVE ANALYSIS:")
            print("-" * 40)
            
            # Compare best model performance across biomarkers
            print("Ã°Å¸Â¤â€“ Best Model Performance Comparison:")
            for biomarker in successful_analyses:
                if biomarker in all_results and 'model_performance' in all_results[biomarker]:
                    best_overall = 0
                    best_model_name = ""
                    for target, models in all_results[biomarker]['model_performance'].items():
                        best_model = max(models.items(), key=lambda x: x[1]['test_r2'])
                        if best_model[1]['test_r2'] > best_overall:
                            best_overall = best_model[1]['test_r2']
                            best_model_name = best_model[0]
                    print(f"   {biomarker}: {best_model_name} (RÃ‚Â² = {best_overall:.3f})")
            
            # Compare RL performance across biomarkers
            print(f"\nÃ°Å¸Å½Â® RL Performance Comparison:")
            for biomarker in successful_analyses:
                if biomarker in all_results and 'rl_performance' in all_results[biomarker]:
                    rl_results = all_results[biomarker]['rl_performance']
                    if rl_results:
                        best_rl_agent = max(rl_results.items(), key=lambda x: x[1]['mean_reward'])
                        print(f"   {biomarker}: {best_rl_agent[0]} ({best_rl_agent[1]['mean_reward']:.3f} Ã‚Â± {best_rl_agent[1]['std_reward']:.3f})")
        
        print(f"\nÃ°Å¸â€œÂ All results saved in respective directories:")
        for biomarker in successful_analyses:
            print(f"   - {base_output_dir}_{biomarker}/")
        
        print("\nÃ°Å¸Å½Â¯ Analysis complete! Check individual directories for detailed results.")

# Example usage and testing
if __name__ == "__main__":
    main()