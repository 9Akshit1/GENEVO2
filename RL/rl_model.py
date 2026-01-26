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
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import lightgbm as lgb
from catboost import CatBoostRegressor
# Add robust preprocessing
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression
from sklearn.decomposition import PCA
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
                    
                    # Additional useful features
                    'sclerostin_max': metadata['sclerostin_max'],
                    'sclerostin_std': metadata['sclerostin_std'],
                }
                
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
        
        # Calculate false_negative_rate (synthesized from detection_rate)
        # FNR = 1 - detection_rate (simplified approximation)
        self.raw_data['false_negative_rate'] = 1.0 - self.raw_data['detection_rate']
        
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
        
        # Choose scaler based on data characteristics
        if outlier_percentage > 10 or feature_skewness > 2:
            print("   Using RobustScaler due to outliers/skewness")
            self.X_scaler = RobustScaler()
        elif feature_skewness > 1:
            print("   Using QuantileTransformer for moderate skewness")
            self.X_scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(self.X_train)))
        else:
            print("   Using StandardScaler for normal distribution")
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
                
                # 6. Quantile transformation
                try:
                    qt_transformer = QuantileTransformer(output_distribution='normal', 
                                                    n_quantiles=min(1000, len(train_data)))
                    train_qt = qt_transformer.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                    transformations.append(comprehensive_transformation_evaluation(
                        train_data.values, train_qt, "QuantileTransform"
                    ))
                except Exception as e:
                    print(f"   QuantileTransform failed: {e}")

                # 7. Robust transformation (for already-normal data with outliers)
                try:
                    from sklearn.preprocessing import RobustScaler
                    robust_scaler = RobustScaler()
                    train_robust = robust_scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                    transformations.append(comprehensive_transformation_evaluation(
                        train_data.values, train_robust, "RobustScale"
                    ))
                except Exception as e:
                    print(f"   RobustScale failed: {e}")

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
        
                    # Apply the best transformation
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
                        
                    elif best_transform['transform'] == "QuantileTransform":
                        transformer = QuantileTransformer(output_distribution='normal', 
                                                    n_quantiles=min(1000, len(train_data)))
                        transformed_train = transformer.fit_transform(train_data.values.reshape(-1, 1)).flatten()
                        transformed_test = transformer.transform(test_data.values.reshape(-1, 1)).flatten()
                        self.y_transformers[col] = transformer
                
                # Apply scaling to transformed data
                target_std = np.std(transformed_train)
                target_range = np.max(transformed_train) - np.min(transformed_train)
                
                # Choose appropriate scaler for target
                if target_std > 1000 or target_range > 10000:
                    scaler = RobustScaler()
                    print(f"      Using RobustScaler for target")
                else:
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

    def _create_robust_mlp_model(self, input_dim: int) -> nn.Module:
        """Create a robust MLP model based on input dimensions"""
        
        class RobustMLP(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                # Adaptive architecture based on input size
                if input_dim < 50:
                    hidden_dims = [128, 64, 32]
                elif input_dim < 200:
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
        
        return RobustMLP(input_dim)

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

    def _handle_problematic_targets(self, target_metric, y_train_target, y_test_target):
        """Handle targets with problematic distributions"""
        
        # SPECIAL HANDLING FOR FALSE NEGATIVE RATE
        if 'false_negative' in target_metric.lower():
            print(f"      Ã°Å¸Å½Â¯ Special handling for False Negative Rate...")
            
            # Check if it's a binary classification problem disguised as regression
            unique_values = len(np.unique(y_train_target))
            if unique_values <= 10:  # Likely categorical
                print(f"      Ã°Å¸â€™Â¡ Converting to classification task ({unique_values} unique values)")
                # Return as-is and flag for classification
                return y_train_target, y_test_target, False
            
            # Check if values are bounded [0, 1] - apply logit transformation
            if y_train_target.min() >= 0 and y_train_target.max() <= 1:
                print(f"      Ã°Å¸â€Â§ Applying logit transformation for bounded [0,1] data")
                
                # Clip to avoid log(0) and log(1)
                epsilon = 1e-7
                y_train_clipped = np.clip(y_train_target, epsilon, 1 - epsilon)
                y_test_clipped = np.clip(y_test_target, epsilon, 1 - epsilon)
                
                # Logit transformation: log(p / (1-p))
                y_train_transformed = np.log(y_train_clipped / (1 - y_train_clipped))
                y_test_transformed = np.log(y_test_clipped / (1 - y_test_clipped))
                
                return y_train_transformed, y_test_transformed, False
        
        # Check for constant targets (like perf_specificity with all 0.0)
        if len(np.unique(y_train_target)) <= 1:
            print(f"      Ã¢Å¡ Ã¯Â¸Â  CRITICAL: Target '{target_metric}' has constant/near-constant values!")
            print(f"      Unique values: {np.unique(y_train_target)}")
            
            # Check if this is a data issue or expected
            if np.all(y_train_target == 0.0):
                print(f"      Ã°Å¸â€™Â¡ All values are 0.0 - this might indicate:")
                print(f"         - Calculation error in the target metric")
                print(f"         - Missing/uninitialized data")
                print(f"         - Metric not applicable to current dataset")
                print(f"      Ã°Å¸â€Â§ SKIPPING this target as it's not learnable")
                return None, None, True  # Skip this target
            
        # Check for targets with very low variance
        y_std = np.std(y_train_target)
        y_range = np.max(y_train_target) - np.min(y_train_target)
        
        if y_std < 1e-10 or y_range < 1e-10:
            print(f"      Ã¢Å¡ Ã¯Â¸Â  Target has extremely low variance (std: {y_std:.2e}, range: {y_range:.2e})")
            print(f"      Ã°Å¸â€Â§ SKIPPING this target as it's not learnable")
            return None, None, True
        
        # Check for targets with extreme outliers
        q1, q3 = np.percentile(y_train_target, [25, 75])
        iqr = q3 - q1
        outlier_threshold = 3.0
        
        if iqr > 0:
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            outliers = np.sum((y_train_target < lower_bound) | (y_train_target > upper_bound))
            outlier_ratio = outliers / len(y_train_target)
            
            if outlier_ratio > 0.1:  # More than 10% outliers
                print(f"      Ã¢Å¡ Ã¯Â¸Â  High outlier ratio: {outlier_ratio:.2%} ({outliers} outliers)")
                print(f"      Ã°Å¸â€Â§ Applying robust outlier handling...")
                
                # Cap outliers instead of removing them
                y_train_capped = np.clip(y_train_target, lower_bound, upper_bound)
                y_test_capped = np.clip(y_test_target, lower_bound, upper_bound)
                
                return y_train_capped, y_test_capped, False
        
        return y_train_target, y_test_target, False

    def _create_adaptive_ensemble_model(self, input_dim, task_type='regression', n_classes=None):
        """Create an ensemble of different model architectures"""
        
        class AdaptiveEnsemble(nn.Module):
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
                class ResidualBlock(nn.Module):
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
                    # Weighted average for regression - FIX: Don't use final_layer
                    ensemble_out = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
                    return ensemble_out  # REMOVED: self.final_layer(ensemble_out)
                else:
                    # Concatenate and process for classification
                    combined = torch.cat([out1, out2, out3], dim=1)
                    return self.final_layer(combined)
        
        return AdaptiveEnsemble(input_dim, task_type, n_classes)

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
        """Create enhanced traditional model with better architecture"""
        
        class EnhancedTraditional(nn.Module):
            def __init__(self, input_dim, task_type='regression', n_classes=None):
                super().__init__()
                
                # More sophisticated architecture
                self.input_norm = nn.BatchNorm1d(input_dim)
                
                # Progressive dimension reduction
                if input_dim < 50:
                    hidden_dims = [256, 128, 64, 32]
                elif input_dim < 150:
                    hidden_dims = [512, 256, 128, 64]
                else:
                    hidden_dims = [1024, 512, 256, 128, 64]
                
                layers = []
                prev_dim = input_dim
                
                for i, hidden_dim in enumerate(hidden_dims):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.GELU(),  # Better activation function
                        nn.Dropout(0.3 if i == 0 else 0.2)
                    ])
                    prev_dim = hidden_dim
                
                self.hidden_layers = nn.Sequential(*layers)
                
                # Output layer
                if task_type == 'regression':
                    self.output = nn.Sequential(
                        nn.Linear(prev_dim, prev_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(prev_dim // 2, 1)
                    )
                else:
                    self.output = nn.Sequential(
                        nn.Linear(prev_dim, prev_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(prev_dim // 2, n_classes)
                    )
                
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        nn.init.zeros_(m.bias)
            
            def forward(self, x):
                x = self.input_norm(x)
                x = self.hidden_layers(x)
                return self.output(x)
        
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

    def _train_single_model(self, model, criterion, X_train_final, y_train_processed, 
                        X_test_final, y_test_processed, strategy_name, is_classification):
        """Train a single model with enhanced training procedure"""
        
        # Enhanced optimizer with different learning rates for different parts
        optimizer = optim.AdamW([
            {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}
        ])
        
        # More sophisticated scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=800, steps_per_epoch=1,
            pct_start=0.1, anneal_strategy='cos'
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_final.values)
        X_test_tensor = torch.FloatTensor(X_test_final.values)
        
        if is_classification:
            y_train_tensor = torch.LongTensor(y_train_processed)
            y_test_tensor = torch.LongTensor(y_test_processed)
        else:
            y_train_tensor = torch.FloatTensor(y_train_processed).unsqueeze(1)
            y_test_tensor = torch.FloatTensor(y_test_processed).unsqueeze(1)
        
        # Training with gradient accumulation
        best_loss = float('inf')
        patience_counter = 0
        patience = 60
        gradient_accumulation_steps = 4
        
        for epoch in range(800):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Enhanced regularization
            l1_reg = torch.tensor(0.)
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
                l2_reg += torch.norm(param, 2)
            
            total_loss = loss + 1e-6 * l1_reg + 1e-5 * l2_reg
            
            # Gradient accumulation
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()
            
            if (epoch + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            scheduler.step()
            
            # Validation check
            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()
        
        with torch.no_grad():
            if is_classification:
                train_pred = torch.argmax(model(X_train_tensor), dim=1).numpy()
                test_pred = torch.argmax(model(X_test_tensor), dim=1).numpy()
                
                performance = {
                    'train_accuracy': accuracy_score(y_train_processed, train_pred),
                    'test_accuracy': accuracy_score(y_test_processed, test_pred),
                    'train_f1': f1_score(y_train_processed, train_pred, average='weighted'),
                    'test_f1': f1_score(y_test_processed, test_pred, average='weighted')
                }
                print(f"      Ã°Å¸Â¤â€“ {strategy_name}: Acc = {performance['test_accuracy']:.4f}")
            else:
                train_pred = model(X_train_tensor).numpy().flatten()
                test_pred = model(X_test_tensor).numpy().flatten()
                
                # Calculate normalized MSE
                train_norm_mse = self._calculate_normalized_mse(y_train_processed, train_pred)
                test_norm_mse = self._calculate_normalized_mse(y_test_processed, test_pred)
                
                performance = {
                    'train_r2': r2_score(y_train_processed, train_pred),
                    'test_r2': r2_score(y_test_processed, test_pred),
                    'train_mse': mean_squared_error(y_train_processed, train_pred),
                    'test_mse': mean_squared_error(y_test_processed, test_pred),
                    'train_normalized_mse': train_norm_mse,
                    'test_normalized_mse': test_norm_mse
                }
                print(f"      Ã°Å¸Â¤â€“ {strategy_name}: RÃ‚Â² = {performance['test_r2']:.4f}")
        
        return performance

    def train_surrogate_models(self) -> Dict:
        """
        ENHANCED VERSION: Train surrogate models with better handling of problematic targets
        """
        print("Ã°Å¸Å½Â­ Training ENHANCED surrogate models for RL...")
        print("=" * 60)
        
        # Initialize storage
        self.label_encoders = {}
        self.target_transformers = {}
        self.surrogate_feature_scalers = {}
        surrogate_performance = {}
        
        # Use the already prepared scaled features
        if not hasattr(self, 'X_train_scaled') or self.X_train_scaled is None:
            print("Ã¢Å¡ Ã¯Â¸Â  Scaled features not found, preparing data...")
            self._prepare_data_with_scaling()
        
        # Use enhanced features if available
        if hasattr(self, 'X_train_enhanced') and self.X_train_enhanced is not None:
            print("Ã°Å¸â€œÅ  Using enhanced feature set for surrogate models")
            X_train = self.X_train_enhanced.copy()
            X_test = self.X_test_enhanced.copy()
        else:
            print("Ã°Å¸â€œÅ  Using scaled feature set for surrogate models")
            X_train = self.X_train_scaled.copy()
            X_test = self.X_test_scaled.copy()
        
        # Apply additional scaling specifically for neural networks
        print("Ã°Å¸â€Â§ Applying neural network specific scaling...")
        
        # Use RobustScaler for surrogate models (more stable for RL)
        from sklearn.preprocessing import RobustScaler
        surrogate_scaler = RobustScaler()
        
        X_train_surrogate = pd.DataFrame(
            surrogate_scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_surrogate = pd.DataFrame(
            surrogate_scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.surrogate_feature_scaler = surrogate_scaler
        
        print(f"Ã°Å¸â€œË† Training surrogate models for {len(self.y_train.columns)} targets")
        print(f"   Feature shape: {X_train_surrogate.shape}")
        
        # Track skipped targets
        skipped_targets = []
        
        # CRITICAL: Train surrogates in dependency order
        # FNR depends on SNR, TTD, and DR, so train those first
        target_order = []
        fnr_targets = []

        for target in self.y_train.columns:
            if 'false_negative' in target.lower():
                fnr_targets.append(target)
            else:
                target_order.append(target)

        # Add FNR targets at the end
        target_order.extend(fnr_targets)

        print(f"Ã°Å¸â€œâ€¹ Training order (FNR components first): {target_order}")

        # Train a surrogate for each target in correct order
        for target_idx, target_metric in enumerate(target_order):
            print(f"\n🎯 Target {target_idx + 1}/{len(self.y_train.columns)}: {target_metric}")
            print("-" * 50)
            
            # ✅ GET TARGET DATA FIRST - before any processing
            y_train_target = self.y_train[target_metric].copy()
            y_test_target = self.y_test[target_metric].copy()
            
            # Handle missing values in targets
            if y_train_target.isnull().any():
                print(f"   Handling {y_train_target.isnull().sum()} missing target values...")
                if y_train_target.dtype == 'object':
                    y_train_target = y_train_target.fillna(y_train_target.mode()[0] if len(y_train_target.mode()) > 0 else 'unknown')
                    y_test_target = y_test_target.fillna(y_train_target.mode()[0] if len(y_train_target.mode()) > 0 else 'unknown')
                else:
                    median_val = y_train_target.median()
                    y_train_target = y_train_target.fillna(median_val)
                    y_test_target = y_test_target.fillna(median_val)
            
            # IMPROVED: Multi-strategy training for difficult targets
            if target_metric in ['time_to_detection_threshold', 'false_negative_rate']:
                print(f"   ðŸŽ¯ DIFFICULT TARGET: Using ensemble strategies for {target_metric}")

                # Prepare data for ensemble strategies
                X_train_final = X_train_surrogate.copy()
                X_test_final = X_test_surrogate.copy()
                y_train_final = y_train_target.values if hasattr(y_train_target, 'values') else y_train_target
                y_test_final = y_test_target.values if hasattr(y_test_target, 'values') else y_test_target
                
                # Strategy 1: XGBoost with aggressive tuning
                # Get target-specific config
                config = self._get_target_specific_config(target_metric) if hasattr(self, '_get_target_specific_config') else {
                    'xgb_depth': 8,
                    'lgb_leaves': 63,
                    'learning_rate': 0.02,
                    'n_estimators': 2000
                }

                xgb_model = xgb.XGBRegressor(
                    n_estimators=config['n_estimators'],
                    max_depth=config['xgb_depth'],
                    learning_rate=config['learning_rate'],
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,      # ðŸ”º DECREASED regularization
                    reg_lambda=0.5,     # ðŸ”º DECREASED regularization
                    gamma=0.05,         # ðŸ”º DECREASED from 0.1
                    min_child_weight=1, # ðŸ”º DECREASED from 2
                    random_state=42,
                    n_jobs=-1
                )
                
                xgb_model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_test_final, y_test_final)],
                    verbose=False
                )
                
                # Strategy 2: LightGBM with different hyperparams
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=config['n_estimators'],
                    max_depth=config['xgb_depth'],
                    learning_rate=config['learning_rate'],
                    num_leaves=config['lgb_leaves'],
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.5,
                    min_child_samples=5, # ðŸ”º DECREASED from 10
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
                
                X_train_clean = self._clean_feature_names(X_train_final)
                X_test_clean = self._clean_feature_names(X_test_final)
                
                lgb_model.fit(
                    X_train_clean, y_train_final,
                    eval_set=[(X_test_clean, y_test_final)],
                    callbacks=[lgb.log_evaluation(0)]
                )
                
                # Strategy 3: Stacking ensemble
                from sklearn.ensemble import StackingRegressor
                from sklearn.linear_model import Ridge
                
                stacked_model = StackingRegressor(
                    estimators=[
                        ('xgb', xgb_model),
                        ('lgb', lgb_model)
                    ],
                    final_estimator=Ridge(alpha=1.0),
                    cv=5
                )
                
                stacked_model.fit(X_train_final, y_train_final)
                
                # Evaluate all three and pick best
                xgb_pred = xgb_model.predict(X_test_final)
                lgb_pred = lgb_model.predict(X_test_clean)
                stack_pred = stacked_model.predict(X_test_final)
                
                xgb_r2 = r2_score(y_test_final, xgb_pred)
                lgb_r2 = r2_score(y_test_final, lgb_pred)
                stack_r2 = r2_score(y_test_final, stack_pred)
                
                print(f"      ðŸŽ¯ XGBoost RÂ²: {xgb_r2:.4f}")
                print(f"      ðŸŽ¯ LightGBM RÂ²: {lgb_r2:.4f}")
                print(f"      ðŸŽ¯ Stacking RÂ²: {stack_r2:.4f}")
                
                # Use best model
                if stack_r2 >= max(xgb_r2, lgb_r2):
                    final_model = stacked_model
                    best_r2 = stack_r2
                    print(f"      âœ… Selected: Stacking Ensemble")
                elif xgb_r2 >= lgb_r2:
                    final_model = xgb_model
                    best_r2 = xgb_r2
                    print(f"      âœ… Selected: XGBoost")
                else:
                    final_model = lgb_model
                    best_r2 = lgb_r2
                    print(f"      âœ… Selected: LightGBM")
                
                final_performance = {
                    'train_r2': r2_score(y_train_final, final_model.predict(X_train_final if not isinstance(final_model, lgb.LGBMRegressor) else X_train_clean)),
                    'test_r2': best_r2,
                    'test_mse': mean_squared_error(y_test_final, final_model.predict(X_test_final if not isinstance(final_model, lgb.LGBMRegressor) else X_test_clean)),
                    'type': 'regression'
                }
            else:
                # Standard handling for other metrics
                y_train_processed, y_test_processed, should_skip = self._handle_problematic_targets(
                    target_metric, y_train_target, y_test_target
                )
            
            if should_skip:
                print(f"   Ã¢ÂÂ­Ã¯Â¸Â  Skipping target '{target_metric}' - not learnable")
                skipped_targets.append(target_metric)
                continue
            
            # Determine if classification or regression
            is_classification = y_train_target.dtype == 'object' or y_train_target.dtype.name == 'category'
            
            if is_classification:
                print(f"   Ã°Å¸â€œÂ Classification task detected")
                
                # Encode categorical targets
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                
                # Fit on combined data to handle all possible labels
                combined_targets = pd.concat([pd.Series(y_train_processed), pd.Series(y_test_processed)], ignore_index=True)
                le.fit(combined_targets)
                
                y_train_final = le.transform(y_train_processed)
                y_test_final = le.transform(y_test_processed)
                
                self.label_encoders[target_metric] = le
                n_classes = len(le.classes_)
                
                print(f"      Classes ({n_classes}): {list(le.classes_)}")
                
            else:
                print(f"   Ã°Å¸â€œÅ  Regression task detected")
                
                # Convert to numeric
                y_train_numeric = pd.to_numeric(y_train_processed, errors='coerce')
                y_test_numeric = pd.to_numeric(y_test_processed, errors='coerce')
                
                # Handle any remaining NaN from conversion
                if y_train_numeric.isnull().any():
                    median_val = y_train_numeric.median()
                    y_train_numeric = y_train_numeric.fillna(median_val)
                    y_test_numeric = y_test_numeric.fillna(median_val)
                
                print(f"      Range: [{y_train_numeric.min():.4f}, {y_train_numeric.max():.4f}]")
                print(f"      Mean: {y_train_numeric.mean():.4f}, Std: {y_train_numeric.std():.4f}")
                print(f"      Skewness: {y_train_numeric.skew():.4f}")
                
                # Apply target transformation if needed
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
                            print("      Ã¢Å“â€¦ Target transformation applied")
                        else:
                            y_train_final = y_train_numeric.values
                            y_test_final = y_test_numeric.values
                            self.target_transformers[target_metric] = None
                            print("      Ã¢ÂÅ’ Target transformation didn't improve, using original")
                            
                    except Exception as e:
                        print(f"      Ã¢ÂÅ’ Target transformation failed: {e}")
                        y_train_final = y_train_numeric.values
                        y_test_final = y_test_numeric.values
                        self.target_transformers[target_metric] = None
                else:
                    y_train_final = y_train_numeric.values
                    y_test_final = y_test_numeric.values
                    self.target_transformers[target_metric] = None
                
                n_classes = 1  # For regression
            
            # Enhanced feature selection with per-metric scaling
            print(f"   ðŸ” Performing enhanced feature selection...")

            from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
            from sklearn.feature_selection import RFECV
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            try:
                # Use RFECV with Random Forest for better feature selection
                if is_classification:
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    min_features = max(5, min(20, len(X_train_surrogate.columns) // 4))
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    min_features = max(5, min(30, len(X_train_surrogate.columns) // 3))
                
                selector = RFECV(estimator, min_features_to_select=min_features, cv=3, scoring=None, n_jobs=-1)
                
                X_train_selected = selector.fit_transform(X_train_surrogate, y_train_final)
                X_test_selected = selector.transform(X_test_surrogate)
                
                selected_features = X_train_surrogate.columns[selector.support_]
                
                # Ã°Å¸"Âº CRITICAL FIX: Create a NEW scaler for THIS SPECIFIC metric
                from sklearn.preprocessing import RobustScaler
                metric_scaler = RobustScaler()

                # FIT the scaler
                metric_scaler.fit(X_train_selected)

                # VERIFY it was fitted properly
                if not hasattr(metric_scaler, 'center_'):
                    raise RuntimeError(f"Scaler failed to fit for {target_metric}!")

                X_train_final = pd.DataFrame(
                    metric_scaler.transform(X_train_selected),
                    columns=selected_features, 
                    index=X_train_surrogate.index
                )

                # STORE immediately after fitting
                if not hasattr(self, 'surrogate_metric_scalers'):
                    self.surrogate_metric_scalers = {}
                self.surrogate_metric_scalers[target_metric] = metric_scaler

                # VERIFY storage
                if target_metric not in self.surrogate_metric_scalers:
                    raise RuntimeError(f"Failed to store scaler for {target_metric}!")
                print(f"      âœ… Scaler fitted and stored for {target_metric}")
                print(f"      Scaler stats: center={metric_scaler.center_[:3]}, scale={metric_scaler.scale_[:3]}")
                X_test_final = pd.DataFrame(
                    metric_scaler.transform(X_test_selected),
                    columns=selected_features,
                    index=X_test_surrogate.index
                )
                
                # ðŸ”º STORE THE PER-METRIC SCALER
                if not hasattr(self, 'surrogate_metric_scalers'):
                    self.surrogate_metric_scalers = {}
                self.surrogate_metric_scalers[target_metric] = metric_scaler

                # Verify scaler was stored correctly
                print(f"      âœ… Scaler stored for {target_metric}: {type(metric_scaler).__name__}")
                print(f"      Features for this scaler: {len(selected_features)}")
                
                print(f"      Selected {len(selected_features)} features from {len(X_train_surrogate.columns)}")
                print(f"      Top 5 features: {list(selected_features[:5])}")
                
            except Exception as e:
                print(f"      Ã¢Å¡ Ã¯Â¸Â  Enhanced feature selection failed: {e}, using simpler method")
                
                # Fallback to simpler feature selection
                if is_classification:
                    selector = SelectKBest(score_func=f_classif, k='all')
                else:
                    selector = SelectKBest(score_func=f_regression, k='all')
                
                X_train_selected = selector.fit_transform(X_train_surrogate, y_train_final)
                X_test_selected = selector.transform(X_test_surrogate)
                
                # Select top features
                feature_scores = selector.scores_
                feature_ranking = np.argsort(feature_scores)[::-1]
                
                n_samples = len(X_train_surrogate)
                max_features = min(50, len(feature_ranking), n_samples // 3)
                
                selected_indices = feature_ranking[:max_features]
                selected_features = X_train_surrogate.columns[selected_indices]
                
                # Ã°Å¸"Âº CRITICAL: Create scaler for fallback path too!
                from sklearn.preprocessing import RobustScaler
                metric_scaler = RobustScaler()

                # FIT first
                metric_scaler.fit(X_train_surrogate[selected_features])

                # VERIFY
                if not hasattr(metric_scaler, 'center_'):
                    raise RuntimeError(f"Fallback scaler failed to fit for {target_metric}!")

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
                
                # ðŸ”º STORE THE PER-METRIC SCALER
                if not hasattr(self, 'surrogate_metric_scalers'):
                    self.surrogate_metric_scalers = {}
                self.surrogate_metric_scalers[target_metric] = metric_scaler
                print(f"      âœ… Fallback scaler stored for {target_metric}")
                
                print(f"      Selected {len(selected_features)} features from {len(X_train_surrogate.columns)}")
                print(f"      Top 5 features: {list(selected_features[:5])}")
            
            # Cross-validation with best model selection
            cv_folds = 5  # DEFINE THIS FIRST
            fold_performances = []

            # Create stratified folds based on target quantiles
            from sklearn.model_selection import StratifiedKFold, KFold

            # Bin targets into 5 quantiles for stratification
            y_binned = pd.qcut(y_train_target, q=5, labels=False, duplicates='drop')

            try:
                kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_splits = list(kf.split(X_train_final, y_binned))
                print(f"   âœ… Using stratified {cv_folds}-fold CV (balanced across target range)")
            except:
                # Fallback if stratification fails
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_splits = list(kf.split(X_train_final))
                print(f"   âš ï¸ Using regular {cv_folds}-fold CV (stratification failed)")

            # Convert to numpy for CV split to avoid index issues
            X_train_np = X_train_final.values if hasattr(X_train_final, 'values') else X_train_final
            y_train_np = y_train_final if isinstance(y_train_final, np.ndarray) else np.array(y_train_final)

            # NOW start the CV loop
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                print(f"      Fold {fold_idx + 1}/{cv_folds}: ", end="")
                
                # Split data for this fold using numpy indexing to avoid pandas index issues
                X_fold_train = X_train_np[train_idx]
                X_fold_val = X_train_np[val_idx]
                y_fold_train = y_train_np[train_idx]
                y_fold_val = y_train_np[val_idx]
                
                # Convert back to DataFrames with reset indices
                X_fold_train_df = pd.DataFrame(X_fold_train, columns=X_train_final.columns)
                X_fold_val_df = pd.DataFrame(X_fold_val, columns=X_train_final.columns)
                
                # Train with multiple strategies and select best
                try:
                    best_model, fold_perf = self._train_with_multiple_strategies(
                        X_fold_train_df, y_fold_train, X_fold_val_df, y_fold_val,
                        target_metric, is_classification, n_classes
                    )
                    
                    if is_classification:
                        print(f"Acc: {fold_perf['test_accuracy']:.3f}")
                    else:
                        print(f"RÃ‚Â²: {fold_perf['test_r2']:.3f}")
                    
                    fold_performances.append(fold_perf)
                    
                except Exception as e:
                    print(f"Failed: {e}")
                    # Create dummy performance for failed fold
                    if is_classification:
                        dummy_perf = {'test_accuracy': 0.0, 'test_f1': 0.0}
                    else:
                        dummy_perf = {'test_r2': -1.0, 'test_mse': float('inf')}
                    fold_performances.append(dummy_perf)
            
            # Calculate average CV performance
            if is_classification:
                valid_performances = [p for p in fold_performances if p['test_accuracy'] > 0]
                if valid_performances:
                    avg_acc = np.mean([p['test_accuracy'] for p in valid_performances])
                    avg_f1 = np.mean([p['test_f1'] for p in valid_performances])
                    print(f"   Ã°Å¸â€œÅ  CV Results: Accuracy: {avg_acc:.4f} Ã‚Â± {np.std([p['test_accuracy'] for p in valid_performances]):.4f}")
                    print(f"                 F1: {avg_f1:.4f} Ã‚Â± {np.std([p['test_f1'] for p in valid_performances]):.4f}")
                    main_metric = avg_acc
                else:
                    print("   Ã¢ÂÅ’ All CV folds failed for classification")
                    continue
            else:
                valid_performances = [p for p in fold_performances if p['test_r2'] > -10]
                if valid_performances:
                    avg_r2 = np.mean([p['test_r2'] for p in valid_performances])
                    avg_mse = np.mean([p['test_mse'] for p in valid_performances])
                    print(f"   Ã°Å¸â€œÅ  CV Results: RÃ‚Â²: {avg_r2:.4f} Ã‚Â± {np.std([p['test_r2'] for p in valid_performances]):.4f}")
                    print(f"                 MSE: {avg_mse:.4f} Ã‚Â± {np.std([p['test_mse'] for p in valid_performances]):.4f}")
                    main_metric = avg_r2
                else:
                    print("   Ã¢ÂÅ’ All CV folds failed for regression")
                    continue
            
            # Train final model with the best strategy
            print(f"   Ã°Å¸Å½Â¯ Training final model with multiple strategies...")
            
            try:
                final_model, final_performance = self._train_with_multiple_strategies(
                    X_train_final, y_train_final, X_test_final, y_test_final,
                    target_metric, is_classification, n_classes
                )
                
                # Add CV results to final performance
                if is_classification:
                    final_performance.update({
                        'cv_accuracy': avg_acc,
                        'cv_f1': avg_f1,
                        'type': 'classification',
                        'n_classes': n_classes
                    })
                    print(f"   Ã¢Å“â€¦ Final Test Accuracy: {final_performance['test_accuracy']:.4f}")
                    
                    # Quality assessment with better thresholds
                    if final_performance['test_accuracy'] < 0.5:
                        print("   Ã¢Å¡ Ã¯Â¸Â  VERY LOW: Test accuracy < 0.5")
                    elif final_performance['test_accuracy'] < 0.7:
                        print("   Ã°Å¸â€œË† LOW: Test accuracy < 0.7") 
                    elif final_performance['test_accuracy'] < 0.85:
                        print("   Ã°Å¸â€œÅ  MODERATE: Test accuracy < 0.85")
                    else:
                        print("   Ã°Å¸Å½Â¯ EXCELLENT: Test accuracy Ã¢â€°Â¥ 0.85")
                        
                else:
                    # Transform predictions back if we used target transformation
                    if target_metric in self.target_transformers and self.target_transformers[target_metric] is not None:
                        # Note: final_performance already contains the back-transformed results from _train_with_multiple_strategies
                        pass
                    
                    # Add normalized MSE metrics
                    y_test_for_norm = y_test_final
                    if 'test_normalized_mse' not in final_performance:
                        # Calculate if not already done
                        test_pred = final_model.predict(X_test_final) if hasattr(final_model, 'predict') else None
                        if test_pred is not None:
                            final_performance['test_normalized_mse'] = self._calculate_normalized_mse(y_test_for_norm, test_pred)
                    
                    final_performance.update({
                        'cv_r2': avg_r2,
                        'cv_mse': avg_mse,
                        'type': 'regression'
                    })
                    
                    print(f"   Ã¢Å“â€¦ Final Test RÃ‚Â²: {final_performance['test_r2']:.4f}")
                    print(f"   Ã¢Å“â€¦ Final Test MSE: {final_performance['test_mse']:.4f}")
                    
                    # Display normalized MSE metrics
                    if 'test_normalized_mse' in final_performance:
                        norm_mse = final_performance['test_normalized_mse']
                        print(f"   Ã°Å¸â€œÅ  Normalized MSE by variance: {norm_mse['mse_normalized_by_variance']:.4f}")
                        print(f"   Ã°Å¸â€œÅ  RMSE normalized by std: {norm_mse['rmse_normalized_by_std']:.4f}")
                        print(f"   Ã°Å¸â€œÅ  RMSE normalized by range: {norm_mse['rmse_normalized_by_range']:.4f}")
                    
                    # Enhanced quality assessment
                    if final_performance['test_r2'] < 0.0:
                        print("   Ã¢Å¡ Ã¯Â¸Â  VERY POOR: Test RÃ‚Â² < 0.0 (worse than mean baseline)")
                    elif final_performance['test_r2'] < 0.3:
                        print("   Ã¢ÂÅ’ POOR: Test RÃ‚Â² < 0.3")
                    elif final_performance['test_r2'] < 0.6:
                        print("   Ã°Å¸â€œË† LOW: Test RÃ‚Â² < 0.6")
                    elif final_performance['test_r2'] < 0.8:
                        print("   Ã°Å¸â€œÅ  MODERATE: Test RÃ‚Â² < 0.8")
                    elif final_performance['test_r2'] < 0.9:
                        print("   Ã°Å¸Å½Â¯ GOOD: Test RÃ‚Â² < 0.9")
                    else:
                        print("   Ã°Å¸Å’Å¸ EXCELLENT: Test RÃ‚Â² Ã¢â€°Â¥ 0.9")
            
            except Exception as e:
                print(f"   Ã¢ÂÅ’ Final model training failed: {e}")
                continue
            
            # Store results
            surrogate_performance[target_metric] = final_performance
            self.surrogate_models[target_metric] = final_model
            
            # Store feature information for this surrogate
            if hasattr(self, 'surrogate_features'):
                self.surrogate_features[target_metric] = selected_features
            else:
                self.surrogate_features = {target_metric: selected_features}
        
        # SAVE SURROGATE MODEL CONFIGURATIONS (for RL reliability)
        config_path = self.models_dir / "surrogate_configs.json"
        surrogate_configs = {
            'target_metrics': self.target_metrics,
            'data_statistics': {},
            'model_architectures': {},
            'r2_scores': {}
        }

        for metric, model in self.surrogate_models.items():
            # Save RÃ‚Â² score
            if metric in surrogate_performance:
                surrogate_configs['r2_scores'][metric] = surrogate_performance[metric].get('test_r2', -1)
            
            # Save data statistics for MOS normalization
            if metric in self.y_train.columns:
                surrogate_configs['data_statistics'][metric] = {
                    'min': float(self.y_train[metric].min()),
                    'max': float(self.y_train[metric].max()),
                    'mean': float(self.y_train[metric].mean()),
                    'std': float(self.y_train[metric].std())
                }
            
            # Save model architecture info
            if hasattr(model, 'get_params'):
                surrogate_configs['model_architectures'][metric] = str(type(model).__name__)

        with open(config_path, 'w') as f:
            json.dump(surrogate_configs, f, indent=2)

        print(f"Ã¢Å“â€¦ Surrogate configurations saved to: {config_path}")

        # Final summary
        print(f"\nÃ°Å¸Å½â€° ENHANCED SURROGATE MODEL TRAINING COMPLETE")
        print("=" * 60)
        
        successful_targets = list(surrogate_performance.keys())
        classification_targets = [k for k, v in surrogate_performance.items() if v['type'] == 'classification']
        regression_targets = [k for k, v in surrogate_performance.items() if v['type'] == 'regression']
        
        print(f"Ã¢Å“â€¦ Successfully trained: {len(successful_targets)}/{len(self.y_train.columns)} targets")
        
        if skipped_targets:
            print(f"Ã¢ÂÂ­Ã¯Â¸Â  Skipped targets ({len(skipped_targets)}): {skipped_targets}")
        
        if classification_targets:
            print(f"\nÃ°Å¸â€œÂ Classification targets ({len(classification_targets)}):")
            for target in classification_targets:
                perf = surrogate_performance[target]
                print(f"   {target}: Test Acc = {perf['test_accuracy']:.4f}")
        
        if regression_targets:
            print(f"\nÃ°Å¸â€œÅ  Regression targets ({len(regression_targets)}):")
            for target in regression_targets:
                perf = surrogate_performance[target]
                print(f"   {target}: Test RÃ‚Â² = {perf['test_r2']:.4f}")
                
                # Show normalized MSE for regression targets
                if 'test_normalized_mse' in perf:
                    norm_mse = perf['test_normalized_mse']
                    print(f"      Normalized RMSE/std: {norm_mse['rmse_normalized_by_std']:.4f}")
        
        # Overall quality assessment
        if classification_targets:
            accuracies = [surrogate_performance[t]['test_accuracy'] for t in classification_targets]
            avg_class_acc = np.mean(accuracies)
            print(f"\nÃ°Å¸â€œË† Average Classification Accuracy: {avg_class_acc:.4f}")
            
            poor_class = [t for t in classification_targets if surrogate_performance[t]['test_accuracy'] < 0.6]
            if poor_class:
                print(f"Ã¢Å¡ Ã¯Â¸Â  Poor classification targets: {poor_class}")
        
        if regression_targets:
            r2_scores = [surrogate_performance[t]['test_r2'] for t in regression_targets]
            avg_reg_r2 = np.mean(r2_scores)
            print(f"Ã°Å¸â€œË† Average Regression RÃ‚Â²: {avg_reg_r2:.4f}")
            
            poor_reg = [t for t in regression_targets if surrogate_performance[t]['test_r2'] < 0.3]
            very_poor_reg = [t for t in regression_targets if surrogate_performance[t]['test_r2'] < 0.0]
            
            if very_poor_reg:
                print(f"Ã¢ÂÅ’ Very poor regression targets (RÃ‚Â² < 0): {very_poor_reg}")
            if poor_reg:
                print(f"Ã¢Å¡ Ã¯Â¸Â  Poor regression targets (RÃ‚Â² < 0.3): {poor_reg}")
        
        # Recommendations for improvement
        print(f"\nÃ°Å¸â€™Â¡ RECOMMENDATIONS:")
        if skipped_targets:
            print(f"   Ã°Å¸â€Â Investigate skipped targets - check data quality and metric calculations")
        
        poor_performers = []
        if regression_targets:
            poor_performers.extend([t for t in regression_targets if surrogate_performance[t]['test_r2'] < 0.3])
        if classification_targets:
            poor_performers.extend([t for t in classification_targets if surrogate_performance[t]['test_accuracy'] < 0.6])
        
        if poor_performers:
            print(f"   Ã°Å¸â€œÅ  Consider collecting more data for poor performers: {poor_performers}")
            print(f"   Ã°Å¸â€Â§ Consider feature engineering or domain-specific transformations")
            print(f"   Ã°Å¸Å½Â¯ Consider if these targets are truly predictable from available features")
        
        print("Ã¢Å“â€¦ Enhanced surrogate models ready for RL!")

        # CRITICAL: Validate RÃ‚Â² thresholds before RL training
        print("\nÃ°Å¸â€Â SURROGATE MODEL FIDELITY VALIDATION")
        print("=" * 60)
        insufficient_surrogates = []
        for target, perf in surrogate_performance.items():
            if perf['type'] == 'regression':
                r2 = perf['test_r2']
                if r2 < 0.90:
                    insufficient_surrogates.append((target, r2))
                    print(f"Ã¢Å¡ Ã¯Â¸Â  {target}: RÃ‚Â² = {r2:.4f} < 0.90 (BELOW THRESHOLD)")
                elif r2 < 0.95:
                    print(f"Ã¢Å“â€¦ {target}: RÃ‚Â² = {r2:.4f} (Acceptable, target 0.95)")
                else:
                    print(f"Ã¢Â­Â {target}: RÃ‚Â² = {r2:.4f} (Outstanding)")

        if insufficient_surrogates:
            print(f"\nÃ¢ÂÅ’ CRITICAL: {len(insufficient_surrogates)} surrogates below RÃ‚Â² = 0.90 threshold!")
            print("   RL agent will learn noise. Consider:")
            print("   1. Increase training data")
            print("   2. Better feature engineering")
            print("   3. Hyperparameter tuning")
            print("   4. Different model architectures")
            for target, r2 in insufficient_surrogates:
                print(f"      - {target}: {r2:.4f}")
        else:
            print("\nÃ¢Å“â€¦ All regression surrogates meet minimum RÃ‚Â² Ã¢â€°Â¥ 0.90 threshold!")

        # Special message for physics-based FNR
        physics_based_metrics = [m for m in surrogate_performance.keys() 
                                if surrogate_performance[m].get('method') == 'physics_based_calculation']
        if physics_based_metrics:
            print(f"\nÃ°Å¸â€Â¬ PHYSICS-BASED CALCULATIONS:")
            for metric in physics_based_metrics:
                perf = surrogate_performance[metric]
                print(f"   {metric}: RÃ‚Â² = {perf['test_r2']:.4f} (calculated from components)")
                print(f"      Ã¢â€ â€™ Not a learned model, performance depends on component accuracy")

        # REMOVE INSUFFICIENT SURROGATES FROM RL TARGETS
        if insufficient_surrogates:
            print(f"\nÃ°Å¸â€Â§ AUTOMATIC FIX: Removing {len(insufficient_surrogates)} insufficient surrogates from RL targets")
            
            insufficient_target_names = [target for target, r2 in insufficient_surrogates]
            
            # Remove from surrogate_models
            for target in insufficient_target_names:
                if target in self.surrogate_models:
                    del self.surrogate_models[target]
                    print(f"   Ã¢ÂÅ’ Removed {target} from surrogate_models")
            
            # Remove from target_metrics
            self.target_metrics = [m for m in self.target_metrics if m not in insufficient_target_names]
            
            print(f"   Ã¢Å“â€¦ Updated target_metrics: {self.target_metrics}")
            print(f"   Ã¢Å¡ Ã¯Â¸Â  RL will only use high-fidelity surrogates (RÃ‚Â² Ã¢â€°Â¥ 0.90)")

        return surrogate_performance

    def _create_composite_fnr_predictor(self, fnr_metric, component_metrics):
        """
        Create a composite FNR predictor that mirrors the dataset generation logic
        
        Dataset FNR calculation:
        1. adaptive_threshold = 5000 + 2000 * (1.0 / (snr + 1))
        2. signal_overlap = abs(final_signal - adaptive_threshold) / (signal_std + 1e-6)
        3. base_error = 0.02 / (1 + signal_overlap * 0.1)
        4. fnr = base_error * 2.0 + 0.02 * noise_factor
        5. fnr = clip(fnr, 0.01, 0.20)
        
        We'll approximate this using available predicted metrics
        """
        print(f"   Ã°Å¸â€Â§ Creating physics-based FNR calculator...")
        
        # Get actual FNR values from training data
        y_train_fnr = self.y_train[fnr_metric].values
        y_test_fnr = self.y_test[fnr_metric].values
        
        # Get component models
        snr_model = self.surrogate_models['signal_to_noise_ratio_SNR']
        ttd_model = self.surrogate_models['time_to_detection_threshold']
        dr_model = self.surrogate_models['dynamic_range_of_output']
        
        # Get component features
        snr_features = self.surrogate_features['signal_to_noise_ratio_SNR']
        ttd_features = self.surrogate_features['time_to_detection_threshold']
        dr_features = self.surrogate_features['dynamic_range_of_output']
        
        # Predict components on training data
        X_train_snr = self.X_train_scaled[snr_features]
        X_train_ttd = self.X_train_scaled[ttd_features]
        X_train_dr = self.X_train_scaled[dr_features]
        
        # Handle both PyTorch and sklearn models
        def predict_component(model, X_data):
            if hasattr(model, 'eval'):  # PyTorch
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(X_data.values)).numpy().flatten()
            else:  # sklearn
                pred = model.predict(X_data)
            return pred
        
        snr_pred_train = predict_component(snr_model, X_train_snr)
        ttd_pred_train = predict_component(ttd_model, X_train_ttd)
        dr_pred_train = predict_component(dr_model, X_train_dr)
        
        # Get background noise if available
        if 'background_noise_level' in self.X_train_scaled.columns:
            noise_train = self.X_train_scaled['background_noise_level'].values
            has_noise = True
        else:
            # Estimate noise from SNR and dynamic range
            # noise Ã¢â€°Ë† dynamic_range / SNR (approximation)
            noise_train = dr_pred_train / (snr_pred_train + 1e-6)
            noise_train = np.clip(noise_train, 0.01, 0.5)  # Reasonable noise bounds
            has_noise = False
        
        # Calculate FNR using dataset generation logic
        def calculate_fnr_from_components(snr, ttd, dr, noise):
            """Mirror the dataset generation FNR calculation"""
            # Adaptive threshold (from your code)
            adaptive_threshold = 5000 + 2000 * (1.0 / (snr + 1))
            
            # Approximate signal_overlap using available metrics
            # Higher SNR Ã¢â€ â€™ better separation Ã¢â€ â€™ lower overlap
            # Lower TTD Ã¢â€ â€™ faster response Ã¢â€ â€™ lower overlap
            # Higher DR Ã¢â€ â€™ stronger signal Ã¢â€ â€™ lower overlap
            signal_overlap = 10.0 / (snr + 1) + ttd / 200.0 - dr / 20000.0
            signal_overlap = np.clip(signal_overlap, 0.1, 10.0)
            
            # Base error (from your code)
            base_error = 0.02 / (1 + signal_overlap * 0.1)
            
            # Noise factor (from your code)
            noise_factor = noise / 0.15  # Normalize to typical noise
            
            # FNR calculation (from your code)
            fnr = base_error * 2.0 + 0.02 * noise_factor
            
            # Clip to realistic bounds (from your code)
            fnr = np.clip(fnr, 0.01, 0.20)
            
            return fnr
        
        # Calculate FNR for training data
        fnr_calc_train = calculate_fnr_from_components(
            snr_pred_train, ttd_pred_train, dr_pred_train, noise_train
        )
        
        # Evaluate on test set
        X_test_snr = self.X_test_scaled[snr_features]
        X_test_ttd = self.X_test_scaled[ttd_features]
        X_test_dr = self.X_test_scaled[dr_features]
        
        snr_pred_test = predict_component(snr_model, X_test_snr)
        ttd_pred_test = predict_component(ttd_model, X_test_ttd)
        dr_pred_test = predict_component(dr_model, X_test_dr)
        
        if has_noise:
            noise_test = self.X_test_scaled['background_noise_level'].values
        else:
            noise_test = dr_pred_test / (snr_pred_test + 1e-6)
            noise_test = np.clip(noise_test, 0.01, 0.5)
        
        fnr_calc_test = calculate_fnr_from_components(
            snr_pred_test, ttd_pred_test, dr_pred_test, noise_test
        )
        
        # Calculate performance
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        train_r2 = r2_score(y_train_fnr, fnr_calc_train)
        test_r2 = r2_score(y_test_fnr, fnr_calc_test)
        test_mse = mean_squared_error(y_test_fnr, fnr_calc_test)
        test_mae = mean_absolute_error(y_test_fnr, fnr_calc_test)
        
        print(f"      Ã°Å¸â€œÅ  Physics-based FNR Calculator Performance:")
        print(f"         Train RÃ‚Â²: {train_r2:.4f}")
        print(f"         Test RÃ‚Â²: {test_r2:.4f}")
        print(f"         Test MSE: {test_mse:.4f}")
        print(f"         Test MAE: {test_mae:.4f}")
        print(f"         Using {'actual' if has_noise else 'estimated'} background noise")
  
        # Create and store the predictor (using top-level class for pickling)
        fnr_predictor = PhysicsBasedFNRPredictor(
            snr_model, ttd_model, dr_model,
            snr_features, ttd_features, dr_features,
            has_noise  # Note: removed self.X_train_scaled parameter
        )
        
        self.surrogate_models[fnr_metric] = fnr_predictor
        
        # Store combined features
        combined_features = list(set(snr_features) | set(ttd_features) | set(dr_features))
        if has_noise and 'background_noise_level' in self.X_train_scaled.columns:
            combined_features.append('background_noise_level')
        self.surrogate_features[fnr_metric] = combined_features
        
        print(f"      Ã¢Å“â€¦ Physics-based FNR predictor created successfully!")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'type': 'regression',
            'method': 'physics_based_calculation'
        }

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
                feature_scaler=None, surrogate_metric_scalers=None):
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
                            # ✅ Validate scaler is fitted
                            if not hasattr(scaler, 'center_'):
                                raise ValueError(f"Scaler for {metric} is not fitted!")
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
                self.max_steps = 200  # âœ… Longer episodes = more learning per episode
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
                self.action_space = gym_spaces.Box(
                    low=-1.0, 
                    high=1.0, 
                    shape=(len(self.continuous_features),), 
                    dtype=np.float32
                )

                print(f"   ✅ Action space created: {self.action_space}")

                print(f"   ✅ Continuous action space: {self.action_space.shape[0]} dimensions")

                # Ã¢Å“â€¦ FIX: Calculate exact observation size based on _get_observation implementation
                # Components: normalized_modifiable + target_predictions + target_gaps + context + bounds_utilization
                obs_size = (
                    self.n_modifiable +  # normalized modifiable features
                    4 +                   # target predictions (SNR, DR, FNR, TTD)
                    4 +                   # target gaps (improvement potential)
                    6 +                   # context (step_ratio, reward_mean, reward_std, reward_trend, current_reward, current_mos)
                    self.n_modifiable +   # bounds utilization
                    self.n_modifiable +   # trajectory velocity (NEW)
                    self.n_modifiable     # action momentum (NEW)
                )

                self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(obs_size,), dtype=np.float32)
                print(f"   ✅ Observation space: {obs_size} dimensions")
                print(f"      = {self.n_modifiable} features + 4 predictions + 4 gaps + 6 context")
                print(f"      + {self.n_modifiable} bounds + {self.n_modifiable} velocity + {self.n_modifiable} momentum")

                self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(obs_size,), dtype=np.float32)
                print(f"   âœ… Observation space: {obs_size} dimensions")
                
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
                        
                        # Check if it's a PyTorch model or sklearn model
                        if hasattr(model, 'eval'):  # PyTorch model
                            dummy_tensor = torch.FloatTensor(dummy_full_state).unsqueeze(0)
                            with torch.no_grad():
                                test_output = model(dummy_tensor)
                                if torch.isnan(test_output).any():
                                    print(f"WARNING: Surrogate model {metric} produces NaN!")
                                else:
                                    print(f"Ã¢Å“â€¦ Surrogate model {metric} validated (PyTorch)")
                        else:  # sklearn model (like GradientBoostingRegressor)
                            test_output = model.predict(dummy_full_state.reshape(1, -1))
                            if np.isnan(test_output).any():
                                print(f"WARNING: Surrogate model {metric} produces NaN!")
                            else:
                                print(f"Ã¢Å“â€¦ Surrogate model {metric} validated (sklearn)")
                    except Exception as e:
                        print(f"ERROR: Surrogate model {metric} validation failed: {e}")

            def set_difficulty(self, difficulty_level):
                """Set environment difficulty (0.0 = easy, 1.0 = hard)"""
                self.difficulty = difficulty_level
                
                # Adjust reward scaling based on difficulty
                if hasattr(self, 'base_metric_weights'):
                    for metric in self.metric_weights:
                        self.metric_weights[metric] = self.base_metric_weights[metric] * (0.5 + 0.5 * difficulty_level)

            def _create_full_state(self, target_metric=None):
                """Create full state vector matching the surrogate model's expected input"""
                
                # ðŸ” DIAGNOSTIC: Check if scalers are available
                if not hasattr(self, 'surrogate_metric_scalers'):
                    print(f"   âš ï¸ WARNING: Environment {id(self)} has NO surrogate_metric_scalers attribute!")
                elif not self.surrogate_metric_scalers:
                    print(f"   âš ï¸ WARNING: Environment {id(self)} has EMPTY surrogate_metric_scalers!")
                elif target_metric and target_metric not in self.surrogate_metric_scalers:
                    print(f"   âš ï¸ WARNING: Metric {target_metric} not in scalers!")
                    print(f"       Available: {list(self.surrogate_metric_scalers.keys())}")
                if target_metric is None:
                    target_metric = list(self.surrogate_features.keys())[0]
                
                # Get features directly from surrogate_features
                if target_metric not in self.surrogate_features:
                    raise RuntimeError(f"âŒ No features found for {target_metric}")
                # Ensure we have the surrogate_features mapping
                if not hasattr(self, 'surrogate_features') or not self.surrogate_features:
                    raise RuntimeError("âŒ surrogate_features not initialized in environment!")
                feature_order = self.surrogate_features[target_metric]
                expected_features = len(feature_order)
                
                # Create UNSCALED state firstp
                full_state_unscaled = np.zeros(expected_features)
                
                for idx, feature in enumerate(feature_order):
                    if feature in self.modifiable_features:
                        modifiable_idx = self.modifiable_features.index(feature)
                    elif feature in self.fixed_values:
                        full_state_unscaled[idx] = self.fixed_values[feature]
                    else:
                        # Feature not found - this is the bug!
                        print(f"   âš ï¸ WARNING: Feature '{feature}' not found in modifiable or fixed!")
                
                # Use the ACTUAL scaler object
                if target_metric not in self.surrogate_metric_scalers:
                    # No scaler available - use unscaled features
                    # Ã¢Å“â€¦ FIX: Only print once per environment instancepppp
                    if not hasattr(self, '_scaler_warnings_shown'):
                        self._scaler_warnings_shown = set()
                        
                    if target_metric not in self._scaler_warnings_shown:
                        self._scaler_warnings_shown.add(target_metric)
                        # Only print in verbose mode (removed to reduce spam)
                    return full_state_unscaled

                metric_scaler = self.surrogate_metric_scalers[target_metric]

                if metric_scaler is None:
                    raise ValueError(
                        f"âŒ Scaler for '{target_metric}' is None! "
                        f"This should have been caught during environment initialization."
                    )
                

                # Validate scaler is fitted
                if metric_scaler is None:
                    # Don't print every time - causes spam
                    if not hasattr(self, '_scaler_warning_printed'):
                        print(f"   âš ï¸  Scaler for {target_metric} is None! Using unscaled features.")
                        self._scaler_warning_printed = set()
                    if target_metric not in self._scaler_warning_printed:
                        self._scaler_warning_printed.add(target_metric)
                    return full_state_unscaled

                if not hasattr(metric_scaler, 'center_'):
                    if not hasattr(self, '_scaler_warning_printed'):
                        print(f"   âš ï¸  Scaler for {target_metric} not fitted! Using unscaled features.")
                        self._scaler_warning_printed = set()
                    if target_metric not in self._scaler_warning_printed:
                        self._scaler_warning_printed.add(target_metric)
                    return full_state_unscaled

                # Create DataFrame with EXACT column names for THIS metric
                state_df = pd.DataFrame([full_state_unscaled], columns=feature_order)

                try:
                    full_state_scaled = metric_scaler.transform(state_df)[0]
                    return full_state_scaled
                except Exception as e:
                    print(f"   âš ï¸ Metric-specific scaling failed for {target_metric}: {e}")
                    return full_state_unscaled

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
                
                # âœ… FIX: Initialize in ORIGINAL data ranges (not scaled)
                # Get original bounds from training data
                init_strategy = np.random.choice(['random', 'center', 'near_best'])
                
                if init_strategy == 'random' or not hasattr(self, 'best_states'):
                    # Random initialization in ORIGINAL space
                    self.modifiable_state = np.zeros(self.n_modifiable, dtype=np.float32)
                    
                    for i, feature in enumerate(self.modifiable_features):
                        # Get ORIGINAL data range (not scaled bounds)
                        if feature in self.X_train.columns:
                            original_min = float(self.X_train[feature].min())
                            original_max = float(self.X_train[feature].max())
                            
                            # Initialize uniformly in this range
                            self.modifiable_state[i] = np.random.uniform(original_min, original_max)
                        else:
                            # Fallback: use scaled bounds
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
                            
                            if hasattr(model, 'eval'):
                                model.eval()
                                with torch.no_grad():
                                    pred = model(state_tensor).item()
                            else:
                                pred = model.predict(full_state.reshape(1, -1))[0]
                            
                            # Normalize predictions to [0, 1]
                            if metric == 'signal_to_noise_ratio_SNR':
                                pred_norm = np.clip(pred / 100.0, 0, 1)
                            elif metric == 'dynamic_range_of_output':
                                pred_norm = np.clip(pred / 20000.0, 0, 1)
                            elif metric == 'false_negative_rate':
                                pred_norm = np.clip(pred, 0, 1)
                            elif metric == 'time_to_detection_threshold':
                                pred_norm = np.clip(pred / 200.0, 0, 1)
                            else:
                                pred_norm = np.clip(pred, 0, 1)
                            
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
                
                # ✅ STEP 6: Trajectory velocity (NEW - shows direction of recent changes)
                trajectory_velocity = np.zeros(self.n_modifiable)
                
                if hasattr(self, 'current_episode_states') and len(self.current_episode_states) > 1:
                    # Get last 5 states (or fewer if just started)
                    recent_states = self.current_episode_states[-5:]
                    
                    if len(recent_states) >= 2:
                        # Compute velocity (average rate of change)
                        velocities = []
                        for i in range(1, len(recent_states)):
                            velocity = recent_states[i] - recent_states[i-1]
                            velocities.append(velocity)
                        
                        # Average velocity over recent steps
                        trajectory_velocity = np.mean(velocities, axis=0)
                        
                        # Normalize velocity by typical feature range
                        for i, feature in enumerate(self.modifiable_features):
                            if feature in self.X_train.columns:
                                feature_std = float(self.X_train[feature].std())
                                if feature_std > 1e-10:
                                    trajectory_velocity[i] /= feature_std
                
                # Clip velocity to reasonable range
                trajectory_velocity = np.clip(trajectory_velocity, -2.0, 2.0)
                
                # ✅ STEP 7: Action momentum (NEW - helps agent understand its recent behavior)
                action_momentum = np.zeros(self.n_modifiable)
                
                if hasattr(self, 'current_episode_actions') and len(self.current_episode_actions) > 0:
                    # Get last 3 actions
                    recent_actions = self.current_episode_actions[-3:]
                    
                    if len(recent_actions) > 0:
                        # Average recent actions (shows persistent direction)
                        # Only use continuous features
                        n_continuous = len(self.continuous_features)
                        
                        action_arrays = []
                        for act in recent_actions:
                            if isinstance(act, np.ndarray):
                                action_arrays.append(act[:n_continuous])
                            elif isinstance(act, (int, float)):
                                # Old discrete action format - skip
                                continue
                        
                        if action_arrays:
                            action_momentum[:len(action_arrays[0])] = np.mean(action_arrays, axis=0)
                
                action_momentum = np.clip(action_momentum, -1.0, 1.0)
                
                # ✅ STEP 8: Concatenate all observation components
                try:
                    obs_components = [
                        normalized_modifiable,      # (n_modifiable,)
                        target_predictions,         # (4,)
                        target_gaps,                # (4,)
                        context,                    # (6,)
                        bounds_utilization,         # (n_modifiable,)
                        trajectory_velocity,        # (n_modifiable,) - NEW
                        action_momentum            # (n_modifiable,) - NEW
                    ]
                    
                    obs = np.concatenate(obs_components).astype(np.float32)
                    
                except Exception as e:
                    print(f"⚠️ Observation construction failed: {e}")
                    # Fallback to minimal observation
                    obs = np.concatenate([
                        normalized_modifiable,
                        np.zeros(4),  # target predictions
                        np.zeros(4),  # target gaps
                        np.array([step_ratio, 0, 0, 0, 0, 0]),  # context
                        np.zeros(self.n_modifiable),  # bounds utilization
                        np.zeros(self.n_modifiable),  # trajectory velocity
                        np.zeros(self.n_modifiable)   # action momentum
                    ]).astype(np.float32)
                
                # ✅ STEP 9: Safety checks
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                obs = np.clip(obs, -3.0, 3.0)
                
                # ✅ CRITICAL: Validate observation size
                expected_size = (
                    self.n_modifiable +  # normalized_modifiable
                    4 +                   # target_predictions
                    4 +                   # target_gaps
                    6 +                   # context
                    self.n_modifiable +   # bounds_utilization
                    self.n_modifiable +   # trajectory_velocity
                    self.n_modifiable     # action_momentum
                )

                if len(obs) != expected_size:
                    print(f"⚠️ WARNING: Observation size mismatch!")
                    print(f"   Expected: {expected_size}, Got: {len(obs)}")
                    print(f"   Components: modifiable={self.n_modifiable}, other=18")
                    # Pad or trim to match
                    if len(obs) < expected_size:
                        obs = np.pad(obs, (0, expected_size - len(obs)), mode='constant')
                    else:
                        obs = obs[:expected_size]

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

                # Apply to each continuous feature
                for i, feature in enumerate(self.continuous_features):
                    if i >= len(action):
                        break
                        
                    feature_idx = self.modifiable_features.index(feature)
                    
                    # Get bounds from modifiable_bounds (already in original scale)
                    bounds = list(self.modifiable_bounds.values())[feature_idx]
                    data_range = bounds[1] - bounds[0]
                    
                    # ✅ CRITICAL FIX: Larger step size for better exploration
                    # PPO needs to see meaningful changes to learn
                    delta = action[i] * 0.15 * data_range  # 15% max change (was 3%)
                    new_val = self.modifiable_state[feature_idx] + delta
                        
                    # ✅ FIXED: Get bounds from stored dictionary
                    if feature in self.X_train.columns:
                        original_min = float(self.X_train[feature].min())
                        original_max = float(self.X_train[feature].max())
                    else:
                        # Fallback to modifiable_bounds
                        bounds_list = list(self.modifiable_bounds.values())
                        original_min, original_max = bounds_list[feature_idx]

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
                
                # Log episode summary when episode ends
                if done:
                    self._log_episode_summary()
                
                # ✅ FIXED: Return Gymnasium-style (5 values)
                terminated = done
                truncated = False  # We don't use truncation
                return self._get_observation(), reward, terminated, truncated, info
         
            def _calculate_reward_with_predictions(self):
                """Calculate reward using MOS with comprehensive error handling"""
                
                # ðŸ”º DEBUG: Check if data_min_max still exists
                if not hasattr(self, 'data_min_max'):
                    print("âŒ CRITICAL: self.data_min_max attribute was DELETED!")
                    print(f"   Available attributes: {[a for a in dir(self) if not a.startswith('_')][:10]}")
                elif not self.data_min_max:
                    print(f"âŒ CRITICAL: self.data_min_max is EMPTY! Type: {type(self.data_min_max)}")
                    print(f"   This happened AFTER __init__ where it had {len(self.data_min_max)} metrics")
                
                try:
                    # Initialize predictions dictionary
                    predictions = {}
                    
                    # Step 1: Get predictions for all available metrics
                    # Predict in dependency order (FNR depends on others)
                    prediction_order = [m for m in self.target_metrics if 'false_negative' not in m.lower()]
                    prediction_order.extend([m for m in self.target_metrics if 'false_negative' in m.lower()])

                    for metric in prediction_order:
                        if metric not in self.surrogate_models:
                            continue
                            
                        try:

                            model = self.surrogate_models[metric]
                            
                            # Special handling for physics-based FNR predictor
                            if 'false_negative' in metric.lower() and hasattr(model, 'snr_model'):
                                # Use all scaled features for physics-based prediction
                                full_state = self._create_full_state_for_all_features()
                            else:
                                full_state = self._create_full_state(target_metric=metric)
                            
                            if hasattr(self, 'feature_scaler'):
                                full_state_scaled = self.feature_scaler.transform(full_state.reshape(1, -1))[0]
                                state_tensor = torch.FloatTensor(full_state_scaled).unsqueeze(0)
                            else:
                                state_tensor = torch.FloatTensor(full_state).unsqueeze(0)
                            
                            if hasattr(model, 'eval'):
                                model.eval()
                                with torch.no_grad():
                                    pred = model(state_tensor)
                                    predictions[metric] = pred.item()
                            else:
                                state_array = state_tensor.squeeze().numpy()
                                pred = model.predict(state_array.reshape(1, -1))
                                predictions[metric] = pred[0]

                            # âœ… FIX: Reduce prediction noise with multiple samples
                            # Make 3 predictions with slight noise and average
                            predictions_samples = []
                            
                            for sample_idx in range(3):
                                if sample_idx == 0:
                                    # First sample: no noise
                                    state_sample = full_state
                                else:
                                    # Add tiny noise for robustness
                                    noise = np.random.normal(0, 0.01, size=full_state.shape)
                                    state_sample = full_state + noise
                                
                                state_tensor = torch.FloatTensor(state_sample).unsqueeze(0)
                                
                                if hasattr(model, 'eval'):
                                    model.eval()
                                    with torch.no_grad():
                                        pred = model(state_tensor).item()
                                else:
                                    pred = model.predict(state_sample.reshape(1, -1))[0]
                                
                                predictions_samples.append(pred)
                            
                            # Average the predictions to reduce noise
                            predictions[metric] = np.mean(predictions_samples)
                                
                        except Exception as e:
                            print(f"Ã¢Å¡ Ã¯Â¸Â  Failed to predict {metric}: {e}")
                            predictions[metric] = 0.0
                    
                    # Step 2: Calculate MOS with NORMALIZED values
                    # Default ranges (used if data_min_max not available)
                    default_ranges = {
                        'signal_to_noise_ratio_SNR': (0, 100),
                        'dynamic_range_of_output': (0, 20000),
                        'false_negative_rate': (0, 1),
                        'time_to_detection_threshold': (0, 200)
                    }
                    
                    # Use provided data_min_max or fall back to defaults
                    ranges = getattr(self, 'data_min_max', default_ranges)
                    
                    def safe_normalize(value, metric_name):
                        """Normalize using ACTUAL data ranges from self.data_min_max"""
                        
                        # ✅ Use stored data ranges (not hardcoded guesses)
                        if metric_name not in self.data_min_max:
                            print(f"⚠️ {metric_name} not in data_min_max, using [0,1]")
                            return np.clip(value, 0.0, 1.0)
                        
                        min_val, max_val = self.data_min_max[metric_name]
                        
                        if max_val - min_val < 1e-10:
                            return 0.5
                        
                        normalized = (value - min_val) / (max_val - min_val)
                        return np.clip(normalized, 0.0, 1.0)

                    # Calculate MOS components with CORRECT normalization
                    mos = 0.0

                    # SNR (maximize, w=0.45)
                    if 'signal_to_noise_ratio_SNR' in predictions:
                        snr_norm = safe_normalize(predictions['signal_to_noise_ratio_SNR'], 
                                                'signal_to_noise_ratio_SNR')
                        mos += 0.45 * snr_norm

                    # Dynamic Range (maximize, w=0.25)
                    if 'dynamic_range_of_output' in predictions:
                        dr_norm = safe_normalize(predictions['dynamic_range_of_output'],
                                                'dynamic_range_of_output')
                        mos += 0.25 * dr_norm

                    # False Negative Rate (minimize, w=0.20)
                    if 'false_negative_rate' in predictions:
                        fnr_norm = safe_normalize(predictions['false_negative_rate'],
                                                'false_negative_rate')
                        mos += 0.20 * (1.0 - fnr_norm)  # Minimize, so invert

                    # Time to Detection (minimize, w=0.10)
                    if 'time_to_detection_threshold' in predictions:
                        ttd_norm = safe_normalize(predictions['time_to_detection_threshold'],
                                                'time_to_detection_threshold')
                        mos += 0.10 * (1.0 - ttd_norm)  # Minimize, so invert

                    # ðŸ”º CRITICAL: MOS is already in [0, 1], don't scale further
                    # Just ensure it's clipped
                    mos = np.clip(mos, 0.0, 1.0)
                    
                    # Constraint penalty (SOFTENED to prevent harsh jumps)
                    constraint_penalty = 0.0
                    for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                        violation = 0.0
                        if self.modifiable_state[i] < bounds[0]:
                            violation = (bounds[0] - self.modifiable_state[i]) / (bounds[1] - bounds[0])
                        elif self.modifiable_state[i] > bounds[1]:
                            violation = (self.modifiable_state[i] - bounds[1]) / (bounds[1] - bounds[0])
                        
                        # Smooth penalty (not binary)
                        if violation > 0:
                            constraint_penalty += 0.1 * np.tanh(violation)  # Soft penalty
                    
                    constraint_penalty = np.clip(constraint_penalty, 0.0, 0.5)  # Max 50% penalty
                    
                    # MULTI-COMPONENT REWARD for better learning signal
                    reward_components = {}

                    # Component 1: Raw MOS (main objective)
                    reward_components['mos'] = mos
                    mos_reward = mos * 2.0  # Weight MOS highly

                    # Component 2: Individual target rewards (help agent understand)
                    target_rewards = 0.0
                    for metric in ['signal_to_noise_ratio_SNR', 'dynamic_range_of_output',
                                'false_negative_rate', 'time_to_detection_threshold']:
                        if metric in predictions:
                            pred = predictions[metric]
                            
                            # Calculate target-specific reward
                            if metric == 'signal_to_noise_ratio_SNR':
                                # Want high SNR (0-100)
                                target_reward = (pred / 100.0) * 0.45
                            elif metric == 'dynamic_range_of_output':
                                # Want high DR (0-20000)
                                target_reward = (pred / 20000.0) * 0.25
                            elif metric == 'false_negative_rate':
                                # Want low FNR (0-1)
                                target_reward = (1.0 - pred) * 0.20
                            elif metric == 'time_to_detection_threshold':
                                # Want low TTD (0-200)
                                target_reward = (1.0 - pred / 200.0) * 0.10
                            else:
                                target_reward = 0.0
                            
                            target_rewards += target_reward
                            reward_components[metric] = target_reward

                    reward_components['targets'] = target_rewards

                    # Component 3: Improvement reward (reward progress)
                    improvement_reward = 0.0
                    if len(self.reward_history) > 0:
                        previous_mos = self.reward_history[-1] if len(self.reward_history) > 0 else 0.0
                        improvement = mos - previous_mos
                        improvement_reward = 0.5 * np.tanh(improvement * 10)  # Scaled improvement bonus
                    reward_components['improvement'] = improvement_reward

                    # Component 4: Constraint penalty (soft, not harsh)
                    soft_constraint_penalty = 0.0
                    for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                        if self.modifiable_state[i] < bounds[0]:
                            violation = (bounds[0] - self.modifiable_state[i]) / (bounds[1] - bounds[0])
                            soft_constraint_penalty += 0.05 * violation
                        elif self.modifiable_state[i] > bounds[1]:
                            violation = (self.modifiable_state[i] - bounds[1]) / (bounds[1] - bounds[0])
                            soft_constraint_penalty += 0.05 * violation

                    reward_components['constraint'] = -soft_constraint_penalty

                    # Component 5: Exploration bonus
                    exploration_bonus = 0.0
                    if hasattr(self, 'previous_state') and self.previous_state is not None:
                        state_change = np.linalg.norm(self.modifiable_state - self.previous_state)
                        # Reward moderate changes (not too small, not too large)
                        if 0.05 < state_change < 0.30:
                            exploration_bonus = 0.1
                    reward_components['exploration'] = exploration_bonus

                    # Base reward from MOS (already [0, 1])
                    base_reward = mos

                    # ✅ CORRECT SCALING: Amplify differences WITHOUT going negative
                    # Map [0,1] -> [0, 2] to make improvements more noticeable
                    amplified_mos = base_reward * 2.0

                    # Add strong improvement bonus
                    improvement_bonus = 0.0
                    if len(self.reward_history) > 0:
                        previous_mos = self.reward_history[-1]
                        improvement = mos - previous_mos
                        # Large bonus for improvements (20x multiplier)
                        improvement_bonus = np.clip(improvement * 20.0, -0.5, 0.5)

                    # Exploration bonus (encourage trying new states)
                    exploration_bonus = 0.0
                    if hasattr(self, 'previous_state') and self.previous_state is not None:
                        state_change = np.linalg.norm(self.modifiable_state - self.previous_state)
                        if 0.05 < state_change < 0.30:
                            exploration_bonus = 0.2

                    # Soft constraint penalty
                    soft_constraint_penalty = 0.0
                    for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                        if self.modifiable_state[i] < bounds[0]:
                            violation = (bounds[0] - self.modifiable_state[i]) / (bounds[1] - bounds[0])
                            soft_constraint_penalty += 0.1 * violation
                        elif self.modifiable_state[i] > bounds[1]:
                            violation = (self.modifiable_state[i] - bounds[1]) / (bounds[1] - bounds[0])
                            soft_constraint_penalty += 0.1 * violation

                    # Final reward (ALWAYS POSITIVE BASE)
                    final_reward = (
                        amplified_mos +              # [0, 2] range
                        improvement_bonus +          # [-0.5, +0.5]
                        exploration_bonus -          # [0, 0.2]
                        soft_constraint_penalty      # [0, 0.5]
                    )

                    # Clip to safe range (but allow negatives for penalties)
                    final_reward = np.clip(final_reward, -1.0, 3.0)

                    # Store for next step
                    self.previous_state = self.modifiable_state.copy()
                    predictions['reward_components'] = reward_components  # Log for analysis

                    # Add small exploration bonus to encourage trying new states
                    exploration_bonus = 0.0
                    if hasattr(self, 'previous_state') and self.previous_state is not None:
                        state_change = np.linalg.norm(self.modifiable_state - self.previous_state)
                        exploration_bonus = 0.05 * np.tanh(state_change)  # Reward for exploration
                        final_reward += exploration_bonus

                    # Store for next step
                    self.previous_state = self.modifiable_state.copy()
                    
                    predictions['mos_score'] = float(mos)
                    predictions['constraint_penalty'] = float(constraint_penalty)
                    
                    return float(final_reward), predictions
                    
                except Exception as e:
                    print(f"Ã¢ÂÅ’ Error in reward calculation: {e}")
                    import traceback
                    traceback.print_exc()
                    return 0.01, {}
    
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
                # Use actual training data range (with 5% margin for safety)
                actual_min = float(self.y_train[metric].min())
                actual_max = float(self.y_train[metric].max())
                range_margin = (actual_max - actual_min) * 0.05
                
                min_val = actual_min - range_margin
                max_val = actual_max + range_margin
                
                # Ensure valid range
                if max_val <= min_val:
                    print(f"   âš ï¸  {metric} has invalid range, using safe defaults")
                    if 'snr' in metric.lower():
                        min_val, max_val = 0, 100
                    elif 'dynamic_range' in metric.lower():
                        min_val, max_val = 0, 20000
                    elif 'false_negative' in metric.lower():
                        min_val, max_val = 0, 1
                    elif 'time_to_detection' in metric.lower():
                        min_val, max_val = 0, 200
                
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
            elif not hasattr(self.surrogate_metric_scalers[metric], 'center_'):
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
                print(f"      {metric}: {type(scaler).__name__}, {len(scaler.center_)} features")

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
                                            from sklearn.preprocessing import RobustScaler
                                            new_scaler = RobustScaler()
                                            new_scaler.fit(self.X_train[features])
                                            _local_scalers[metric] = new_scaler
                                            print(f"   ✅ Rebuilt scaler for {metric}")
                    
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
                        surrogate_metric_scalers=_local_scalers
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
                surrogate_metric_scalers=_local_scalers
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
            
            # ✅ OPTIMIZED HYPERPARAMETERS
            "learning_rate": 3e-4,       # Constant (no decay initially)
            "n_steps": 8192,             # 2x more steps per update
            "batch_size": 512,           # 2x larger batches
            "n_epochs": 30,              # More epochs (was 20)
            "gamma": 0.995,              # Slightly higher discount
            "gae_lambda": 0.98,          # Better advantage estimation
            "clip_range": 0.25,          # Slightly more aggressive clipping
            
            # ✅ CRITICAL: High entropy for exploration
            "ent_coef": 0.01,            # Entropy bonus
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            
            "normalize_advantage": True,
            "policy_kwargs": {
                "net_arch": dict(
                    pi=[512, 512, 256],  # 3-layer policy network
                    vf=[512, 512, 256]   # 3-layer value network
                ),
                "activation_fn": torch.nn.ReLU,
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
                    
                    # Evaluate every 3 chunks
                    if (i + 1) % 3 == 0:
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
                if hasattr(model, 'state_dict'):  # PyTorch
                    # Determine input dimension from model or features
                    if hasattr(model, 'layers') and len(model.layers) > 0:
                        input_dim = model.layers[0].in_features
                    elif metric in self.surrogate_features:
                        input_dim = len(self.surrogate_features[metric])
                    else:
                        input_dim = len(self.X_train.columns)
                    
                    surrogate_cache['models'][metric] = {
                        'type': 'pytorch',
                        'state_dict': model.state_dict(),
                        'input_dim': input_dim
                    }
                    print(f"   ðŸ“¦ Cached PyTorch model {metric} (input_dim={input_dim})")
                elif isinstance(model, PhysicsBasedFNRPredictor):
                    surrogate_cache['models'][metric] = {
                        'type': 'physics_based',
                        'snr_features': model.snr_features,
                        'ttd_features': model.ttd_features,
                        'dr_features': model.dr_features,
                        'has_noise_feature': model.has_noise_feature
                    }
                else:  # sklearn
                    surrogate_cache['models'][metric] = {
                        'type': 'sklearn',
                        'model': model
                    }
            
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
                    
                    if not hasattr(scaler, 'center_'):
                        print(f"   ❌ Scaler for {metric} not fitted!")
                        continue
                    
                    # ✅ Store the actual fitted scaler
                    scalers_cache['surrogate_metric_scalers'][metric] = scaler
                    print(f"   ✅ Cached scaler for {metric} (fitted on {len(scaler.center_)} features)")
                
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
                from sklearn.model_selection import train_test_split
                X = self.processed_data['X']
                y = self.processed_data['y']
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                print("   âœ… Training data reconstructed for scaler rebuilding")

                # AUGMENT training data (NOT test data)
                #print("🔬 Augmenting training data...")
                #self.augment_training_data(augmentation_factor=3)
        
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
                # Check if it's already a fitted scaler object
                if scaler is not None and hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
                    # ✅ It's a fitted scaler - use it directly!
                    self.surrogate_metric_scalers[metric] = scaler
                    print(f"   ✅ Loaded fitted scaler for {metric} ({len(scaler.center_)} features)")
                elif isinstance(scaler, dict) and 'center_' in scaler:
                    # ✅ It's scaler parameters - reconstruct
                    from sklearn.preprocessing import RobustScaler
                    reconstructed = RobustScaler()
                    reconstructed.center_ = np.array(scaler['center_'])
                    reconstructed.scale_ = np.array(scaler['scale_'])
                    self.surrogate_metric_scalers[metric] = reconstructed
                    print(f"   ✅ Reconstructed scaler for {metric}")
                else:
                    print(f"   ❌ Invalid scaler format for {metric}: {type(scaler)}")

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
                if not hasattr(scaler, 'center_'):
                    print(f"   ❌ Scaler for {metric} is not fitted!")
                    return False
                
                print(f"   ✅ Verified {metric}: {len(scaler.center_)} features, center range [{scaler.center_.min():.3f}, {scaler.center_.max():.3f}]")

            print(f"   🎉 All {len(self.surrogate_metric_scalers)} scalers verified!")

            # ✅ CRITICAL: Verify all target metrics have scalers (NO EMERGENCY REBUILD)
            missing_scalers = []
            for metric in self.target_metrics:
                if metric not in self.surrogate_metric_scalers:
                    missing_scalers.append(metric)
                elif self.surrogate_metric_scalers[metric] is None:
                    missing_scalers.append(metric)
                elif not hasattr(self.surrogate_metric_scalers[metric], 'center_'):
                    missing_scalers.append(metric)

            if missing_scalers:
                print(f"   ❌ CRITICAL: Missing scalers for: {missing_scalers}")
                print(f"   ❌ Cannot use cached models - scalers are corrupted!")
                print(f"   ❌ You must retrain models to regenerate scalers.")
                return False  # Force retraining

            print(f"   ✅ All target metrics have valid scalers")
            
            # 3. Reconstruct surrogate models (now features and scalers are available)
            self.surrogate_models = {}
            for metric, model_data in surrogate_cache['models'].items():
                if model_data['type'] == 'pytorch':
                    # Reconstruct PyTorch model
                    try:
                        input_dim = model_data['input_dim']
                        
                        # ðŸ”§ FIX: Use ENHANCED model architecture (matches saved models)
                        model = self._create_enhanced_traditional_model(input_dim, 'regression')
                        
                        # Load with strict=False to see what's missing/extra
                        try:
                            model.load_state_dict(model_data['state_dict'], strict=True)
                            print(f"   âœ… PyTorch model {metric} reconstructed (dim={input_dim})")
                        except RuntimeError as e:
                            print(f"   âš ï¸ State dict mismatch for {metric}, trying non-strict load...")
                            model.load_state_dict(model_data['state_dict'], strict=False)
                            print(f"   âš ï¸ PyTorch model {metric} loaded with warnings")
                        
                        model.eval()
                        self.surrogate_models[metric] = model
                        model.eval()
                        self.surrogate_models[metric] = model
                        print(f"   âœ… PyTorch model {metric} reconstructed (dim={input_dim})")
                    except Exception as e:
                        print(f"   âŒ Failed to reconstruct {metric}: {e}")
                elif model_data['type'] == 'physics_based':
                    # Reconstruct PhysicsBasedFNRPredictor requires component models
                    print(f"   âš ï¸  Physics-based model {metric} needs component models (skipping)")
                else:  # sklearn
                    self.surrogate_models[metric] = model_data['model']
                    print(f"   âœ… sklearn model {metric} loaded")
            
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
    
    
    def train_rl_agents_with_curriculum(self, total_timesteps: int = 2000000):
        """
        Train RL agents with automatic curriculum learning
        Start with easier goals, gradually increase difficulty
        """
        print(f"Ã°Å¸Å½â€œ Training RL agents with CURRICULUM LEARNING...")
        
        # Stage 1: Focus on SNR only (25% of training)
        stage1_timesteps = int(total_timesteps * 0.25)
        print(f"\nÃ°Å¸â€œÅ¡ STAGE 1: Focus on SNR (easiest metric)")
        # Set reward weights (handle both single and vectorized environments)
        if hasattr(self.rl_env, 'envs'):
            # Vectorized environment
            for env in self.rl_env.envs:
                env.set_reward_weights({
                    'signal_to_noise_ratio_SNR': 0.90,
                    'dynamic_range_of_output': 0.05,
                    'false_negative_rate': 0.03,
                    'time_to_detection_threshold': 0.02
                })
        else:
            # Single environment
            self.rl_env.set_reward_weights({
                'signal_to_noise_ratio_SNR': 0.90,
                'dynamic_range_of_output': 0.05,
                'false_negative_rate': 0.03,
                'time_to_detection_threshold': 0.02
            })
        
        ppo_agent = PPO(**self.get_stable_training_configs()[1])
        ppo_agent.learn(total_timesteps=stage1_timesteps)
        
        # Stage 2: Add Dynamic Range (25% of training)
        stage2_timesteps = int(total_timesteps * 0.25)
        print(f"\nÃ°Å¸â€œÅ¡ STAGE 2: Add Dynamic Range")
        # Set reward weights (handle both single and vectorized environments)
        if hasattr(self.rl_env, 'envs'):
            # Vectorized environment
            for env in self.rl_env.envs:
                env.set_reward_weights({
                    'signal_to_noise_ratio_SNR': 0.90,
                    'dynamic_range_of_output': 0.05,
                    'false_negative_rate': 0.03,
                    'time_to_detection_threshold': 0.02
                })
        else:
            # Single environment
            self.rl_env.set_reward_weights({
                'signal_to_noise_ratio_SNR': 0.90,
                'dynamic_range_of_output': 0.05,
                'false_negative_rate': 0.03,
                'time_to_detection_threshold': 0.02
            })
        
        ppo_agent.learn(total_timesteps=stage2_timesteps)
        
        # Stage 3: Add FNR (25% of training)
        stage3_timesteps = int(total_timesteps * 0.25)
        print(f"\nÃ°Å¸â€œÅ¡ STAGE 3: Add False Negative Rate")
        # Set reward weights (handle both single and vectorized environments)
        if hasattr(self.rl_env, 'envs'):
            # Vectorized environment
            for env in self.rl_env.envs:
                env.set_reward_weights({
                    'signal_to_noise_ratio_SNR': 0.90,
                    'dynamic_range_of_output': 0.05,
                    'false_negative_rate': 0.03,
                    'time_to_detection_threshold': 0.02
                })
        else:
            # Single environment
            self.rl_env.set_reward_weights({
                'signal_to_noise_ratio_SNR': 0.90,
                'dynamic_range_of_output': 0.05,
                'false_negative_rate': 0.03,
                'time_to_detection_threshold': 0.02
            })
        
        ppo_agent.learn(total_timesteps=stage3_timesteps)
        
        # Stage 4: Full multi-objective (25% of training)
        stage4_timesteps = total_timesteps - stage1_timesteps - stage2_timesteps - stage3_timesteps
        print(f"\nÃ°Å¸â€œÅ¡ STAGE 4: Full Multi-Objective Optimization")
        # Set reward weights (handle both single and vectorized environments)
        if hasattr(self.rl_env, 'envs'):
            # Vectorized environment
            for env in self.rl_env.envs:
                env.set_reward_weights({
                    'signal_to_noise_ratio_SNR': 0.90,
                    'dynamic_range_of_output': 0.05,
                    'false_negative_rate': 0.03,
                    'time_to_detection_threshold': 0.02
                })
        else:
            # Single environment
            self.rl_env.set_reward_weights({
                'signal_to_noise_ratio_SNR': 0.90,
                'dynamic_range_of_output': 0.05,
                'false_negative_rate': 0.03,
                'time_to_detection_threshold': 0.02
            })
        
        ppo_agent.learn(total_timesteps=stage4_timesteps)
        
        # Final evaluation
        print(f"\nðŸŽ¯ Curriculum training complete. Running final evaluation...")

        ppo_rewards = []
        n_eval_episodes = 20

        # Check if vectorized
        is_vec_env = hasattr(self.rl_env, 'num_envs')
        n_parallel = self.rl_env.num_envs if is_vec_env else 1

        episodes_completed = 0

        while episodes_completed < n_eval_episodes:
            obs = self.rl_env.reset()
            done_flags = np.zeros(n_parallel, dtype=bool)
            episode_rewards = np.zeros(n_parallel)
            step_counts = np.zeros(n_parallel)
            
            for step in range(1000):  # Max steps per episode
                action, _ = ppo_agent.predict(obs, deterministic=True)
                obs, reward, done, _ = self.rl_env.step(action)
                
                # Convert to arrays for consistent handling
                reward_array = np.atleast_1d(reward)
                done_array = np.atleast_1d(done).astype(bool)
                
                # Accumulate rewards only for non-done envs
                episode_rewards[~done_flags] += reward_array[~done_flags]
                step_counts[~done_flags] += 1
                
                # Update done flags
                done_flags = np.logical_or(done_flags, done_array)
                
                # Break if all environments are done
                if np.all(done_flags):
                    break
            
            # Average per-step reward instead of cumulative
            avg_episode_rewards = episode_rewards / np.maximum(step_counts, 1)
            
            # Record completed episodes
            if is_vec_env:
                for i in range(n_parallel):
                    if episodes_completed < n_eval_episodes:
                        ppo_rewards.append(avg_episode_rewards[i])
                        episodes_completed += 1
            else:
                ppo_rewards.append(avg_episode_rewards[0])
                episodes_completed += 1
            
            if episodes_completed % 5 == 0:
                print(f"   Evaluated {episodes_completed}/{n_eval_episodes} episodes...")

        # Return performance dict
        return {
            'PPO': {
                'mean_reward': np.mean(ppo_rewards),
                'std_reward': np.std(ppo_rewards),
                'max_reward': np.max(ppo_rewards),
                'rewards': ppo_rewards
            }
        }

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
            preprocessing_info = self.preprocess_data(apply_pca=apply_pca)

            # ðŸ”º NEW: Augment data before training
            self.augment_training_data(augmentation_factor=2)  # Reduce to 2x only
            # REMOVE noise augmentation entirely - it's adding too much noise

            self.plot_target_metrics_distributions(self.plots_dir / 'target_metric_plots')
            self.plot_target_metrics_correlation_matrix(self.plots_dir / 'target_metric_plots')
            self.plot_target_metrics_summary(self.plots_dir / 'target_metric_plots')

            # Check if we should use cached models
            models_loaded_from_cache = False
            if use_cached_models:
                models_loaded_from_cache = self._load_ml_models_cache()

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

        rl_timesteps = 5000000  # 5M timesteps for proper convergence
        # Rationale: 500k was insufficient based on analysis showing premature convergence         ------------------------------------------------ NEED TO CHANGE
        
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