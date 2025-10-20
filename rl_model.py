"""
Synthetic Biology Biosensor Design: Complete ML/RL Pipeline
==========================================================

This pipeline implements a comprehensive computational framework for optimizing
gene circuits for robust biomarker detection using machine learning and 
reinforcement learning techniques.

Stages:
1. Data Preprocessing & Feature Selection
2. Supervised Learning Models
3. Surrogate Modeling
4. Reinforcement Learning (DQN/PPO)
5. Graph-Based Analysis & Visualization

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

# RL Libraries
import gym
from gym import spaces
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
        
        print(f"🔍 Logging to: {self.log_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        if self.log_handle:
            self.log_handle.close()
        
        print(f"✅ Log saved to: {self.log_file}")

# Suppress warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

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
        
        # 🎯 FOCUSED TARGET METRICS - Only these will be used for training
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
        
        print(f"🚀 Biosensor Pipeline initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Target metrics: {len(self.target_metrics)} focused metrics")
    
    def load_data(self, biomarker: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset (full or biomarker-specific)
        
        Args:
            biomarker: If specified, load biomarker-specific dataset
            
        Returns:
            Loaded DataFrame
        """
        if biomarker:
            biomarker_path = Path(self.data_path).parent / f"biomarker_{biomarker}_dataset.csv"
            if biomarker_path.exists():
                print(f"📊 Loading biomarker-specific dataset: {biomarker}")
                self.raw_data = pd.read_csv(biomarker_path)
            else:
                print(f"⚠️ Biomarker dataset not found: {biomarker_path}")
                print(f"📊 Loading main dataset instead")
                self.raw_data = pd.read_csv(self.data_path)
        else:
            print(f"📊 Loading main dataset: {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
        
        print(f"✅ Dataset loaded: {self.raw_data.shape} (rows, columns)")
        
        # Check which target metrics are actually available
        available_targets = [metric for metric in self.target_metrics if metric in self.raw_data.columns]
        missing_targets = [metric for metric in self.target_metrics if metric not in self.raw_data.columns]
        
        if missing_targets:
            print(f"⚠️  Missing target metrics: {missing_targets}")
        print(f"✅ Available target metrics: {available_targets}")
        
        # Update target metrics to only include available ones
        self.target_metrics = available_targets
        
        # Calculate multi-objective score if not present
        if 'multi_objective_score' not in self.raw_data.columns:
            required_metrics = ['signal_to_noise_ratio_SNR', 'false_negative_rate', 
                              'time_to_detection_threshold', 'dynamic_range_of_output']
            
            if all(m in self.raw_data.columns for m in required_metrics):
                self.raw_data['multi_objective_score'] = self._calculate_multi_objective_score(self.raw_data)
                print("âœ… Multi-objective score calculated and added to dataset")
            else:
                missing = [m for m in required_metrics if m not in self.raw_data.columns]
                print(f"âš ï¸  Cannot calculate multi-objective score. Missing: {missing}")
        
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

        print("🔄 Starting advanced data preprocessing with proper variable classification...")

        # 🟩 MODIFIABLE INPUTS (RL Agent can change these)
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

        # 🟦 FIXED INPUTS (Environmental/Simulation - not modifiable by RL agent)
        fixed_features = [
            'sclerostin_concentration',
            'dkk1_concentration',
            'local_pH',
            'mechanical_loading'
        ]

        # 🟥 OUTPUT METRICS (Performance - never inputs, always targets)
        output_metrics = self.target_metrics  # Use focused target metrics directly  

        # Handle missing values
        print("🧹 Handling missing values...")
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
        print(f"   Missing values: {missing_before} → {missing_after}")

        # 🎯 PROPER VARIABLE SEPARATION
        print("🎯 Separating variables by classification...")

        # Get available columns for each category
        available_modifiable = [col for col in modifiable_features if col in self.raw_data.columns]
        available_fixed = [col for col in fixed_features if col in self.raw_data.columns]
        available_outputs = [col for col in output_metrics if col in self.raw_data.columns]

        # Store the classification for later use
        self.modifiable_features = available_modifiable
        self.fixed_features = available_fixed
        self.output_metrics = available_outputs

        print(f"📋 Modifiable features (RL actions): {len(available_modifiable)}")
        print(f"📋 Fixed features (environment): {len(available_fixed)}")
        print(f"📋 Output metrics (targets): {len(available_outputs)}")

        if not available_outputs:
            raise ValueError("No output metrics available in the dataset!")

        # For ML models: Use ALL features (modifiable + fixed) as inputs
        X_features = available_modifiable + available_fixed
        X = self.raw_data[X_features].copy()
        y = self.raw_data[available_outputs].copy()

        # Update target metrics to only available ones
        self.target_metrics = available_outputs

        # Encode categorical variables
        print("🔤 Encoding categorical variables...")
        categorical_features = X.select_dtypes(include=['object']).columns

        if len(categorical_features) > 0:
            X_categorical = pd.get_dummies(X[categorical_features], prefix=categorical_features)
            X_numerical = X.select_dtypes(include=[np.number])
            X = pd.concat([X_numerical, X_categorical], axis=1)
            print(f"   Categorical features encoded: {len(categorical_features)} → {X_categorical.shape[1]} columns")

        # Scale numerical features
        print("📏 Scaling numerical features...")
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
            print(f"   Feature selection: {X_scaled.shape[1]} → {X_selected.shape[1]} features")
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
        print("✂️ Splitting data...")
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

        print("✅ Advanced preprocessing with proper classification completed!")
        print(f"   Final shape: {X_final.shape}")
        print(f"   Train/Test: {self.X_train.shape[0]}/{self.X_test.shape[0]}")
        print(f"   Features: {len(X_features)} → {X_final.shape[1]} ({preprocessing_info['feature_reduction_ratio']:.2f})")
        print(f"   Modifiable: {len(available_modifiable)}, Fixed: {len(available_fixed)}")
        print(f"   Target metrics: {len(available_outputs)}")

        return preprocessing_info

    def _perform_feature_selection(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """
        Feature selection for Sclerostin biosensor - Use ALL 13 features (no selection needed)
        """
        print("Feature selection for Sclerostin biosensor...")
        
        # For Sclerostin dataset, we explicitly use all 13 features
        # No feature selection needed as features are already minimal and well-defined
        
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
        
        # Use all features that are available in X
        available_features = [f for f in modifiable_features + fixed_features if f in X.columns]
        
        print(f"   Using all {len(available_features)} available features (no selection)")
        print(f"   - Modifiable: {len([f for f in modifiable_features if f in X.columns])}")
        print(f"   - Fixed: {len([f for f in fixed_features if f in X.columns])}")
        
        return {
            'selected_features': {'all': available_features},
            'final_features': available_features,
            'scores': {},
            'reduction_ratio': 1.0  # No reduction
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
        print(f"🔍 Applying dimensionality reduction: {method}")
        
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
            
            print(f"   PCA: {X.shape[1]} → {n_components} components")
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
        
        print(f"📊 Creating distribution plots for {len(self.output_metrics)} target metrics...")
        print(f"💾 Saving plots to: {save_folder}/")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots for each target metric
        for i, metric in enumerate(self.output_metrics):
            if metric not in y_data.columns:
                print(f"⚠️  Skipping {metric} - not found in data")
                continue
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Distribution Analysis: {metric}', fontsize=16, fontweight='bold')
            
            # Get the data for this metric
            metric_data = y_data[metric].dropna()

            # Skip boolean columns
            if metric_data.dtype == bool:
                print(f"⚠️  Skipping {metric} - boolean data type not suitable for distribution plots")
                plt.close(fig)
                continue

            # Convert to numeric and filter out non-numeric data
            metric_data = pd.to_numeric(metric_data, errors='coerce').dropna()

            if len(metric_data) == 0:
                print(f"⚠️  Skipping {metric} - no valid numeric data")
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
            
            print(f"✅ Saved: {filename}")
        
        print(f"🎉 All distribution plots saved to {save_folder}/")

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
        
        print(f"✅ Saved correlation matrix: target_metrics_correlation_matrix.png")

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
        
        print(f"✅ Saved summary plot: target_metrics_summary.png")

    def _prepare_data_with_scaling(self):
        """Properly scale features AND targets with automatic transformation selection - FIXED VERSION"""
        from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
        from scipy import stats
        import numpy as np
        
        print("🔄 Preparing data with scaling and transformation...")
        
        # Initialize scalers and transformers
        self.X_scaler = None
        self.y_scalers = {}
        self.y_transformers = {}
        
        # STEP 1: Handle missing values and outliers FIRST
        print("🧹 Cleaning data...")
        
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
        print("📊 Analyzing feature distributions...")
        
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
        print("🎯 Processing target variables...")
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
                print(f"\n🔍 Processing {col}...")
                
                # Convert to numeric and handle issues
                train_data = pd.to_numeric(self.y_train[col], errors='coerce').dropna()
                test_data = pd.to_numeric(self.y_test[col], errors='coerce')
                
                if len(train_data) < 10:
                    print(f"⚠️  Skipping {col} - insufficient valid data")
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
                
                # Select best transformation
                valid_transformations = [t for t in transformations if t['score'] > -1000]

                if not valid_transformations:
                    print(f"   No valid transformations found for {col}, using original data")
                    transformed_train = train_data.values
                    transformed_test = test_data.values
                    self.y_transformers[col] = None
                else:
                    best_transform = max(valid_transformations, key=lambda x: x['score'])
                    
                    # IMPROVEMENT: Only use transformation if it significantly improves skewness
                    orig_abs_skew = abs(best_transform.get('orig_skew', 0))
                    trans_abs_skew = abs(best_transform.get('trans_skew', 0))
                    
                    skew_improvement = orig_abs_skew - trans_abs_skew
                    
                    # Require at least 20% reduction in absolute skewness
                    if skew_improvement < 0.2 * orig_abs_skew and best_transform['transform'] != "None":
                        print(f"   ⚠️ Transformation doesn't significantly improve skewness, using original")
                        print(f"      Original skew: {orig_abs_skew:.3f}, Transformed: {trans_abs_skew:.3f}")
                        transformed_train = train_data.values
                        transformed_test = test_data.values
                        self.y_transformers[col] = None
                    else:
                        print(f"   📈 Transformation results for {col}:")
                        for t in sorted(valid_transformations, key=lambda x: x['score'], reverse=True)[:3]:
                            print(f"      {t['transform']}: Score={t['score']:.2f}, "
                                f"Skew: {t['orig_skew']:.3f}→{t['trans_skew']:.3f}")
                        
                        print(f"   ✅ Selected: {best_transform['transform']}")
        
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
        
        print("\n✅ Data preparation complete!")
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
        """Create advanced features for better model performance - IMPROVED AND SAFER VERSION"""
        X_enhanced = X.copy()
        
        # Ensure we're working with numeric data
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return X_enhanced
        
        # 1. Statistical features (robust to missing values)
        print("   Creating statistical features...")
        X_enhanced['feature_mean'] = X[numeric_cols].mean(axis=1)
        X_enhanced['feature_std'] = X[numeric_cols].std(axis=1).fillna(0)
        X_enhanced['feature_median'] = X[numeric_cols].median(axis=1)
        X_enhanced['feature_max'] = X[numeric_cols].max(axis=1)
        X_enhanced['feature_min'] = X[numeric_cols].min(axis=1)
        X_enhanced['feature_range'] = X_enhanced['feature_max'] - X_enhanced['feature_min']
        X_enhanced['feature_skew'] = X[numeric_cols].skew(axis=1).fillna(0)
        
        # 2. Feature interactions (only if dataset is not too large)
        if len(numeric_cols) <= 20 and len(X) <= 10000:
            print("   Creating feature interactions...")
            # Create interactions between top correlated features
            corr_matrix = X[numeric_cols].corr().abs()
            
            # Find top correlated pairs
            top_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if 0.3 < corr_matrix.iloc[i, j] < 0.95:  # Moderate correlation
                        top_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            # Sort by correlation and take top 5
            top_pairs = sorted(top_pairs, key=lambda x: x[2], reverse=True)[:5]
            
            for feat1, feat2, corr in top_pairs:
                X_enhanced[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                X_enhanced[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-8)
        
        # 3. Domain-specific features (only if column names suggest biological data)
        print("   Creating domain-specific features...")
        
        # Group features by common prefixes
        feature_groups = {}
        for col in numeric_cols:
            prefix = col.split('_')[0] if '_' in col else 'other'
            if prefix not in feature_groups:
                feature_groups[prefix] = []
            feature_groups[prefix].append(col)
        
        # Create group-based features
        for group_name, group_cols in feature_groups.items():
            if len(group_cols) > 1:
                X_enhanced[f'{group_name}_sum'] = X[group_cols].sum(axis=1)
                X_enhanced[f'{group_name}_mean'] = X[group_cols].mean(axis=1)
                X_enhanced[f'{group_name}_std'] = X[group_cols].std(axis=1).fillna(0)
        
        # 4. Polynomial features for top features (if we have feature importance)
        if hasattr(self, 'feature_importance') and len(self.feature_importance) > 0:
            print("   Creating polynomial features...")
            # Get top 3 features from best model
            best_model = list(self.feature_importance.keys())[0]
            if len(self.feature_importance[best_model]) > 0:
                top_features = self.feature_importance[best_model].head(3)['feature'].tolist()
                
                for feat in top_features:
                    if feat in X.columns:
                        X_enhanced[f'{feat}_squared'] = X[feat] ** 2
                        X_enhanced[f'{feat}_log'] = np.log(np.abs(X[feat]) + 1e-8)
        
        # 5. Clean up any invalid values
        X_enhanced = X_enhanced.replace([np.inf, -np.inf], 0)
        X_enhanced = X_enhanced.fillna(0)
        
        # 6. Remove low-variance features from the new features only
        original_cols = X.columns
        new_features = [col for col in X_enhanced.columns if col not in original_cols]
        
        if new_features:
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            
            # Check variance of new features
            new_feature_data = X_enhanced[new_features]
            valid_new_features = []
            
            for col in new_features:
                if X_enhanced[col].var() > 0.01:
                    valid_new_features.append(col)
            
            # Keep original features + valid new features
            X_enhanced = X_enhanced[original_cols.tolist() + valid_new_features]
            
            print(f"   Added {len(valid_new_features)} new features (removed {len(new_features) - len(valid_new_features)} low-variance)")
        
        return X_enhanced


    def _validate_data_quality(self):
        """Enhanced data validation focusing on model training readiness"""
        print("🔍 Validating data quality for model training...")
        
        issues = []
        warnings = []
        
        # Check feature matrix
        print(f"📊 Feature matrix validation:")
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
        print(f"🎯 Target variables validation:")
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
        print(f"📏 Dataset size validation:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Features: {len(self.X_train.columns)}")
        print(f"   Feature-to-sample ratio: {len(self.X_train.columns) / len(self.X_train):.3f}")
        
        if len(self.X_train.columns) / len(self.X_train) > 0.5:
            warnings.append("High feature-to-sample ratio may cause overfitting")
        
        if len(self.X_train) < 100:
            issues.append("Very small training set - results may be unreliable")
        elif len(self.X_train) < 500:
            warnings.append("Small training set - consider regularization")
        
        # Summary
        print(f"\n📋 Validation Summary:")
        if issues:
            print(f"❌ Critical Issues ({len(issues)}):")
            for issue in issues:
                print(f"   • {issue}")
        
        if warnings:
            print(f"⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   • {warning}")
        
        if not issues and not warnings:
            print("✅ Data quality is good for model training")
        
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
            else:  # high complexity
                return xgb.XGBRegressor(
                    n_estimators=800,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.3,
                    reg_lambda=1.0,
                    gamma=0.1,
                    min_child_weight=2,
                    random_state=42,
                    n_jobs=-1
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
            else:  # high complexity
                return RandomForestRegressor(
                    n_estimators=500,
                    max_depth=15,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
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

    def train_supervised_models(self, target_metric: str = 'multi_objective_score') -> Dict:
        """
        COMPLETELY REWRITTEN: Train multiple supervised learning models with best practices
        """
        if target_metric not in self.y_train.columns:
            print(f"⚠️ Target metric '{target_metric}' not found. Using first available: {self.y_train.columns[0]}")
            target_metric = self.y_train.columns[0]
        
        print(f"🤖 Training supervised models for: {target_metric}")
        print("=" * 60)
        
        # Step 1: Validate data quality FIRST
        if not self._validate_data_quality():
            print("❌ Critical data quality issues found. Please fix before training.")
            return {}
        
        # Step 2: Prepare data with proper scaling and transformation
        self._prepare_data_with_scaling()
        
        # Step 3: Create enhanced features AFTER scaling
        print("🔧 Creating enhanced features...")
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
        print(f"\n📊 Target Analysis for '{target_metric}':")
        print(f"   Training samples: {len(y_target_original)}")
        print(f"   Range: [{y_target_original.min():.6f}, {y_target_original.max():.6f}]")
        print(f"   Mean ± Std: {y_target_original.mean():.6f} ± {y_target_original.std():.6f}")
        print(f"   Median: {y_target_original.median():.6f}")
        print(f"   Skewness: {y_target_original.skew():.4f}")
        print(f"   Missing values: {y_target_original.isnull().sum()}")
        print(f"   Target variance: {y_target_original.var():.6f}")
        
        # Check if target needs special handling
        target_range = y_target_original.max() - y_target_original.min()
        target_std = y_target_original.std()
        
        if target_range == 0 or target_std == 0:
            print("❌ Target has no variance - cannot train models")
            return {}
        
        # Step 6: Split data for cross-validation
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Use cross-validation for more robust evaluation
        cv_folds = 5
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Step 7: Train models with cross-validation
        models_to_train = ['XGBoost', 'LightGBM', 'RandomForest', 'MLP']
        all_results = {}
        
        for model_name in models_to_train:
            print(f"\n🏋️ Training {model_name}...")
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
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train_enhanced)):
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
                
                print(f"      R²: {fold_performance['val_r2']:.4f}, RMSE: {fold_performance['val_rmse']:.4f}, Norm MSE: {fold_performance['val_norm_mse']:.4f}")
            
            # Calculate average CV performance
            avg_performance = {}
            for metric, scores in cv_scores.items():
                avg_performance[f'{metric}_mean'] = np.mean(scores)
                avg_performance[f'{metric}_std'] = np.std(scores)
            
            print(f"   📊 CV Results:")
            print(f"      Validation R²: {avg_performance['val_r2_mean']:.4f} ± {avg_performance['val_r2_std']:.4f}")
            print(f"      Validation RMSE: {avg_performance['val_rmse_mean']:.4f} ± {avg_performance['val_rmse_std']:.4f}")
            print(f"      Validation Norm MSE: {avg_performance['val_norm_mse_mean']:.4f} ± {avg_performance['val_norm_mse_std']:.4f}")
            print(f"      Validation MAE: {avg_performance['val_mae_mean']:.4f} ± {avg_performance['val_mae_std']:.4f}")
            
            # Train final model on full training set
            print(f"   🎯 Training final model on full dataset...")
            
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
            
            print(f"   ✅ Final Test R²: {final_performance['test_r2']:.4f}")
            print(f"   ✅ Final Test RMSE: {final_performance['test_rmse']:.4f}")
            print(f"   ✅ Final Test Norm MSE: {final_performance['test_norm_mse']:.4f}")
            
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
                print(f"   ✅ Feature importance extracted for {model_name}")
            elif hasattr(final_model, 'coef_'):  # Linear models
                feature_names = self.X_train_enhanced.columns
                feature_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(final_model.coef_)
                }).sort_values('importance', ascending=False)
                self.feature_importance[f'{model_name}_{target_metric}'] = feature_imp
                print(f"   ✅ Feature importance extracted from coefficients for {model_name}")
        
        # Step 8: Model comparison and selection
        print(f"\n🏆 MODEL COMPARISON for {target_metric}")
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
        
        print(f"\n🎯 BEST MODEL: {best_model_name}")
        print(f"   Test R²: {best_performance['test_r2']:.4f}")
        print(f"   Test RMSE: {best_performance['test_rmse']:.4f}")
        print(f"   Test Normalized MSE: {best_performance['test_norm_mse']:.4f}")
        print(f"   CV R² (mean ± std): {best_performance['val_r2_mean']:.4f} ± {best_performance['val_r2_std']:.4f}")
        
        # Performance validation
        if best_performance['test_r2'] < 0.5:
            print("⚠️  WARNING: Best model R² < 0.5. Consider:")
            print("   • More feature engineering")
            print("   • Different target transformations")
            print("   • Ensemble methods")
            print("   • More data collection")
        elif best_performance['test_r2'] < 0.7:
            print("⚠️  MODERATE: Best model R² < 0.7. Room for improvement.")
        else:
            print("✅ GOOD: Model performance is satisfactory")
        
        # Store performance comparison
        self.model_performance[target_metric] = all_results
        self.model_comparison = comparison_df
        
        print("✅ Supervised model training completed!")
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
            print(f"      ⚠️  Metric calculation failed, using scaled targets: {e}")
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
            print(f"      ❌ Tensor conversion failed: {e}")
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
            print(f"      ❌ Final metric calculation failed: {e}")
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
            print(f"      🎯 Special handling for False Negative Rate...")
            
            # Check if it's a binary classification problem disguised as regression
            unique_values = len(np.unique(y_train_target))
            if unique_values <= 10:  # Likely categorical
                print(f"      💡 Converting to classification task ({unique_values} unique values)")
                # Return as-is and flag for classification
                return y_train_target, y_test_target, False
            
            # Check if values are bounded [0, 1] - apply logit transformation
            if y_train_target.min() >= 0 and y_train_target.max() <= 1:
                print(f"      🔧 Applying logit transformation for bounded [0,1] data")
                
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
            print(f"      ⚠️  CRITICAL: Target '{target_metric}' has constant/near-constant values!")
            print(f"      Unique values: {np.unique(y_train_target)}")
            
            # Check if this is a data issue or expected
            if np.all(y_train_target == 0.0):
                print(f"      💡 All values are 0.0 - this might indicate:")
                print(f"         - Calculation error in the target metric")
                print(f"         - Missing/uninitialized data")
                print(f"         - Metric not applicable to current dataset")
                print(f"      🔧 SKIPPING this target as it's not learnable")
                return None, None, True  # Skip this target
            
        # Check for targets with very low variance
        y_std = np.std(y_train_target)
        y_range = np.max(y_train_target) - np.min(y_train_target)
        
        if y_std < 1e-10 or y_range < 1e-10:
            print(f"      ⚠️  Target has extremely low variance (std: {y_std:.2e}, range: {y_range:.2e})")
            print(f"      🔧 SKIPPING this target as it's not learnable")
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
                print(f"      ⚠️  High outlier ratio: {outlier_ratio:.2%} ({outliers} outliers)")
                print(f"      🔧 Applying robust outlier handling...")
                
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
        """Train multiple models with different strategies and select the best"""
        
        strategies = []
        
        # Strategy 1: Ensemble model
        try:
            if is_classification:
                model1 = self._create_adaptive_ensemble_model(X_train_final.shape[1], 'classification', n_classes)
                criterion1 = nn.CrossEntropyLoss()
            else:
                model1 = self._create_adaptive_ensemble_model(X_train_final.shape[1], 'regression')
                criterion1 = nn.MSELoss()
            
            performance1 = self._train_single_model(model1, criterion1, X_train_final, y_train_processed, 
                                                X_test_final, y_test_processed, "Ensemble", is_classification)
            strategies.append(("Ensemble", model1, performance1))
        except Exception as e:
            print(f"      ❌ Ensemble strategy failed: {e}")
        
        # Strategy 2: Enhanced traditional model with better hyperparameters
        try:
            if is_classification:
                model2 = self._create_enhanced_traditional_model(X_train_final.shape[1], 'classification', n_classes)
                criterion2 = nn.CrossEntropyLoss()
            else:
                model2 = self._create_enhanced_traditional_model(X_train_final.shape[1], 'regression')
                criterion2 = nn.MSELoss()
            
            performance2 = self._train_single_model(model2, criterion2, X_train_final, y_train_processed, 
                                                X_test_final, y_test_processed, "Enhanced", is_classification)
            strategies.append(("Enhanced", model2, performance2))
        except Exception as e:
            print(f"      ❌ Enhanced strategy failed: {e}")
        
        # Strategy 3: XGBoost (if regression and performance is very poor)
        if not is_classification:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                xgb_model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    random_state=42
                )
                
                xgb_model.fit(X_train_final, y_train_processed)
                
                train_pred = xgb_model.predict(X_train_final)
                test_pred = xgb_model.predict(X_test_final)
                
                performance3 = {
                    'train_r2': r2_score(y_train_processed, train_pred),
                    'test_r2': r2_score(y_test_processed, test_pred),
                    'train_mse': mean_squared_error(y_train_processed, train_pred),
                    'test_mse': mean_squared_error(y_test_processed, test_pred)
                }
                
                strategies.append(("XGBoost", xgb_model, performance3))
                print(f"      🌳 XGBoost: R² = {performance3['test_r2']:.4f}")
                
            except Exception as e:
                print(f"      ❌ XGBoost strategy failed: {e}")
        
        # Strategy 4: CatBoost for difficult targets (if regression)
        if not is_classification and 'false_negative' in target_metric.lower():
            try:
                from catboost import CatBoostRegressor
                catboost_model = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.03,
                    depth=6,
                    loss_function='RMSE',
                    eval_metric='R2',
                    random_seed=42,
                    verbose=False
                )
                
                catboost_model.fit(X_train_final, y_train_processed)
                
                train_pred = catboost_model.predict(X_train_final)
                test_pred = catboost_model.predict(X_test_final)
                
                performance4 = {
                    'train_r2': r2_score(y_train_processed, train_pred),
                    'test_r2': r2_score(y_test_processed, test_pred),
                    'train_mse': mean_squared_error(y_train_processed, train_pred),
                    'test_mse': mean_squared_error(y_test_processed, test_pred)
                }
                
                strategies.append(("CatBoost", catboost_model, performance4))
                print(f"      🐱 CatBoost: R² = {performance4['test_r2']:.4f}")
                
            except Exception as e:
                print(f"      ❌ CatBoost strategy failed: {e}")
        
        # Select best strategy
        if not strategies:
            raise Exception("All training strategies failed!")
        
        if is_classification:
            best_strategy = max(strategies, key=lambda x: x[2].get('test_accuracy', 0))
            best_metric = best_strategy[2]['test_accuracy']
        else:
            best_strategy = max(strategies, key=lambda x: x[2].get('test_r2', -float('inf')))
            best_metric = best_strategy[2]['test_r2']
        
        print(f"      🏆 Best strategy: {best_strategy[0]} (score: {best_metric:.4f})")
        
        return best_strategy[1], best_strategy[2]

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
            print(f"      ⚠️  Index alignment failed, using provided y_data: {e}")
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
                print(f"      🤖 {strategy_name}: Acc = {performance['test_accuracy']:.4f}")
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
                print(f"      🤖 {strategy_name}: R² = {performance['test_r2']:.4f}")
        
        return performance

    def train_surrogate_models(self) -> Dict:
        """
        ENHANCED VERSION: Train surrogate models with better handling of problematic targets
        """
        print("🎭 Training ENHANCED surrogate models for RL...")
        print("=" * 60)
        
        # Initialize storage
        self.label_encoders = {}
        self.target_transformers = {}
        self.surrogate_feature_scalers = {}
        surrogate_performance = {}
        
        # Use the already prepared scaled features
        if not hasattr(self, 'X_train_scaled') or self.X_train_scaled is None:
            print("⚠️  Scaled features not found, preparing data...")
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
        
        # Apply additional scaling specifically for neural networks
        print("🔧 Applying neural network specific scaling...")
        
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
        
        print(f"📈 Training surrogate models for {len(self.y_train.columns)} targets")
        print(f"   Feature shape: {X_train_surrogate.shape}")
        
        # Track skipped targets
        skipped_targets = []
        
        # Train a surrogate for each target
        for target_idx, target_metric in enumerate(self.y_train.columns):
            print(f"\n🎯 Target {target_idx + 1}/{len(self.y_train.columns)}: {target_metric}")
            print("-" * 50)
            
            # Get target data
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
            
            # Handle problematic targets
            y_train_processed, y_test_processed, should_skip = self._handle_problematic_targets(
                target_metric, y_train_target, y_test_target
            )
            
            if should_skip:
                print(f"   ⏭️  Skipping target '{target_metric}' - not learnable")
                skipped_targets.append(target_metric)
                continue
            
            # Determine if classification or regression
            is_classification = y_train_target.dtype == 'object' or y_train_target.dtype.name == 'category'
            
            if is_classification:
                print(f"   📝 Classification task detected")
                
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
                print(f"   📊 Regression task detected")
                
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
                            print("      ✅ Target transformation applied")
                        else:
                            y_train_final = y_train_numeric.values
                            y_test_final = y_test_numeric.values
                            self.target_transformers[target_metric] = None
                            print("      ❌ Target transformation didn't improve, using original")
                            
                    except Exception as e:
                        print(f"      ❌ Target transformation failed: {e}")
                        y_train_final = y_train_numeric.values
                        y_test_final = y_test_numeric.values
                        self.target_transformers[target_metric] = None
                else:
                    y_train_final = y_train_numeric.values
                    y_test_final = y_test_numeric.values
                    self.target_transformers[target_metric] = None
                
                n_classes = 1  # For regression
            
            # Enhanced feature selection
            print(f"   🔍 Performing enhanced feature selection...")
            
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
                
                X_train_final = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train_surrogate.index)
                X_test_final = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test_surrogate.index)
                
                print(f"      Selected {len(selected_features)} features from {len(X_train_surrogate.columns)}")
                print(f"      Top 5 features: {list(selected_features[:5])}")
                
            except Exception as e:
                print(f"      ⚠️  Enhanced feature selection failed: {e}, using simpler method")
                
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
                
                X_train_final = X_train_surrogate[selected_features]
                X_test_final = X_test_surrogate[selected_features]
                
                print(f"      Selected {len(selected_features)} features from {len(X_train_surrogate.columns)}")
                print(f"      Top 5 features: {list(selected_features[:5])}")
            
            # Cross-validation with enhanced models
            from sklearn.model_selection import StratifiedKFold, KFold
            
            n_folds = min(5, len(X_train_final) // 15)
            n_folds = max(3, n_folds)
            
            if is_classification:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            print(f"   🔄 Training with {n_folds}-fold cross-validation using multiple strategies...")
            
            # Cross-validation with best model selection
            fold_performances = []
            
            # Convert to numpy for CV split to avoid index issues
            X_train_np = X_train_final.values if hasattr(X_train_final, 'values') else X_train_final
            y_train_np = y_train_final if isinstance(y_train_final, np.ndarray) else np.array(y_train_final)
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_np, y_train_np)):
                print(f"      Fold {fold_idx + 1}/{n_folds}: ", end="")
                
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
                        print(f"R²: {fold_perf['test_r2']:.3f}")
                    
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
                    print(f"   📊 CV Results: Accuracy: {avg_acc:.4f} ± {np.std([p['test_accuracy'] for p in valid_performances]):.4f}")
                    print(f"                 F1: {avg_f1:.4f} ± {np.std([p['test_f1'] for p in valid_performances]):.4f}")
                    main_metric = avg_acc
                else:
                    print("   ❌ All CV folds failed for classification")
                    continue
            else:
                valid_performances = [p for p in fold_performances if p['test_r2'] > -10]
                if valid_performances:
                    avg_r2 = np.mean([p['test_r2'] for p in valid_performances])
                    avg_mse = np.mean([p['test_mse'] for p in valid_performances])
                    print(f"   📊 CV Results: R²: {avg_r2:.4f} ± {np.std([p['test_r2'] for p in valid_performances]):.4f}")
                    print(f"                 MSE: {avg_mse:.4f} ± {np.std([p['test_mse'] for p in valid_performances]):.4f}")
                    main_metric = avg_r2
                else:
                    print("   ❌ All CV folds failed for regression")
                    continue
            
            # Train final model with the best strategy
            print(f"   🎯 Training final model with multiple strategies...")
            
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
                    print(f"   ✅ Final Test Accuracy: {final_performance['test_accuracy']:.4f}")
                    
                    # Quality assessment with better thresholds
                    if final_performance['test_accuracy'] < 0.5:
                        print("   ⚠️  VERY LOW: Test accuracy < 0.5")
                    elif final_performance['test_accuracy'] < 0.7:
                        print("   📈 LOW: Test accuracy < 0.7") 
                    elif final_performance['test_accuracy'] < 0.85:
                        print("   📊 MODERATE: Test accuracy < 0.85")
                    else:
                        print("   🎯 EXCELLENT: Test accuracy ≥ 0.85")
                        
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
                    
                    print(f"   ✅ Final Test R²: {final_performance['test_r2']:.4f}")
                    print(f"   ✅ Final Test MSE: {final_performance['test_mse']:.4f}")
                    
                    # Display normalized MSE metrics
                    if 'test_normalized_mse' in final_performance:
                        norm_mse = final_performance['test_normalized_mse']
                        print(f"   📊 Normalized MSE by variance: {norm_mse['mse_normalized_by_variance']:.4f}")
                        print(f"   📊 RMSE normalized by std: {norm_mse['rmse_normalized_by_std']:.4f}")
                        print(f"   📊 RMSE normalized by range: {norm_mse['rmse_normalized_by_range']:.4f}")
                    
                    # Enhanced quality assessment
                    if final_performance['test_r2'] < 0.0:
                        print("   ⚠️  VERY POOR: Test R² < 0.0 (worse than mean baseline)")
                    elif final_performance['test_r2'] < 0.3:
                        print("   ❌ POOR: Test R² < 0.3")
                    elif final_performance['test_r2'] < 0.6:
                        print("   📈 LOW: Test R² < 0.6")
                    elif final_performance['test_r2'] < 0.8:
                        print("   📊 MODERATE: Test R² < 0.8")
                    elif final_performance['test_r2'] < 0.9:
                        print("   🎯 GOOD: Test R² < 0.9")
                    else:
                        print("   🌟 EXCELLENT: Test R² ≥ 0.9")
            
            except Exception as e:
                print(f"   ❌ Final model training failed: {e}")
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
            # Save R² score
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

        print(f"✅ Surrogate configurations saved to: {config_path}")

        # Final summary
        print(f"\n🎉 ENHANCED SURROGATE MODEL TRAINING COMPLETE")
        print("=" * 60)
        
        successful_targets = list(surrogate_performance.keys())
        classification_targets = [k for k, v in surrogate_performance.items() if v['type'] == 'classification']
        regression_targets = [k for k, v in surrogate_performance.items() if v['type'] == 'regression']
        
        print(f"✅ Successfully trained: {len(successful_targets)}/{len(self.y_train.columns)} targets")
        
        if skipped_targets:
            print(f"⏭️  Skipped targets ({len(skipped_targets)}): {skipped_targets}")
        
        if classification_targets:
            print(f"\n📝 Classification targets ({len(classification_targets)}):")
            for target in classification_targets:
                perf = surrogate_performance[target]
                print(f"   {target}: Test Acc = {perf['test_accuracy']:.4f}")
        
        if regression_targets:
            print(f"\n📊 Regression targets ({len(regression_targets)}):")
            for target in regression_targets:
                perf = surrogate_performance[target]
                print(f"   {target}: Test R² = {perf['test_r2']:.4f}")
                
                # Show normalized MSE for regression targets
                if 'test_normalized_mse' in perf:
                    norm_mse = perf['test_normalized_mse']
                    print(f"      Normalized RMSE/std: {norm_mse['rmse_normalized_by_std']:.4f}")
        
        # Overall quality assessment
        if classification_targets:
            accuracies = [surrogate_performance[t]['test_accuracy'] for t in classification_targets]
            avg_class_acc = np.mean(accuracies)
            print(f"\n📈 Average Classification Accuracy: {avg_class_acc:.4f}")
            
            poor_class = [t for t in classification_targets if surrogate_performance[t]['test_accuracy'] < 0.6]
            if poor_class:
                print(f"⚠️  Poor classification targets: {poor_class}")
        
        if regression_targets:
            r2_scores = [surrogate_performance[t]['test_r2'] for t in regression_targets]
            avg_reg_r2 = np.mean(r2_scores)
            print(f"📈 Average Regression R²: {avg_reg_r2:.4f}")
            
            poor_reg = [t for t in regression_targets if surrogate_performance[t]['test_r2'] < 0.3]
            very_poor_reg = [t for t in regression_targets if surrogate_performance[t]['test_r2'] < 0.0]
            
            if very_poor_reg:
                print(f"❌ Very poor regression targets (R² < 0): {very_poor_reg}")
            if poor_reg:
                print(f"⚠️  Poor regression targets (R² < 0.3): {poor_reg}")
        
        # Recommendations for improvement
        print(f"\n💡 RECOMMENDATIONS:")
        if skipped_targets:
            print(f"   🔍 Investigate skipped targets - check data quality and metric calculations")
        
        poor_performers = []
        if regression_targets:
            poor_performers.extend([t for t in regression_targets if surrogate_performance[t]['test_r2'] < 0.3])
        if classification_targets:
            poor_performers.extend([t for t in classification_targets if surrogate_performance[t]['test_accuracy'] < 0.6])
        
        if poor_performers:
            print(f"   📊 Consider collecting more data for poor performers: {poor_performers}")
            print(f"   🔧 Consider feature engineering or domain-specific transformations")
            print(f"   🎯 Consider if these targets are truly predictable from available features")
        
        print("✅ Enhanced surrogate models ready for RL!")

        # CRITICAL: Validate R² thresholds before RL training
        print("\n🔍 SURROGATE MODEL FIDELITY VALIDATION")
        print("=" * 60)
        insufficient_surrogates = []
        for target, perf in surrogate_performance.items():
            if perf['type'] == 'regression':
                r2 = perf['test_r2']
                if r2 < 0.90:
                    insufficient_surrogates.append((target, r2))
                    print(f"⚠️  {target}: R² = {r2:.4f} < 0.90 (BELOW THRESHOLD)")
                elif r2 < 0.95:
                    print(f"✅ {target}: R² = {r2:.4f} (Acceptable, target 0.95)")
                else:
                    print(f"⭐ {target}: R² = {r2:.4f} (Outstanding)")

        if insufficient_surrogates:
            print(f"\n❌ CRITICAL: {len(insufficient_surrogates)} surrogates below R² = 0.90 threshold!")
            print("   RL agent will learn noise. Consider:")
            print("   1. Increase training data")
            print("   2. Better feature engineering")
            print("   3. Hyperparameter tuning")
            print("   4. Different model architectures")
            for target, r2 in insufficient_surrogates:
                print(f"      - {target}: {r2:.4f}")
        else:
            print("\n✅ All regression surrogates meet minimum R² ≥ 0.90 threshold!")
        # REMOVE INSUFFICIENT SURROGATES FROM RL TARGETS
        if insufficient_surrogates:
            print(f"\n🔧 AUTOMATIC FIX: Removing {len(insufficient_surrogates)} insufficient surrogates from RL targets")
            
            insufficient_target_names = [target for target, r2 in insufficient_surrogates]
            
            # Remove from surrogate_models
            for target in insufficient_target_names:
                if target in self.surrogate_models:
                    del self.surrogate_models[target]
                    print(f"   ❌ Removed {target} from surrogate_models")
            
            # Remove from target_metrics
            self.target_metrics = [m for m in self.target_metrics if m not in insufficient_target_names]
            
            print(f"   ✅ Updated target_metrics: {self.target_metrics}")
            print(f"   ⚠️  RL will only use high-fidelity surrogates (R² ≥ 0.90)")
        return surrogate_performance

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
        
        print(f"🎮 Setting up RL environment for: {target_metrics}")
        print("   RL Agent can ONLY modify circuit design parameters!")
        
        # Create logging directory
        self.log_dir = self.output_dir / "rl_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        class BiosensorEnv(gym.Env):
            def __init__(self, modifiable_bounds, fixed_values, surrogate_models, target_metrics, log_dir, surrogate_features, X_train, data_min_max):
                """
                Initialize environment with proper variable separation
                
                Args:
                    modifiable_bounds: Bounds for modifiable features only
                    fixed_values: Fixed values for environmental/simulation parameters
                    surrogate_models: Trained surrogate models
                    target_metrics: Target metrics to optimize
                    log_dir: Directory for logging
                    surrogate_features: Mapping from original to processed features
                """
                super(BiosensorEnv, self).__init__()
                
                self.log_dir = log_dir
                self.episode_logs = []
                self.step_logs = []
                self.current_episode = 0
                self.global_step = 0
                self.data_min_max = data_min_max
                
                # Store variable classifications
                self.modifiable_bounds = modifiable_bounds
                self.fixed_values = fixed_values
                self.surrogate_models = surrogate_models
                self.target_metrics = target_metrics
                self.surrogate_features = surrogate_features

                # STORE DATA MIN/MAX FOR MOS NORMALIZATION
                self.data_min_max = {}
                if hasattr(self, 'X_train'):
                    # Calculate from actual training data
                    for metric in target_metrics:
                        if metric in ['signal_to_noise_ratio_SNR', 'dynamic_range_of_output', 
                                    'false_negative_rate', 'time_to_detection_threshold']:
                            # Get from training data (you need to pass y_train to environment)
                            # For now, initialize with placeholders
                            self.data_min_max[metric] = (0, 1)  # MUST UPDATE from actual data
                
                # Only modifiable features can be changed by RL agent
                self.modifiable_features = list(modifiable_bounds.keys())
                self.fixed_features = list(fixed_values.keys())
                self.n_modifiable = len(self.modifiable_features)

                # Handle categorical features separately for RL
                self.categorical_features = {}
                self.continuous_features = []
                
                for feature in self.modifiable_features:
                    # Check if feature is categorical
                    if feature == 'circuit_topology':
                        # Define possible topologies from your dataset
                        self.categorical_features[feature] = ['direct', 'cascade', 'incoherent_feedforward']
                    elif feature == 'feedback_presence':
                        self.categorical_features[feature] = [0, 1]  # Binary
                    else:
                        self.continuous_features.append(feature)
                
                print(f"   Categorical modifiable features: {list(self.categorical_features.keys())}")
                print(f"   Continuous modifiable features: {self.continuous_features}")
                
                # Add after calculating modifiable_bounds:
                print(f"\n🔍 Feature Classification Check:")
                print(f"Modifiable features found: {len(modifiable_bounds)}")
                print(f"Fixed features found: {len(fixed_values)}")

                self.X_train = X_train

                # Check if key features from modifiable_bounds are properly set up
                if len(modifiable_bounds) > 0:
                    print(f"✅ {len(modifiable_bounds)} modifiable features configured:")
                    for feature in list(modifiable_bounds.keys())[:5]:  # Show first 5
                        print(f"   - {feature}")
                else:
                    print(f"❌ WARNING: No modifiable features found!")
                
                # Reward weights for different metrics (NEW SCLEROSTIN BIOSENSOR)
                self.metric_weights = {
                    'signal_to_noise_ratio_SNR': 0.45,      # Highest priority
                    'dynamic_range_of_output': 0.25,        # High priority
                    'false_negative_rate': 0.20,            # Low error priority (negative weight)
                    'time_to_detection_threshold': 0.10,    # Speed priority (negative weight)
                    'multi_objective_score': 1.0            # Combined metric
                }
                
                # Initialize tracking variables
                self.step_count = 0
                self.max_steps = 100  # Increase from 50
                self.reward_history = []
                self.best_reward = float('-inf')
                self.best_states = []

                # Add these missing attributes
                self.current_episode_rewards = []
                self.current_episode_states = []
                self.current_episode_actions = []
                self.current_episode_predictions = []
                
                # ACTION SPACE TRANSFORMATION (Section IV.1)
                # Discrete actions for computational efficiency
                self.action_mapping = {}
                action_idx = 0

                # For each modifiable continuous feature: 3 actions (decrease, no-op, increase)
                for feature in self.continuous_features:
                    self.action_mapping[action_idx] = ('continuous', feature, 'decrease', 0.9)  # 10% decrease
                    action_idx += 1
                    self.action_mapping[action_idx] = ('continuous', feature, 'keep', 1.0)  # NO-OP
                    action_idx += 1
                    self.action_mapping[action_idx] = ('continuous', feature, 'increase', 1.1)  # 10% increase
                    action_idx += 1

                # For each categorical feature: N actions (one per category)
                for feature, categories in self.categorical_features.items():
                    for category in categories:
                        self.action_mapping[action_idx] = ('categorical', feature, 'set', category)
                        action_idx += 1

                # Total discrete actions
                self.action_space = spaces.Discrete(len(self.action_mapping))

                print(f"   Discrete action space: {len(self.action_mapping)} total actions")
                print(f"   - Continuous features: {len(self.continuous_features)} × 3 = {len(self.continuous_features) * 3}")
                print(f"   - Categorical features: {sum(len(cats) for cats in self.categorical_features.values())}")
                
                # Observation space: modifiable features + 6 context values (added MOS)
                obs_size = self.n_modifiable + 6  # +6 for context info (including predicted MOS)
                self.observation_space = spaces.Box(
                    low=-3.0,
                    high=3.0,
                    shape=(obs_size,),
                    dtype=np.float32
                )
                
                # Initialize modifiable state
                self.modifiable_state = np.array([
                    np.random.uniform(bounds[0], bounds[1])
                    for bounds in self.modifiable_bounds.values()
                ], dtype=np.float32)
                
                # Validate surrogate models
                self._validate_surrogate_models()
                self._initialize_log_files()
                
                print(f"✅ Environment initialized with {self.n_modifiable} modifiable parameters")

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
                                    print(f"✅ Surrogate model {metric} validated (PyTorch)")
                        else:  # sklearn model (like GradientBoostingRegressor)
                            test_output = model.predict(dummy_full_state.reshape(1, -1))
                            if np.isnan(test_output).any():
                                print(f"WARNING: Surrogate model {metric} produces NaN!")
                            else:
                                print(f"✅ Surrogate model {metric} validated (sklearn)")
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
                """Create full state vector matching the surrogate model's expected input size"""
                # If no specific metric, use the first one as default
                if target_metric is None:
                    target_metric = list(self.surrogate_features.keys())[0]
                
                # Get the correct feature order for this specific surrogate model
                feature_order = self.surrogate_features[target_metric]
                expected_features = len(feature_order)
                
                full_state = np.zeros(expected_features)
                
                # Map features to their correct positions
                for idx, feature in enumerate(feature_order):
                    if feature in self.modifiable_features:
                        modifiable_idx = self.modifiable_features.index(feature)
                        full_state[idx] = self.modifiable_state[modifiable_idx]
                    elif feature in self.fixed_values:
                        full_state[idx] = self.fixed_values[feature]
                    # If feature not found, it stays 0 (already initialized)
                
                return full_state

            def _initialize_log_files(self):
                """Initialize CSV log files"""
                episode_log_path = os.path.join(self.log_dir, "episode_summary.csv")
                with open(episode_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'episode', 'total_reward', 'mean_reward', 'best_reward_in_episode',
                        'final_reward', 'steps_taken', 'convergence_achieved', 'reward_improvement', 'state_stability'
                    ])
                
                step_log_path = os.path.join(self.log_dir, "step_details.csv")
                with open(step_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = ['episode', 'step', 'action', 'reward', 'cumulative_reward']
                    headers.extend([f'modifiable_{name}' for name in self.modifiable_features])
                    headers.extend([f'pred_{metric}' for metric in self.target_metrics])
                    writer.writerow(headers)
                
                # Initialize trajectory evolution CSV with the specified columns
                trajectory_log_path = os.path.join(self.log_dir, "trajectory_evolution.csv")
                with open(trajectory_log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Define the specific column headers as requested
                    headers = [
                        'episode', 'step',
                        # Value columns
                        'reporter_maturation_time_value', 'env_intrinsic_noise_level_value', 'reporter_signal_strength_value',
                        'terminator_stability_value', 'arch_complexity_value', 'env_translational_variability_value',
                        'promoter_strength_value', 'env_oxidative_stress_value', 'env_dynamic_stress_value',
                        'arch_regulatory_elements_value', 'env_resource_competition_value', 'sim_mRNA_degradation_rate_value',
                        'rbs_temperature_sensitivity_value', 'sim_protein_degradation_rate_value', 'copy_multiplier_value',
                        'env_pH_actual_value', 'env_total_stress_value', 'env_noise_frequency_value',
                        'target_biomarker_binding_affinity_value', 'env_metabolic_flux_noise_value', 'copy_noise_factor_value',
                        'promoter_noise_sensitivity_value', 'terminator_efficiency_value', 'concentration_ratio_value',
                        'sim_transcription_rate_value', 'promoter_leakiness_value', 'env_ionic_strength_value',
                        'env_immune_signal_level_value', 'env_temperature_actual_value', 'target_biomarker_threshold_value',
                        'conc_category_Low_value', 'env_noise_amplitude_modulation_value', 'env_metabolic_load_value',
                        'env_static_stress_value', 'env_temperature_variation_value', 'sim_random_seed_value',
                        'env_pH_variation_value', 'env_extrinsic_noise_level_value', 'sim_translation_rate_value',
                        'env_target_biomarker_concentration_value', 'env_noise_autocorrelation_value', 'rbs_efficiency_value',
                        'env_transcriptional_bursting_value',
                        # Change columns (absolute)
                        'reporter_maturation_time_change', 'env_intrinsic_noise_level_change', 'reporter_signal_strength_change',
                        'terminator_stability_change', 'arch_complexity_change', 'env_translational_variability_change',
                        'promoter_strength_change', 'env_oxidative_stress_change', 'env_dynamic_stress_change',
                        'arch_regulatory_elements_change', 'env_resource_competition_change', 'sim_mRNA_degradation_rate_change',
                        'rbs_temperature_sensitivity_change', 'sim_protein_degradation_rate_change', 'copy_multiplier_change',
                        'env_pH_actual_change', 'env_total_stress_change', 'env_noise_frequency_change',
                        'target_biomarker_binding_affinity_change', 'env_metabolic_flux_noise_change', 'copy_noise_factor_change',
                        'promoter_noise_sensitivity_change', 'terminator_efficiency_change', 'concentration_ratio_change',
                        'sim_transcription_rate_change', 'promoter_leakiness_change', 'env_ionic_strength_change',
                        'env_immune_signal_level_change', 'env_temperature_actual_change', 'target_biomarker_threshold_change',
                        'conc_category_Low_change', 'env_noise_amplitude_modulation_change', 'env_metabolic_load_change',
                        'env_static_stress_change', 'env_temperature_variation_change', 'sim_random_seed_change',
                        'env_pH_variation_change', 'env_extrinsic_noise_level_change', 'sim_translation_rate_change',
                        'env_target_biomarker_concentration_change', 'env_noise_autocorrelation_change', 'rbs_efficiency_change',
                        'env_transcriptional_bursting_change',
                        # Change percentage columns
                        'reporter_maturation_time_change_pct', 'env_intrinsic_noise_level_change_pct', 'reporter_signal_strength_change_pct',
                        'terminator_stability_change_pct', 'arch_complexity_change_pct', 'env_translational_variability_change_pct',
                        'promoter_strength_change_pct', 'env_oxidative_stress_change_pct', 'env_dynamic_stress_change_pct',
                        'arch_regulatory_elements_change_pct', 'env_resource_competition_change_pct', 'sim_mRNA_degradation_rate_change_pct',
                        'rbs_temperature_sensitivity_change_pct', 'sim_protein_degradation_rate_change_pct', 'copy_multiplier_change_pct',
                        'env_pH_actual_change_pct', 'env_total_stress_change_pct', 'env_noise_frequency_change_pct',
                        'target_biomarker_binding_affinity_change_pct', 'env_metabolic_flux_noise_change_pct', 'copy_noise_factor_change_pct',
                        'promoter_noise_sensitivity_change_pct', 'terminator_efficiency_change_pct', 'concentration_ratio_change_pct',
                        'sim_transcription_rate_change_pct', 'promoter_leakiness_change_pct', 'env_ionic_strength_change_pct',
                        'env_immune_signal_level_change_pct', 'env_temperature_actual_change_pct', 'target_biomarker_threshold_change_pct',
                        'conc_category_Low_change_pct', 'env_noise_amplitude_modulation_change_pct', 'env_metabolic_load_change_pct',
                        'env_static_stress_change_pct', 'env_temperature_variation_change_pct', 'sim_random_seed_change_pct',
                        'env_pH_variation_change_pct', 'env_extrinsic_noise_level_change_pct', 'sim_translation_rate_change_pct',
                        'env_target_biomarker_concentration_change_pct', 'env_noise_autocorrelation_change_pct', 'rbs_efficiency_change_pct',
                        'env_transcriptional_bursting_change_pct'
                    ]
                    
                    writer.writerow(headers)

            def reset(self):
                """Reset environment with better initialization strategies"""
                
                # IMPORTANT: Log episode summary for the previous episode (if it exists)
                if hasattr(self, 'current_episode_rewards') and len(self.current_episode_rewards) > 0:
                    self._log_episode_summary()
                
                # Better initialization strategies
                init_strategy = np.random.choice(['random', 'near_best', 'opposite_best', 'center'])
                
                if init_strategy == 'random':
                    # Pure random initialization
                    self.modifiable_state = np.array([
                        np.random.uniform(bounds[0], bounds[1])
                        for bounds in self.modifiable_bounds.values()
                    ], dtype=np.float32)
                
                elif init_strategy == 'near_best' and hasattr(self, 'best_states') and len(self.best_states) > 0:
                    # Start near a good state but with more noise
                    base_state = self.best_states[np.random.randint(len(self.best_states))]
                    noise_scale = 0.2  # Larger noise
                    noise = np.random.normal(0, noise_scale, size=len(base_state))
                    self.modifiable_state = base_state + noise
                
                elif init_strategy == 'opposite_best' and hasattr(self, 'best_states') and len(self.best_states) > 0:
                    # Start at opposite end of parameter space from best
                    best_state = self.best_states[-1]  # Most recent best
                    opposite_state = np.zeros_like(best_state)
                    
                    for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                        range_size = bounds[1] - bounds[0]
                        normalized_best = (best_state[i] - bounds[0]) / range_size
                        opposite_normalized = 1.0 - normalized_best
                        opposite_state[i] = bounds[0] + opposite_normalized * range_size
                    
                    self.modifiable_state = opposite_state
                
                else:  # center
                    # Start at center of parameter space
                    self.modifiable_state = np.array([
                        (bounds[0] + bounds[1]) / 2.0
                        for bounds in self.modifiable_bounds.values()
                    ], dtype=np.float32)
                
                self._clip_modifiable_state()
                
                # Reset tracking
                self.step_count = 0
                self.reward_history = []
                self.current_episode += 1
                
                self.current_episode_rewards.clear()
                self.current_episode_states.clear()
                self.current_episode_actions.clear()
                self.current_episode_predictions.clear()
                
                return self._get_observation()

            def _clip_modifiable_state(self):
                """Clip modifiable state to bounds"""
                for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                    self.modifiable_state[i] = np.clip(self.modifiable_state[i], bounds[0], bounds[1])

            def _get_observation(self):
                """Get normalized observation of modifiable features + context"""
                # Normalize modifiable features
                normalized_modifiable = np.zeros_like(self.modifiable_state)
                for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                    range_val = bounds[1] - bounds[0]
                    if range_val > 0:
                        normalized_val = (self.modifiable_state[i] - bounds[0]) / range_val
                        normalized_modifiable[i] = 2 * normalized_val - 1
                    else:
                        normalized_modifiable[i] = 0.0
                
                # ENHANCED STATE SPACE (Section IV.2)
                # Include predicted MOS in observation
                step_ratio = self.step_count / self.max_steps

                # Calculate current predicted MOS
                try:
                    current_mos = 0.0
                    if hasattr(self, 'surrogate_models'):
                        # Quick MOS calculation
                        for metric in ['signal_to_noise_ratio_SNR', 'dynamic_range_of_output', 
                                    'false_negative_rate', 'time_to_detection_threshold']:
                            if metric in self.surrogate_models:
                                model = self.surrogate_models[metric]
                                full_state = self._create_full_state(target_metric=metric)
                                state_tensor = torch.FloatTensor(full_state).unsqueeze(0)
                                
                                if hasattr(model, 'eval'):
                                    model.eval()
                                    with torch.no_grad():
                                        pred = model(state_tensor).item()
                                else:
                                    pred = model.predict(full_state.reshape(1, -1))[0]
                                
                                # Add to MOS (simplified, not normalized)
                                if metric == 'signal_to_noise_ratio_SNR':
                                    current_mos += 0.45 * (pred / 100.0)
                                elif metric == 'dynamic_range_of_output':
                                    current_mos += 0.25 * (pred / 20000.0)
                                # FNR and TTD would be subtracted
                except:
                    current_mos = 0.0

                if len(self.reward_history) > 0:
                    recent_rewards = self.reward_history[-5:]
                    reward_mean = np.mean(recent_rewards)
                    reward_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
                    reward_trend = recent_rewards[-1] - recent_rewards[0] if len(recent_rewards) > 1 else 0.0
                    current_reward = self.reward_history[-1]
                else:
                    reward_mean = reward_std = reward_trend = current_reward = 0.0

                # INCLUDE PREDICTED MOS IN STATE (Section IV.2)
                obs = np.concatenate([
                    normalized_modifiable,
                    [step_ratio, reward_mean, reward_std, reward_trend, current_reward, current_mos]
                ]).astype(np.float32)
                
                # Safety check
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                obs = np.clip(obs, -3.0, 3.0)
                
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
                """Step environment with proper handling of continuous and categorical actions"""
                self.step_count += 1
                self.global_step += 1
                
                # Store previous state for change tracking
                prev_state = self.modifiable_state.copy()
                
                # DECODE DISCRETE ACTION using mapping
                action_info = self.action_mapping[action]
                action_type = action_info[0]  # 'continuous' or 'categorical'
                feature_name = action_info[1]
                operation = action_info[2]
                value = action_info[3]

                actual_idx = self.modifiable_features.index(feature_name)

                if action_type == 'continuous':
                    # Apply multiplier
                    bounds = list(self.modifiable_bounds.values())[actual_idx]
                    current_val = self.modifiable_state[actual_idx]
                    
                    if operation == 'decrease':
                        new_value = current_val * value  # value = 0.9
                    elif operation == 'keep':
                        new_value = current_val  # NO-OP
                    else:  # increase
                        new_value = current_val * value  # value = 1.1
                    
                    # Clip to bounds
                    self.modifiable_state[actual_idx] = np.clip(new_value, bounds[0], bounds[1])

                elif action_type == 'categorical':
                    # Set categorical value
                    if feature_name == 'feedback_presence':
                        self.modifiable_state[actual_idx] = float(value)
                    elif feature_name == 'circuit_topology':
                        topology_map = {'direct': 0.0, 'cascade': 0.5, 'incoherent_feedforward': 1.0}
                        self.modifiable_state[actual_idx] = topology_map.get(value, 0.0)
                
                # Calculate reward and predictions
                reward, predictions = self._calculate_reward_with_predictions()
                
                # Track performance
                self.reward_history.append(reward)
                self.current_episode_rewards.append(reward)
                self.current_episode_states.append(self.modifiable_state.copy())
                self.current_episode_actions.append(action)
                self.current_episode_predictions.append(predictions)
                
                if reward > self.best_reward:
                    self.best_reward = reward
                    if not hasattr(self, 'best_states'):
                        self.best_states = []
                    self.best_states.append(self.modifiable_state.copy())
                    if len(self.best_states) > 20:  # Keep more best states
                        self.best_states = self.best_states[-20:]
                
                # Log step details
                self._log_step_details(action, reward, predictions)
                
                # Log trajectory evolution
                self._log_trajectory_evolution(prev_state)
                
                # Determine if episode is done
                done = bool(self.step_count >= self.max_steps)  # Ensure scalar boolean
                
                # IMPORTANT: Log episode summary when episode ends
                if done:
                    self._log_episode_summary()
                
                # Ensure reward is scalar, not array
                reward = float(reward) if hasattr(reward, '__float__') else reward
                done = bool(done)

                # Ensure reward and done are proper scalar/array types for vectorized env
                info = {
                    'raw_reward': reward if isinstance(reward, np.ndarray) else float(reward),
                    'step_count': int(self.step_count),
                    'best_reward': float(self.best_reward),
                    'predictions': predictions,
                    'exploration_bonus': 0.0,
                    'parameter_diversity': int(len([i for i in range(len(self.modifiable_state)) 
                                                if abs(self.modifiable_state[i] - prev_state[i]) > 0.01]))
                }

                return self._get_observation(), reward, done, info
                            
            def _calculate_reward_with_predictions(self):
                """Calculate reward using MOS with comprehensive error handling"""
                try:
                    # Initialize predictions dictionary
                    predictions = {}
                    
                    # Step 1: Get predictions for all available metrics
                    for metric in self.target_metrics:
                        if metric not in self.surrogate_models:
                            continue
                            
                        try:
                            model = self.surrogate_models[metric]
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
                                
                        except Exception as e:
                            print(f"⚠️  Failed to predict {metric}: {e}")
                            predictions[metric] = 0.0
                    
                    # Step 2: Calculate MOS
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
                        """Safely normalize with fallbacks"""
                        if metric_name in ranges:
                            min_val, max_val = ranges[metric_name]
                        elif metric_name in default_ranges:
                            min_val, max_val = default_ranges[metric_name]
                        else:
                            return 0.5  # Default middle value
                        
                        if max_val - min_val < 1e-10:
                            return 0.5
                        
                        normalized = (value - min_val) / (max_val - min_val)
                        return np.clip(normalized, 0.0, 1.0)
                    
                    # Calculate MOS components
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
                        mos += 0.20 * (1.0 - fnr_norm)
                    
                    # Time to Detection (minimize, w=0.10)
                    if 'time_to_detection_threshold' in predictions:
                        ttd_norm = safe_normalize(predictions['time_to_detection_threshold'],
                                                'time_to_detection_threshold')
                        mos += 0.10 * (1.0 - ttd_norm)
                    
                    # Constraint penalty
                    constraint_penalty = 0.0
                    for i, (feature, bounds) in enumerate(self.modifiable_bounds.items()):
                        if self.modifiable_state[i] < bounds[0] or self.modifiable_state[i] > bounds[1]:
                            constraint_penalty = 1.0
                            break
                    
                    # Final reward
                    final_reward = mos - constraint_penalty
                    final_reward = np.clip(final_reward, 0.0, 1.0)
                    
                    predictions['mos_score'] = mos
                    predictions['constraint_penalty'] = constraint_penalty
                    
                    return final_reward, predictions
                    
                except Exception as e:
                    print(f"❌ Error in reward calculation: {e}")
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
            
            def _log_step_details(self, action, reward, predictions):
                """Log detailed step information"""
                step_log_path = os.path.join(self.log_dir, "step_details.csv")
                with open(step_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    cumulative_reward = sum(self.current_episode_rewards)
                    row = [
                        self.current_episode, self.step_count, 
                        action, reward, cumulative_reward
                    ]
                    
                    # Add modifiable state values
                    row.extend(self.modifiable_state.tolist())
                    
                    # Add predictions
                    for metric in self.target_metrics:
                        row.append(predictions.get(metric, np.nan))
                    
                    writer.writerow(row)
            
            def _log_trajectory_evolution(self, prev_state):
                """Log trajectory evolution with state changes"""
                trajectory_log_path = os.path.join(self.log_dir, "trajectory_evolution.csv")
                with open(trajectory_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    row = [self.current_episode, self.step_count]
                    
                    # Current state values
                    row.extend(self.modifiable_state.tolist())
                    
                    # State changes (absolute)
                    changes = self.modifiable_state - prev_state
                    row.extend(changes.tolist())
                    
                    # State changes (percentage)
                    pct_changes = []
                    for i, (current, previous) in enumerate(zip(self.modifiable_state, prev_state)):
                        if abs(previous) > 1e-10:
                            pct_change = (current - previous) / previous * 100
                        else:
                            pct_change = 0.0
                        pct_changes.append(pct_change)
                    
                    row.extend(pct_changes)
                    writer.writerow(row)

            def _log_episode_summary(self):
                """Log episode summary statistics"""
                # Only log if we have rewards to log
                if not hasattr(self, 'current_episode_rewards') or len(self.current_episode_rewards) == 0:
                    return
                    
                episode_log_path = os.path.join(self.log_dir, "episode_summary.csv")
                with open(episode_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    total_reward = sum(self.current_episode_rewards)
                    mean_reward = np.mean(self.current_episode_rewards)
                    best_reward = max(self.current_episode_rewards)
                    final_reward = self.current_episode_rewards[-1]
                    
                    # Check convergence (reward stability in last 5 steps)
                    if len(self.current_episode_rewards) >= 5:
                        last_5_rewards = self.current_episode_rewards[-5:]
                        reward_std = np.std(last_5_rewards)
                        convergence_achieved = reward_std < 0.01
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
                        self.current_episode, total_reward, mean_reward, best_reward,
                        final_reward, len(self.current_episode_rewards), convergence_achieved,
                        reward_improvement, state_stability
                    ])

                    # ENHANCED DIAGNOSIS LOGGING (Section VI)
                    if self.current_episode % 50 == 0:  # Log every 50 episodes
                        print(f"\n📊 CONVERGENCE DIAGNOSTICS (Episode {self.current_episode}):")
                        print(f"   Mean Reward (last 50): {np.mean(self.current_episode_rewards):.4f}")
                        print(f"   Std Reward: {np.std(self.current_episode_rewards):.4f}")
                        print(f"   Best Reward: {best_reward:.4f}")
                        print(f"   Reward Improvement: {reward_improvement:.4f}")
                        print(f"   State Stability: {state_stability:.4f}")
                        
                        # Check for learning stagnation
                        if len(self.current_episode_rewards) >= 10:
                            recent_trend = np.mean(self.current_episode_rewards[-10:]) - np.mean(self.current_episode_rewards[-20:-10]) if len(self.current_episode_rewards) >= 20 else 0
                            if abs(recent_trend) < 0.001:
                                print(f"   ⚠️  WARNING: Possible learning stagnation (trend: {recent_trend:.6f})")
                    
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
                
                print(f"✅ All logs saved to: {self.log_dir}")

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
        print("📊 Calculating data ranges for MOS normalization...")
        data_min_max = {}

        # Target metrics that need normalization for MOS
        mos_metrics = [
            'signal_to_noise_ratio_SNR',
            'dynamic_range_of_output', 
            'false_negative_rate',
            'time_to_detection_threshold'
        ]

        for metric in mos_metrics:
            if metric in self.y_train.columns:
                min_val = float(self.y_train[metric].min())
                max_val = float(self.y_train[metric].max())
                
                # Ensure valid range
                if max_val <= min_val:
                    print(f"⚠️  {metric} has invalid range [{min_val}, {max_val}], using defaults")
                    if 'snr' in metric.lower():
                        min_val, max_val = 0, 100
                    elif 'dynamic_range' in metric.lower():
                        min_val, max_val = 0, 20000
                    elif 'false_negative' in metric.lower():
                        min_val, max_val = 0, 1
                    elif 'time_to_detection' in metric.lower():
                        min_val, max_val = 0, 200
                
                data_min_max[metric] = (min_val, max_val)
                print(f"   ✅ {metric}: [{min_val:.4f}, {max_val:.4f}]")
            else:
                print(f"   ⚠️  {metric} not found in training data, will use defaults")

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
                print(f"   ⚠️  Using default range for {metric}: {default_range}")

        # VERIFY target metrics match available surrogates
        print("\n🔍 Verifying RL environment setup...")
        print(f"   Requested target_metrics: {target_metrics}")
        print(f"   Available surrogate_models: {list(self.surrogate_models.keys())}")

        # Filter to only use metrics that have surrogate models
        valid_target_metrics = [m for m in target_metrics if m in self.surrogate_models]

        if len(valid_target_metrics) < len(target_metrics):
            missing = set(target_metrics) - set(valid_target_metrics)
            print(f"   ⚠️  WARNING: Some target metrics have no surrogate model: {missing}")
            print(f"   ✅ Using only valid metrics: {valid_target_metrics}")
            target_metrics = valid_target_metrics

        if not target_metrics:
            raise ValueError("No valid target metrics with surrogate models available!")

        # Pass it when creating environment:
        self.rl_env = BiosensorEnv(
            modifiable_bounds=modifiable_bounds,
            fixed_values=fixed_values,
            surrogate_models=self.surrogate_models,
            target_metrics=target_metrics,
            log_dir=self.log_dir,
            surrogate_features=self.surrogate_features,
            X_train=self.X_train,
            data_min_max=data_min_max 
        )

        # PARALLELIZATION (Section V: Time 1)
        # Wrap in vectorized environment for faster training
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            
            # SAVE log_dir BEFORE wrapping
            self.rl_log_dir = self.log_dir  # Store at pipeline level
            
            # Use 4-8 parallel environments (Section V)
            n_envs = min(8, os.cpu_count() or 4)
            print(f"🚀 Creating {n_envs} parallel environments for faster training...")
            
            # Create environment factory
            def make_env():
                return BiosensorEnv(
                    modifiable_bounds=modifiable_bounds,
                    fixed_values=fixed_values,
                    surrogate_models=self.surrogate_models,
                    target_metrics=target_metrics,
                    log_dir=self.log_dir,
                    surrogate_features=self.surrogate_features,
                    X_train=self.X_train,
                    data_min_max=data_min_max.copy()  
                )
                        
            # Use SubprocVecEnv for true parallelization
            env_fns = [make_env for _ in range(n_envs)]
            self.rl_env = SubprocVecEnv(env_fns)
            print(f"✅ Parallel environments created successfully!")
            
        except Exception as e:
            print(f"⚠️  Could not create parallel environments: {e}")
            print("   Falling back to single environment")
            # self.rl_env already set to single environment above

        # VALIDATION: Test environment before training
        print("\n🧪 Testing RL environment setup...")
        try:
            obs = self.rl_env.reset()
            print(f"✅ Reset successful, obs shape: {obs.shape if hasattr(obs, 'shape') else 'scalar'}")
            
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
            
            obs, reward, done, info = self.rl_env.step(actions)
        
            # Handle both single env (info is dict) and vectorized env (info is list of dicts)
            if isinstance(info, list):
                # Vectorized environment - info is a list of dicts, one per env
                first_info = info[0] if info else {}
                predictions_keys = list(first_info.get('predictions', {}).keys()) if isinstance(first_info, dict) else []
            elif isinstance(info, dict):
                # Single environment - info is a dict
                print(type(info))
                print(info.keys())
                print(type(info.get('predictions')))
                predictions_keys = list(info.get('predictions', {}).keys())
            else:
                print("WARNING: self.rl_env.step(actions) did not return a dictionary or lists!")
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

            print(f"✅ Step successful")
            print(f"   Reward: {reward} (valid: {reward_is_scalar})")
            print(f"   Done: {done} (bool: {done_is_bool})")
            print(f"   Predictions: {predictions_keys}")
                        
            if not reward_is_scalar or not done_is_bool:
                print(f"⚠️  WARNING: Non-scalar reward or done - this may cause training issues")
                
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("✅ Advanced RL environment with proper variable classification created!")

        return self.rl_env
    
    
    def get_stable_training_configs(self):
        # Improved DQN config - better exploration
        dqn_config = {
            "policy": "MlpPolicy",
            "env": self.rl_env,
            "verbose": 1,
            "learning_rate": 3e-4,  # Slightly higher for faster learning
            "buffer_size": 100000,  # Reduced for faster updates
            "learning_starts": 5000,  # Reduced warm-up
            "batch_size": 64,  # Smaller batches for more frequent updates
            "target_update_interval": 500,  # More frequent target updates
            "exploration_fraction": 0.8,  # More exploration time
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.15,  # Higher final exploration
            "train_freq": 2,  # More frequent training
            "gradient_steps": 1,
            "policy_kwargs": {
                "net_arch": [128, 128, 64],  # Smaller but faster network
                "activation_fn": torch.nn.ReLU,
            }
        }
        
        # OPTIMIZED PPO CONFIG (Section IV.3)
        ppo_config = {
            "policy": "MlpPolicy", 
            "env": self.rl_env,
            "verbose": 1,
            "learning_rate": self._linear_schedule(3e-4),  # Linear decay
            "n_steps": 4096,  # High for quality updates
            "batch_size": 256,  # Efficient batch size
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.08,  # Higher entropy for exploration (Section IV.3)
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256, 128],  # Larger network
                "activation_fn": torch.nn.ReLU,
            }
        }
        
        return dqn_config, ppo_config
    
    def _linear_schedule(self, initial_value):
        """
        Linear learning rate schedule (Section IV.3)
        """
        def schedule(progress_remaining):
            return progress_remaining * initial_value
        return schedule

    def train_rl_agents(self, total_timesteps: int = 500000):  # SECTION IV.3: Minimum 500k
        """Train RL agents with much better exploration and longer training"""
        
        if self.rl_env is None:
            print("⚠️ Setting up RL environment first...")
            self.setup_rl_environment(self.target_metrics)
        
        print(f"🎯 Training RL agents for {total_timesteps} timesteps...")
        
        # Get stable configs
        dqn_config, ppo_config = self.get_stable_training_configs()
        
        agents_performance = {}
        
        # REMOVE CURRICULUM LEARNING - let agents explore freely
        # Initialize agents
        print("   Initializing DQN and PPO agents...")
        try:
            dqn_agent = DQN(**dqn_config)
            ppo_agent = PPO(**ppo_config)
        except Exception as e:
            print(f"      ❌ Agent initialization failed: {e}")
            return {}
        
        # Train DQN with checkpoints
        if dqn_agent is not None:
            print("   Training DQN with exploration focus...")
            try:
                # Train in chunks to monitor progress
                chunk_size = total_timesteps // 5
                for i in range(5):
                    print(f"      DQN Training chunk {i+1}/5...")
                    dqn_agent.learn(total_timesteps=chunk_size)
                    
                    # Quick evaluation
                    if i % 2 == 0:
                        obs = self.rl_env.reset()
                        total_reward = 0
                        for _ in range(50):
                            action, _ = dqn_agent.predict(obs, deterministic=False)  # Keep stochastic
                            obs, reward, done, _ = self.rl_env.step(action)
                            total_reward += reward
                            if done:
                                break
                        print(f"         Current performance: {total_reward:.4f}")
                        
            except Exception as e:
                print(f"      ❌ DQN training failed: {e}")
                dqn_agent = None
        
        # Train PPO with checkpoints
        if ppo_agent is not None:
            print("   Training PPO with exploration focus...")
            try:
                # Train in chunks to monitor progress
                chunk_size = total_timesteps // 5
                for i in range(5):
                    print(f"      PPO Training chunk {i+1}/5...")
                    ppo_agent.learn(total_timesteps=chunk_size)
                    
                    # Quick evaluation
                    if i % 2 == 0:
                        obs = self.rl_env.reset()
                        total_reward = 0
                        for _ in range(50):
                            action, _ = ppo_agent.predict(obs, deterministic=False)  # Keep stochastic
                            obs, reward, done, _ = self.rl_env.step(action)
                            total_reward += reward
                            if done:
                                break
                        print(f"         Current performance: {total_reward:.4f}")
                        
            except Exception as e:
                print(f"      ❌ PPO training failed: {e}")
                ppo_agent = None
        
        # Final evaluation with MORE episodes
        if dqn_agent is not None:
            print("   Final DQN evaluation...")
            try:
                dqn_rewards = []
                n_eval_episodes = 20
                
                # Check if vectorized
                is_vec_env = hasattr(self.rl_env, 'num_envs')
                
                for i in range(n_eval_episodes):
                    obs = self.rl_env.reset()
                    episode_rewards = 0.0
                    done_flags = np.zeros(self.rl_env.num_envs if is_vec_env else 1, dtype=bool)
                    
                    for step in range(1000):  # Max steps per episode
                        action, _ = dqn_agent.predict(obs, deterministic=True)
                        obs, reward, done, _ = self.rl_env.step(action)
                        
                        # Handle rewards
                        if is_vec_env:
                            # For vectorized env: accumulate only for non-done envs
                            episode_rewards += np.sum(reward[~done_flags])
                            done_flags = np.logical_or(done_flags, done)
                            
                            # Break if all environments are done
                            if np.all(done_flags):
                                break
                        else:
                            # Single environment - ensure scalar
                            reward_scalar = float(reward) if hasattr(reward, '__float__') else reward
                            done_scalar = bool(done) if hasattr(done, '__bool__') else done
                            
                            episode_rewards += reward_scalar
                            if done_scalar:
                                break
                    
                    # Average reward across environments if vectorized
                    if is_vec_env:
                        episode_rewards /= self.rl_env.num_envs
                    
                    dqn_rewards.append(episode_rewards)
                    
                    if i % 5 == 0:
                        print(f"      DQN Episode {i+1}: {episode_rewards:.4f}")
                
                agents_performance['DQN'] = {
                    'mean_reward': np.mean(dqn_rewards),
                    'std_reward': np.std(dqn_rewards),
                    'max_reward': np.max(dqn_rewards),
                    'rewards': dqn_rewards
                }
                
                self.rl_agents['DQN'] = dqn_agent
                print(f"      ✅ DQN Final: {np.mean(dqn_rewards):.4f} ± {np.std(dqn_rewards):.4f}")
                
            except Exception as e:
                print(f"      ❌ DQN evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Similar for PPO...
        if ppo_agent is not None:
            print("   Final PPO evaluation...")
            try:
                ppo_rewards = []
                n_eval_episodes = 20
                
                # Check if vectorized
                is_vec_env = hasattr(self.rl_env, 'num_envs')
                
                for i in range(n_eval_episodes):
                    obs = self.rl_env.reset()
                    episode_rewards = 0.0
                    done_flags = np.zeros(self.rl_env.num_envs if is_vec_env else 1, dtype=bool)
                    
                    for step in range(1000):  # Max steps per episode
                        action, _ = ppo_agent.predict(obs, deterministic=True)
                        obs, reward, done, _ = self.rl_env.step(action)
                        
                        # Handle rewards
                        if is_vec_env:
                            episode_rewards += np.sum(reward[~done_flags])
                            done_flags = np.logical_or(done_flags, done)
                            
                            if np.all(done_flags):
                                break
                        else:
                            # Single environment - ensure scalar
                            reward_scalar = float(reward) if hasattr(reward, '__float__') else reward
                            done_scalar = bool(done) if hasattr(done, '__bool__') else done
                            
                            episode_rewards += reward_scalar
                            if done_scalar:
                                break
                    
                    # Average reward across environments if vectorized
                    if is_vec_env:
                        episode_rewards /= self.rl_env.num_envs
                    
                    ppo_rewards.append(episode_rewards)
                    
                    if i % 5 == 0:
                        print(f"      PPO Episode {i+1}: {episode_rewards:.4f}")
                
                agents_performance['PPO'] = {
                    'mean_reward': np.mean(ppo_rewards),
                    'std_reward': np.std(ppo_rewards),
                    'max_reward': np.max(ppo_rewards),
                    'rewards': ppo_rewards
                }
                
                self.rl_agents['PPO'] = ppo_agent
                print(f"      ✅ PPO Final: {np.mean(ppo_rewards):.4f} ± {np.std(ppo_rewards):.4f}")
                
            except Exception as e:
                print(f"      ❌ PPO evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Find best agent
        if agents_performance:
            best_agent = max(agents_performance.keys(), key=lambda k: agents_performance[k]['mean_reward'])
            best_performance = agents_performance[best_agent]['mean_reward']
            print(f"\n🏆 Best Agent: {best_agent} with mean reward: {best_performance:.4f}")

            self._plot_rl_performance(agents_performance)

        # 🔥 ADD THIS HERE - Close environment to save logs
        # Handle both single and vectorized environments
        if hasattr(self.rl_env, 'log_dir'):
            log_dir = self.rl_env.log_dir
        elif hasattr(self, 'rl_log_dir'):
            log_dir = self.rl_log_dir
        else:
            log_dir = self.log_dir

        print(f"💾 Saving training logs to: {log_dir}")
        # Close environment properly (handles both single and vectorized)
        try:
            if hasattr(self.rl_env, 'close'):
                self.rl_env.close()
                print(f"✅ Environment logs saved")
        except Exception as e:
            print(f"⚠️  Could not close environment cleanly: {e}")

        return agents_performance
            
    def generate_comprehensive_analysis(self):
        """Generate all visualization and analysis plots"""
        print("📊 Generating comprehensive analysis...")
        
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
        
        print("✅ Analysis generation completed!")
    
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
            print("⚠️  No feature importance data available. Training a model to generate it...")
            
            # Train a quick model to get feature importance
            try:
                if hasattr(self, 'processed_data') and self.processed_data:
                    if 'multi_objective_score' in self.processed_data['target_names']:
                        self.train_supervised_models('multi_objective_score')
                    else:
                        self.train_supervised_models(self.processed_data['target_names'][0])
                else:
                    print("❌ Cannot generate feature importance - no processed data available")
                    return
            except Exception as e:
                print(f"❌ Failed to generate feature importance: {e}")
                return
            
            # Check again
            if not self.feature_importance:
                print("❌ Still no feature importance after model training")
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
        
        print(f"âœ… Sclerostin feature importance plot saved")
    
    def _plot_model_comparison(self, performance_results, target_metric):
        """Plot model performance comparison as separate plots"""
        # Add this at the beginning of the method
        plt.figure(figsize=(12, 8))  # Set explicit, reasonable figure size

        models = list(performance_results.keys())
        r2_scores = [performance_results[model]['test_r2'] for model in models]
        mse_scores = [performance_results[model]['test_mse'] for model in models]
        
        # R² comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.ylabel('R² Score')
        plt.title(f'Model Comparison - R² Score ({target_metric})')
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
            plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center')
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
        
        print(f"📊 RL performance plots saved separately:")
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
            print("âš ï¸  No processed data available")
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
        
        plt.title('Sclerostin Biosensor: Input-Output Correlation Matrix\n(13 Features × 4 Targets)', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Output Targets (Y)', fontsize=12, fontweight='bold')
        plt.ylabel('Input Features (X)', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "sclerostin_input_output_correlation.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"âœ… Sclerostin correlation matrix saved")


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
                    plt.title(f'{model_name} - {target_metric}\nR² = {perf["test_r2"]:.3f}')
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

        print(f"✅ Results summary saved to {summary_path}")
    
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
    
    def save_best_models_only(self):
        """Save only the best models with proper handling for different model types"""
        print("💾 Saving best models only...")
        
        best_models_dir = self.output_dir / "best_models"
        os.makedirs(best_models_dir, exist_ok=True)
        
        for metric, model in self.surrogate_models.items():
            model_path = best_models_dir / f"best_{metric}_model.pkl"
            
            # Handle different model types
            if hasattr(model, 'state_dict'):  # PyTorch model
                torch_path = best_models_dir / f"best_{metric}_model.pth"
                torch.save(model.state_dict(), torch_path)
                print(f"   ✅ Saved PyTorch model: {torch_path}")
            else:  # sklearn model
                import joblib
                joblib.dump(model, model_path)
                print(f"   ✅ Saved sklearn model: {model_path}")
        
        print(f"✅ Best models saved to: {best_models_dir}")
    
    def run_complete_pipeline(self, biomarker: Optional[str] = None, 
                             apply_pca: bool = False, 
                             rl_timesteps: int = 500000):         # STRATEGIC DOCUMENT: Section IV.3
        """
        Run the complete pipeline from data loading to analysis
        
        Args:
            biomarker: Specific biomarker to focus on
            apply_pca: Whether to apply PCA
            rl_timesteps: Number of RL training timesteps
        """
        print("🎯 Starting Complete Biosensor Pipeline")
        print("=" * 50)
        
        try:
            # Stage 1: Data Preprocessing
            print("\n📊 STAGE 1: Data Preprocessing & Feature Selection")
            self.load_data(biomarker)
            preprocessing_info = self.preprocess_data(apply_pca=apply_pca)

            self.plot_target_metrics_distributions(self.plots_dir / 'target_metric_plots')
            self.plot_target_metrics_correlation_matrix(self.plots_dir / 'target_metric_plots')
            self.plot_target_metrics_summary(self.plots_dir / 'target_metric_plots')
            
            # Stage 2: Supervised Learning
            '''print("\n🤖 STAGE 2: Supervised Learning Models")
            for target_metric in self.processed_data['target_names'][:3]:  # Train on first 3 targets
                self.train_supervised_models(target_metric)'''
            
            # Stage 3: Surrogate Modeling
            print("\n🎭 STAGE 3: Surrogate Modeling")
            surrogate_performance = self.train_surrogate_models()
            
            # Stage 4: Reinforcement Learning
            print("\n🎮 STAGE 4: Reinforcement Learning")
            self.setup_rl_environment(self.target_metrics)
            rl_performance = self.train_rl_agents(rl_timesteps)

            # Ensure we have feature importance by training at least one supervised model
            if not self.feature_importance:
                print("\n🤖 STAGE 2: Training Supervised Models for Feature Importance")
                try:
                    # Train on multi-objective score first
                    if 'multi_objective_score' in self.processed_data['target_names']:
                        self.train_supervised_models('multi_objective_score')
                    else:
                        # Fall back to first available target
                        self.train_supervised_models(self.processed_data['target_names'][0])
                except Exception as e:
                    print(f"   ⚠️ Supervised model training failed: {e}")
            
            # Stage 5: Sclerostin-Specific Analysis
            print("\nSTAGE 5: Sclerostin Biosensor Analysis")
            self.plot_feature_importance_for_sclerostin()
            self.plot_sclerostin_correlation_matrix()
            print("âœ… Sclerostin-specific visualizations completed")
            
            # Save results and best models only
            print("\n💾 Saving Results & Best Models")
            self.save_best_models_only()
            self.save_results_summary()
            
            print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📁 All results saved to: {self.output_dir}")
            print(f"📊 Plots saved to: {self.plots_dir}")
            print(f"🤖 Models saved to: {self.models_dir}")
            print(f"📋 Results saved to: {self.results_dir}")
            
            return {
                'preprocessing_info': preprocessing_info,
                'model_performance': self.model_performance,
                'surrogate_performance': surrogate_performance,
                'rl_performance': rl_performance
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run the biosensor pipeline"""

    # Use context manager for automatic logging
    with DualLogger(log_dir="logs"):
        print("🧬 Synthetic Biology Biosensor Design Pipeline")
        print("=" * 60)
        
        # Configuration for NEW SCLEROSTIN DATASET
        data_path = r"sclerostin_biosensor_dataset.csv"  
        base_output_dir = "sclerostin_biosensor_results"
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
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
            
        print(f"\n🎯 Automated Biomarker Analysis for {len(biomarkers)} biomarkers:")
        for i, biomarker in enumerate(biomarkers, 1):
            print(f"   {i}. {biomarker}")

        rl_timesteps = 500000  # STRATEGIC DOCUMENT MINIMUM (Section IV.3)          ------------------------------------------------ NEED TO CHANGE
        
        # Results storage
        all_results = {}
        successful_analyses = []
        failed_analyses = []
        
        print(f"\n🚀 Starting automated analysis...")
        print(f"📊 RL Timesteps: {rl_timesteps}")
        print("=" * 60)
        
        # Run analysis for each biomarker
        for i, biomarker in enumerate(biomarkers, 1):
            print(f"\n[{i}/{len(biomarkers)}] 🧬 Processing: {biomarker}")
            print("-" * 40)
            
            # Create biomarker-specific output directory
            biomarker_output_dir = f"{base_output_dir}_{biomarker}"
            
            try:
                # Initialize pipeline for this biomarker
                pipeline = BiosensorPipeline(data_path, biomarker_output_dir)
                
                # Run complete pipeline
                results = pipeline.run_complete_pipeline(
                    biomarker=biomarker,
                    rl_timesteps=rl_timesteps
                )
                
                if results:
                    all_results[biomarker] = results
                    successful_analyses.append(biomarker)
                    print(f"✅ {biomarker} analysis completed successfully!")
                    
                    # Print quick summary
                    if 'model_performance' in results:
                        print("   🤖 Best Model Performance:")
                        for target, models in results['model_performance'].items():
                            best_model = max(models.items(), key=lambda x: x[1]['test_r2'])
                            print(f"      {target}: {best_model[0]} (R² = {best_model[1]['test_r2']:.3f})")
                    
                    if 'rl_performance' in results and results['rl_performance']:
                        print("   🎮 RL Agent Performance:")
                        for agent, perf in results['rl_performance'].items():
                            print(f"      {agent}: {perf['mean_reward']:.3f} ± {perf['std_reward']:.3f}")
                else:
                    failed_analyses.append(biomarker)
                    print(f"❌ {biomarker} analysis failed!")
                    
            except Exception as e:
                failed_analyses.append(biomarker)
                print(f"❌ {biomarker} analysis failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            print(f"📁 Results saved to: {biomarker_output_dir}")
            print("-" * 40)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 AUTOMATED ANALYSIS COMPLETE!")
        print("=" * 60)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   Total biomarkers: {len(biomarkers)}")
        print(f"   Successful: {len(successful_analyses)}")
        print(f"   Failed: {len(failed_analyses)}")
        
        if successful_analyses:
            print(f"\n✅ Successfully analyzed:")
            for biomarker in successful_analyses:
                print(f"   - {biomarker}")
        
        if failed_analyses:
            print(f"\n❌ Failed analyses:")
            for biomarker in failed_analyses:
                print(f"   - {biomarker}")
        
        # Generate comparative summary
        if len(successful_analyses) > 1:
            print(f"\n📈 COMPARATIVE ANALYSIS:")
            print("-" * 40)
            
            # Compare best model performance across biomarkers
            print("🤖 Best Model Performance Comparison:")
            for biomarker in successful_analyses:
                if biomarker in all_results and 'model_performance' in all_results[biomarker]:
                    best_overall = 0
                    best_model_name = ""
                    for target, models in all_results[biomarker]['model_performance'].items():
                        best_model = max(models.items(), key=lambda x: x[1]['test_r2'])
                        if best_model[1]['test_r2'] > best_overall:
                            best_overall = best_model[1]['test_r2']
                            best_model_name = best_model[0]
                    print(f"   {biomarker}: {best_model_name} (R² = {best_overall:.3f})")
            
            # Compare RL performance across biomarkers
            print(f"\n🎮 RL Performance Comparison:")
            for biomarker in successful_analyses:
                if biomarker in all_results and 'rl_performance' in all_results[biomarker]:
                    rl_results = all_results[biomarker]['rl_performance']
                    if rl_results:
                        best_rl_agent = max(rl_results.items(), key=lambda x: x[1]['mean_reward'])
                        print(f"   {biomarker}: {best_rl_agent[0]} ({best_rl_agent[1]['mean_reward']:.3f} ± {best_rl_agent[1]['std_reward']:.3f})")
        
        print(f"\n📁 All results saved in respective directories:")
        for biomarker in successful_analyses:
            print(f"   - {base_output_dir}_{biomarker}/")
        
        print("\n🎯 Analysis complete! Check individual directories for detailed results.")

# Example usage and testing
if __name__ == "__main__":
    main()