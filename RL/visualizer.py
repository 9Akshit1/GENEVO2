#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization module for RL training pipeline
Generates plots for dataset analysis and surrogate quality assessment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Generate visualizations for RL pipeline"""

    def __init__(self, output_dir: Path, logger_obj=None):
        self.output_dir = Path(output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger_obj or logger

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_dataset_overview(self, data: dict):
        """Plot dataset statistics"""
        self.logger.info("Generating dataset overview plots...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')

        # Detection rate distribution
        axes[0, 0].hist(data['detection_rate'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Detection Rate')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Detection Rate Distribution')
        axes[0, 0].axvline(data['detection_rate'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()

        # FNR distribution
        axes[0, 1].hist(data['fnr'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('False Negative Rate')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('FNR Distribution')
        axes[0, 1].axvline(data['fnr'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()

        # TTD distribution
        axes[0, 2].hist(data['ttd'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].set_xlabel('Time to Detection (s)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('TTD Distribution')
        axes[0, 2].axvline(data['ttd'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 2].legend()

        # SNR distribution
        axes[1, 0].hist(data['snr'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('SNR (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('SNR Distribution')

        # SNR vs Detection Rate
        axes[1, 1].scatter(data['snr'], data['detection_rate'], alpha=0.5, s=20)
        axes[1, 1].set_xlabel('SNR (dB)')
        axes[1, 1].set_ylabel('Detection Rate')
        axes[1, 1].set_title('SNR vs Detection Rate')

        # SNR vs TTD
        axes[1, 2].scatter(data['snr'], data['ttd'], alpha=0.5, s=20, color='green')
        axes[1, 2].set_xlabel('SNR (dB)')
        axes[1, 2].set_ylabel('TTD (s)')
        axes[1, 2].set_title('SNR vs TTD')

        plt.tight_layout()
        plot_path = self.output_dir / "01_dataset_overview.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"  Saved: {plot_path.name}")
        plt.close()

    def plot_surrogate_quality(self, surrogates: dict, X: np.ndarray, y_dict: dict):
        """Plot surrogate model prediction accuracy"""
        self.logger.info("Generating surrogate quality plots...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Surrogate Model Quality', fontsize=14, fontweight='bold')

        metrics = ['detection_rate', 'fnr', 'ttd']
        colors = ['blue', 'orange', 'green']

        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            if metric not in surrogates or metric not in y_dict:
                continue

            model = surrogates[metric]
            y_actual = y_dict[metric]
            y_pred = model.predict(X)

            # Scatter plot
            axes[idx].scatter(y_actual, y_pred, alpha=0.5, s=20, color=color)

            # Perfect prediction line
            lims = [
                np.min([y_actual.min(), y_pred.min()]),
                np.max([y_actual.max(), y_pred.max()]),
            ]
            axes[idx].plot(lims, lims, 'r--', lw=2, label='Perfect')

            # Metrics
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

            axes[idx].set_xlabel('Actual')
            axes[idx].set_ylabel('Predicted')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
            axes[idx].legend()

        plt.tight_layout()
        plot_path = self.output_dir / "02_surrogate_quality.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"  Saved: {plot_path.name}")
        plt.close()

    def plot_metric_correlations(self, data: dict):
        """Plot correlations between metrics"""
        self.logger.info("Generating metric correlation plots...")

        # Prepare correlation data
        corr_data = np.column_stack([
            data['detection_rate'],
            data['fnr'],
            data['ttd'],
            data['snr'],
        ])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Metric Correlations', fontsize=14, fontweight='bold')

        # Correlation matrix heatmap
        corr_matrix = np.corrcoef(corr_data.T)
        im = axes[0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_xticks(range(4))
        axes[0].set_yticks(range(4))
        labels = ['DR', 'FNR', 'TTD', 'SNR']
        axes[0].set_xticklabels(labels)
        axes[0].set_yticklabels(labels)
        axes[0].set_title('Correlation Matrix')

        # Add correlation values
        for i in range(4):
            for j in range(4):
                text = axes[0].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=10)

        plt.colorbar(im, ax=axes[0])

        # DR vs FNR scatter with SNR coloring
        scatter = axes[1].scatter(data['detection_rate'], data['fnr'],
                                  c=data['snr'], cmap='viridis', s=30, alpha=0.6)
        axes[1].set_xlabel('Detection Rate')
        axes[1].set_ylabel('FNR')
        axes[1].set_title('Detection Rate vs FNR (colored by SNR)')
        cbar = plt.colorbar(scatter, ax=axes[1])
        cbar.set_label('SNR (dB)')

        plt.tight_layout()
        plot_path = self.output_dir / "03_metric_correlations.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"  Saved: {plot_path.name}")
        plt.close()

    def plot_training_summary(self, training_info: dict):
        """Plot training summary statistics"""
        self.logger.info("Generating training summary plots...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Summary', fontsize=14, fontweight='bold')

        # Training configuration
        ax = axes[0, 0]
        ax.axis('off')
        config_text = f"""
Training Configuration:
  Total Episodes: {training_info.get('total_episodes', 'N/A')}
  Total Timesteps: {training_info.get('total_timesteps', 'N/A'):,}
  Training Time: {training_info.get('training_time', 0):.1f} seconds
  Speed: {training_info.get('fps', 0):.0f} FPS

Environment:
  Observation Space: 4D (SNR, Scenario, Biosensor, Noise)
  Action Space: 3D (Kd, Sensitivity, Threshold adjustments)
  Max Episode Steps: 2048
        """
        ax.text(0.1, 0.5, config_text, fontsize=10, family='monospace',
               verticalalignment='center')

        # Surrogate metrics
        ax = axes[0, 1]
        ax.axis('off')
        surr_text = f"""
Surrogate Models:
  Detection Rate:
    R² = {training_info.get('dr_r2', 'N/A')}
  FNR:
    R² = {training_info.get('fnr_r2', 'N/A')}
  TTD:
    R² = {training_info.get('ttd_r2', 'N/A')}

Reward Composition:
  45% Detection Rate
  20% FNR Minimization
  25% TTD Minimization
  10% Action Penalty
        """
        ax.text(0.1, 0.5, surr_text, fontsize=10, family='monospace',
               verticalalignment='center')

        # Dataset statistics
        ax = axes[1, 0]
        ax.axis('off')
        data_text = f"""
Dataset:
  Samples: {training_info.get('n_samples', 'N/A')}
  Detection Rate: {training_info.get('dr_mean', 0):.4f} ± {training_info.get('dr_std', 0):.4f}
  FNR: {training_info.get('fnr_mean', 0):.4f} ± {training_info.get('fnr_std', 0):.4f}
  TTD: {training_info.get('ttd_mean', 0):.1f} ± {training_info.get('ttd_std', 0):.1f} s
  SNR Range: [{training_info.get('snr_min', 0):.1f}, {training_info.get('snr_max', 0):.1f}] dB
        """
        ax.text(0.1, 0.5, data_text, fontsize=10, family='monospace',
               verticalalignment='center')

        # Hardware/runtime info
        ax = axes[1, 1]
        ax.axis('off')
        runtime_text = f"""
System Info:
  Device: Auto (GPU if available)
  PPO Algorithm Parameters:
    Learning Rate: 3e-4
    Discount (γ): 0.99
    GAE Lambda: 0.95
    PPO Clip Range: 0.2
    Batch Size: 64
    Epochs per Update: 10
        """
        ax.text(0.1, 0.5, runtime_text, fontsize=10, family='monospace',
               verticalalignment='center')

        plt.tight_layout()
        plot_path = self.output_dir / "04_training_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"  Saved: {plot_path.name}")
        plt.close()
