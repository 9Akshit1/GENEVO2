#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Action Sensitivity Map Diagnostic

Measures how much each action dimension affects the reward.
Tests whether the environment is actually responsive to actions or
if it's insensitive/noisy.

For each action dimension, sweeps from -0.5 to 0.5 and plots:
- Reward response
- Per-component rewards (DR, FNR, TTD)
- State changes
"""

import numpy as np
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from data_processor import DataProcessor
from surrogate_trainer import SurrogateTrainer
from rl_environment import BiosensorOptimizationEnv
from logging_config import configure_logging

import matplotlib.pyplot as plt


class ActionSensitivityAnalyzer:
    """Analyze how sensitive rewards are to each action dimension"""

    def __init__(self, surrogates, data, surrogate_scaler, logger):
        self.surrogates = surrogates
        self.data = data
        self.surrogate_scaler = surrogate_scaler
        self.logger = logger

    def sweep_action_dimension(self, action_idx: int, n_samples: int = 10,
                               n_test_states: int = 5):
        """
        Sweep a single action dimension and measure reward response.

        Args:
            action_idx: Which action to sweep [0, 1, 2]
            n_samples: Number of points to sample in [-0.5, 0.5]
            n_test_states: Number of random initial states to test

        Returns:
            results: Dict with sweep results
        """
        self.logger.info(f"\nAction Sensitivity: Dimension {action_idx}")
        self.logger.info("=" * 60)

        # Sample random initial states
        np.random.seed(42)  # Deterministic
        state_indices = np.random.choice(len(self.data), n_test_states, replace=False)
        test_states = self.data[state_indices].astype(np.float32).copy()

        # Sweep action values
        action_values = np.linspace(-0.5, 0.5, n_samples)

        results = {
            'action_idx': action_idx,
            'action_values': action_values.tolist(),
            'test_states': [],
            'sweep_results': []
        }

        for state_idx, initial_state in enumerate(test_states):
            self.logger.info(f"  Testing with initial state {state_idx + 1}/{n_test_states}")

            state_results = {
                'initial_state': initial_state.tolist(),
                'rewards': [],
                'dr_rewards': [],
                'fnr_rewards': [],
                'ttd_rewards': [],
                'state_changes': [],
            }

            for action_val in action_values:
                # Create action (only one dimension non-zero)
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                action[action_idx] = action_val

                # Apply action to state
                state_delta = np.array([action[0], 0.0, action[1], action[2]], dtype=np.float32)
                next_state = initial_state + state_delta

                # Clip to data bounds
                next_state = np.clip(next_state, self.data.min(axis=0), self.data.max(axis=0))

                # Compute surrogate features (SNR, biosensor, noise - exclude scenario)
                surrogate_features = np.array(
                    [next_state[0], next_state[2], next_state[3]], dtype=np.float32
                ).reshape(1, -1)

                # Compute rewards (component-wise)
                dr_reward, fnr_reward, ttd_reward, total_reward = self._compute_component_rewards(
                    surrogate_features
                )

                state_results['rewards'].append(float(total_reward))
                state_results['dr_rewards'].append(float(dr_reward))
                state_results['fnr_rewards'].append(float(fnr_reward))
                state_results['ttd_rewards'].append(float(ttd_reward))
                state_results['state_changes'].append((next_state - initial_state).tolist())

            results['sweep_results'].append(state_results)
            results['test_states'].append(initial_state.tolist())

        # Compute statistics
        all_rewards = []
        for sr in results['sweep_results']:
            all_rewards.extend(sr['rewards'])

        results['statistics'] = {
            'mean_reward': float(np.mean(all_rewards)),
            'max_reward': float(np.max(all_rewards)),
            'min_reward': float(np.min(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'reward_range': float(np.max(all_rewards) - np.min(all_rewards)),
        }

        self.logger.info(f"  Reward statistics:")
        self.logger.info(f"    Mean: {results['statistics']['mean_reward']:.4f}")
        self.logger.info(f"    Range: {results['statistics']['reward_range']:.4f}")
        self.logger.info(f"    Std: {results['statistics']['std_reward']:.4f}")

        # Assess sensitivity
        reward_range = results['statistics']['reward_range']
        if reward_range < 0.05:
            self.logger.warning(f"  ⚠️  LOW SENSITIVITY: Reward range {reward_range:.4f} (flat response)")
        elif reward_range > 0.3:
            self.logger.info(f"  ✓ GOOD SENSITIVITY: Reward range {reward_range:.4f} (responsive)")
        else:
            self.logger.info(f"  ~ MODERATE SENSITIVITY: Reward range {reward_range:.4f}")

        return results

    def _compute_component_rewards(self, features: np.ndarray):
        """Compute individual reward components"""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features
        if self.surrogate_scaler is not None:
            features_scaled = self.surrogate_scaler.transform(features)
        else:
            features_scaled = features

        dr_reward = 0.0
        fnr_reward = 0.0
        ttd_reward = 0.0

        if 'detection_rate' in self.surrogates:
            dr_pred = np.clip(self.surrogates['detection_rate'].predict(features_scaled)[0], 0, 1)
            dr_reward = 0.50 * float(dr_pred)

        if 'fnr' in self.surrogates:
            fnr_pred = np.clip(self.surrogates['fnr'].predict(features_scaled)[0], 0, 1)
            fnr_reward = 0.25 * (1.0 - float(fnr_pred))

        if 'ttd' in self.surrogates:
            ttd_pred = np.clip(self.surrogates['ttd'].predict(features_scaled)[0], 0, 10000)
            ttd_reward = 0.25 * max(0.0, 1.0 - float(ttd_pred) / 5000.0)

        total_reward = dr_reward + fnr_reward + ttd_reward
        return dr_reward, fnr_reward, ttd_reward, total_reward

    def plot_results(self, results: dict, output_dir: Path):
        """Plot action sensitivity results"""
        action_names = ['SNR', 'Biosensor', 'Noise']
        action_idx = results['action_idx']
        action_name = action_names[action_idx]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Action Sensitivity: {action_name} (Dimension {action_idx})', fontsize=14)

        action_values = np.array(results['action_values'])

        # Plot 1: Total reward
        ax = axes[0, 0]
        for state_idx, sr in enumerate(results['sweep_results']):
            ax.plot(action_values, sr['rewards'], marker='o', label=f'State {state_idx}', alpha=0.7)
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward Response')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Component rewards (averaged across states)
        ax = axes[0, 1]
        mean_dr = np.mean([sr['dr_rewards'] for sr in results['sweep_results']], axis=0)
        mean_fnr = np.mean([sr['fnr_rewards'] for sr in results['sweep_results']], axis=0)
        mean_ttd = np.mean([sr['ttd_rewards'] for sr in results['sweep_results']], axis=0)

        ax.plot(action_values, mean_dr, marker='o', label='DR (50%)', linewidth=2)
        ax.plot(action_values, mean_fnr, marker='s', label='FNR (25%)', linewidth=2)
        ax.plot(action_values, mean_ttd, marker='^', label='TTD (25%)', linewidth=2)
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Component Reward')
        ax.set_title('Reward Components (Average)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Smoothness assessment
        ax = axes[1, 0]
        for state_idx, sr in enumerate(results['sweep_results']):
            rewards = np.array(sr['rewards'])
            diffs = np.diff(rewards)
            ax.plot(action_values[1:], np.abs(diffs), marker='o', label=f'State {state_idx}', alpha=0.7)
        ax.set_xlabel('Action Value')
        ax.set_ylabel('|Reward Difference|')
        ax.set_title('Reward Smoothness (Abs Differences)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 4: Statistics
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
Reward Statistics:
  Mean: {results['statistics']['mean_reward']:.4f}
  Min: {results['statistics']['min_reward']:.4f}
  Max: {results['statistics']['max_reward']:.4f}
  Range: {results['statistics']['reward_range']:.4f}
  Std Dev: {results['statistics']['std_reward']:.4f}

Sensitivity Assessment:
  Range {results['statistics']['reward_range']:.4f}

  Flat (<0.05): Not learnable
  Moderate (0.05-0.3): Weak signal
  Good (>0.3): Learnable
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        output_file = output_dir / f"action_sensitivity_dim{action_idx}_{action_name}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        self.logger.info(f"  Plot saved: {output_file}")
        plt.close()


def run_action_sensitivity_diagnostic(data_dir: Path, output_dir: Path):
    """Run complete action sensitivity analysis"""

    output_dir = Path(output_dir)
    log_dir = output_dir / "diagnostic_logs"
    plot_dir = output_dir / "diagnostic_plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(log_dir, "action_sensitivity", verbose=True)

    logger.info("=" * 80)
    logger.info("ACTION SENSITIVITY DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info(f"Measures how reward responds to each action dimension")
    logger.info(f"Tests if environment is learnable or insensitive to actions")

    try:
        # Load data
        logger.info("\nLoading data...")
        processor = DataProcessor(logger)
        df = processor.load_master_index(data_dir)
        processor.validate_data(df, processed=False)
        df = processor.process_data(df)
        processor.validate_data(df, processed=True)

        X_rl, _ = processor.prepare_rl_features(df)
        X_surr, y_dr, y_fnr, y_ttd = processor.prepare_surrogate_features(df)

        # Train surrogates
        logger.info("Training surrogates...")
        trainer = SurrogateTrainer(logger)
        surr_metrics = trainer.train_all_surrogates(X_surr, y_dr, y_fnr, y_ttd)
        surrogates = trainer.get_models()

        encoders = processor.get_encoders() if hasattr(processor, 'get_encoders') else {}
        surrogate_scaler = encoders.get('surrogate_feature_scaler', None)

        # Create analyzer
        analyzer = ActionSensitivityAnalyzer(surrogates, X_rl, surrogate_scaler, logger)

        # Sweep each action dimension
        logger.info("\n" + "=" * 80)
        logger.info("SWEEPING ACTION DIMENSIONS")
        logger.info("=" * 80)

        all_results = {}
        for action_idx in range(3):
            results = analyzer.sweep_action_dimension(action_idx, n_samples=10, n_test_states=5)
            all_results[action_idx] = results
            analyzer.plot_results(results, plot_dir)

        # Summary report
        logger.info("\n" + "=" * 80)
        logger.info("SENSITIVITY SUMMARY")
        logger.info("=" * 80)

        action_names = ['SNR', 'Biosensor', 'Noise']
        for action_idx, results in all_results.items():
            action_name = action_names[action_idx]
            reward_range = results['statistics']['reward_range']

            logger.info(f"\n{action_name} (Dim {action_idx}):")
            logger.info(f"  Reward range: {reward_range:.4f}")
            logger.info(f"  Std dev: {results['statistics']['std_reward']:.4f}")

            if reward_range < 0.05:
                logger.warning(f"  Status: ⚠️  INSENSITIVE (flat reward)")
                logger.warning(f"         Agent cannot learn from this action")
            elif reward_range > 0.3:
                logger.info(f"  Status: ✓ SENSITIVE (learnable)")
            else:
                logger.info(f"  Status: ~ MODERATE (weak signal)")

        # Overall assessment
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL ASSESSMENT")
        logger.info("=" * 80)

        total_range = sum([r['statistics']['reward_range'] for r in all_results.values()])
        avg_range = total_range / len(all_results)

        logger.info(f"\nAverage reward range: {avg_range:.4f}")
        if avg_range < 0.05:
            logger.warning(f"\n⚠️  ENVIRONMENT IS INSENSITIVE TO ACTIONS")
            logger.warning(f"    Reward barely changes with any action")
            logger.warning(f"    RL will NOT be able to learn effectively")
            logger.warning(f"    Consider: Switch to random search or direct optimization")
        elif avg_range < 0.15:
            logger.warning(f"\n⚠️  ENVIRONMENT IS WEAKLY RESPONSIVE")
            logger.warning(f"    Reward signal is weak, learning will be slow")
            logger.warning(f"    May work with careful tuning, but not recommended")
        else:
            logger.info(f"\n✓ ENVIRONMENT IS RESPONSIVE")
            logger.info(f"  RL learning should be possible")

        # Save detailed results
        results_file = output_dir / "diagnostic_logs" / "action_sensitivity_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': {str(k): v for k, v in all_results.items()},
            }, f, indent=2)
        logger.info(f"\nDetailed results saved: {results_file}")

        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATION")
        logger.info("=" * 80)
        if avg_range < 0.15:
            logger.info("\nBased on action sensitivity:")
            logger.info("1. Run RANDOM BASELINE diagnostic next")
            logger.info("2. If PPO returns << Random, environment is not learnable")
            logger.info("3. Consider switching to Bayesian Optimization or CMA-ES")
        else:
            logger.info("\nEnvironment appears learnable. Continue with:")
            logger.info("1. RANDOM BASELINE diagnostic for comparison")
            logger.info("2. Reward decomposition analysis")
            logger.info("3. Full 50-episode training with improved diagnostics")

        return 0

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Action sensitivity diagnostic')
    parser.add_argument('--data-dir', type=Path, default=Path("data"))
    parser.add_argument('--output-dir', type=Path,
                       default=Path("RL/rl_results_diagnostic"))

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    sys.exit(run_action_sensitivity_diagnostic(args.data_dir, args.output_dir))
