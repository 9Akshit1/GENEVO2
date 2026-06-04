#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Baseline Diagnostic

Runs 20 episodes with random actions only (no learning).
This establishes a baseline for what PPO needs to beat.

If PPO returns >> Random, RL is working.
If PPO returns ≈ Random, RL is not providing value.
If Random >> PPO, something is seriously wrong.
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


class RandomBaselineRunner:
    """Run episodes with random actions"""

    def __init__(self, surrogates, data, surrogate_scaler, logger):
        self.surrogates = surrogates
        self.data = data
        self.surrogate_scaler = surrogate_scaler
        self.logger = logger

    def run_random_episodes(self, n_episodes: int = 20, seed: int = 42) -> dict:
        """
        Run random baseline episodes.

        Args:
            n_episodes: Number of episodes to run
            seed: Random seed for reproducibility

        Returns:
            results: Dict with episode statistics
        """
        np.random.seed(seed)

        self.logger.info(f"\nRunning {n_episodes} random baseline episodes...")
        self.logger.info("=" * 60)

        # Create environment
        env = BiosensorOptimizationEnv(
            self.surrogates,
            self.data,
            self.logger,
            surrogate_feature_scaler=self.surrogate_scaler
        )

        episode_returns = []
        episode_lengths = []
        episode_details = []

        for episode_num in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode_num)
            episode_return = 0.0
            episode_step = 0

            step_rewards = []
            step_actions = []

            done = False
            while not done:
                # Random action
                action = env.action_space.sample()
                step_actions.append(action.copy())

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                step_rewards.append(reward)
                episode_return += reward
                episode_step += 1
                done = terminated or truncated

            episode_returns.append(episode_return)
            episode_lengths.append(episode_step)

            episode_details.append({
                'episode': episode_num + 1,
                'return': float(episode_return),
                'length': episode_step,
                'mean_reward': float(np.mean(step_rewards)) if step_rewards else 0.0,
                'max_reward': float(np.max(step_rewards)) if step_rewards else 0.0,
                'min_reward': float(np.min(step_rewards)) if step_rewards else 0.0,
                'mean_action_magnitude': float(np.mean([np.linalg.norm(a) for a in step_actions])),
            })

            if (episode_num + 1) % 5 == 0:
                self.logger.info(f"  Episode {episode_num + 1}/{n_episodes} | "
                               f"Return: {episode_return:.2f} | "
                               f"Length: {episode_step}")

        env.close()

        # Compute statistics
        returns = np.array(episode_returns)
        results = {
            'n_episodes': n_episodes,
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'episode_details': episode_details,
            'statistics': {
                'mean_return': float(np.mean(returns)),
                'median_return': float(np.median(returns)),
                'max_return': float(np.max(returns)),
                'min_return': float(np.min(returns)),
                'std_return': float(np.std(returns)),
                'mean_length': float(np.mean(episode_lengths)),
            }
        }

        return results

    def plot_comparison(self, random_results: dict, ppo_results: dict, output_dir: Path):
        """Plot comparison between random and PPO"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Random Baseline vs PPO Comparison', fontsize=14)

        random_returns = random_results['episode_returns']
        ppo_returns = ppo_results['episode_returns'] if isinstance(ppo_results['episode_returns'], list) \
                      else ppo_results['all_returns']

        # Plot 1: Distribution comparison
        ax = axes[0, 0]
        ax.hist(random_returns, bins=10, alpha=0.6, label='Random', edgecolor='black')
        ax.hist(ppo_returns, bins=10, alpha=0.6, label='PPO', edgecolor='black')
        ax.set_xlabel('Episode Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Return Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Episode returns over time
        ax = axes[0, 1]
        ax.plot(random_returns, marker='o', label='Random', linewidth=2, alpha=0.7)
        ax.plot(ppo_returns, marker='s', label='PPO', linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title('Returns Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Statistics comparison
        ax = axes[1, 0]
        ax.axis('off')

        random_stats = random_results['statistics']
        ppo_stats = ppo_results['statistics'] if 'statistics' in ppo_results \
                   else {
                       'mean_return': np.mean(ppo_returns),
                       'median_return': np.median(ppo_returns),
                       'max_return': np.max(ppo_returns),
                       'min_return': np.min(ppo_returns),
                       'std_return': np.std(ppo_returns),
                   }

        improvement = (ppo_stats['mean_return'] - random_stats['mean_return']) / random_stats['mean_return'] * 100

        stats_text = f"""
Baseline Comparison:

Random Baseline:
  Mean: {random_stats['mean_return']:.2f}
  Median: {random_stats['median_return']:.2f}
  Std: {random_stats['std_return']:.2f}
  Range: [{random_stats['min_return']:.2f}, {random_stats['max_return']:.2f}]

PPO Performance:
  Mean: {ppo_stats['mean_return']:.2f}
  Median: {ppo_stats['median_return']:.2f}
  Std: {ppo_stats['std_return']:.2f}
  Range: [{ppo_stats['min_return']:.2f}, {ppo_stats['max_return']:.2f}]

Improvement:
  {improvement:+.1f}%
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')

        # Plot 4: Cumulative performance
        ax = axes[1, 1]
        random_cumsum = np.cumsum(random_returns) / np.arange(1, len(random_returns) + 1)
        ppo_cumsum = np.cumsum(ppo_returns) / np.arange(1, len(ppo_returns) + 1)

        ax.plot(random_cumsum, marker='o', label='Random', linewidth=2, alpha=0.7)
        ax.plot(ppo_cumsum, marker='s', label='PPO', linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Mean Return')
        ax.set_title('Learning Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / "random_vs_ppo_comparison.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Comparison plot saved: {output_file}")
        plt.close()


def run_random_baseline_diagnostic(data_dir: Path, output_dir: Path,
                                   ppo_diagnostic_results: dict = None):
    """Run random baseline and compare to PPO"""

    output_dir = Path(output_dir)
    log_dir = output_dir / "diagnostic_logs"
    plot_dir = output_dir / "diagnostic_plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(log_dir, "random_baseline", verbose=True)

    logger.info("=" * 80)
    logger.info("RANDOM BASELINE DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info(f"Runs 20 episodes with random actions")
    logger.info(f"Compares to PPO performance")
    logger.info(f"If PPO >> Random: RL is learning")
    logger.info(f"If PPO ≈ Random: RL is not helping")

    try:
        # Load data and train surrogates
        logger.info("\nLoading data and training surrogates...")
        processor = DataProcessor(logger)
        df = processor.load_master_index(data_dir)
        processor.validate_data(df, processed=False)
        df = processor.process_data(df)
        processor.validate_data(df, processed=True)

        X_rl, _ = processor.prepare_rl_features(df)
        X_surr, y_dr, y_fnr, y_ttd = processor.prepare_surrogate_features(df)

        trainer = SurrogateTrainer(logger)
        surr_metrics = trainer.train_all_surrogates(X_surr, y_dr, y_fnr, y_ttd)
        surrogates = trainer.get_models()

        encoders = processor.get_encoders() if hasattr(processor, 'get_encoders') else {}
        surrogate_scaler = encoders.get('surrogate_feature_scaler', None)

        # Run random baseline
        logger.info("\n" + "=" * 80)
        logger.info("RANDOM BASELINE RUN")
        logger.info("=" * 80)

        runner = RandomBaselineRunner(surrogates, X_rl, surrogate_scaler, logger)
        random_results = runner.run_random_episodes(n_episodes=20, seed=42)

        # Load PPO results from diagnostic (if available)
        ppo_results_file = log_dir / "ppo_diagnostic_results.json"
        ppo_results = None
        if ppo_results_file.exists():
            with open(ppo_results_file, 'r') as f:
                ppo_results = json.load(f)
        elif ppo_diagnostic_results:
            ppo_results = ppo_diagnostic_results
        else:
            # Try to load from recent diagnostic run
            logger.warning("Could not find PPO results for comparison")
            logger.warning("Please provide PPO results or run the diagnostic with saved data")

        # Report results
        logger.info("\n" + "=" * 80)
        logger.info("RANDOM BASELINE SUMMARY")
        logger.info("=" * 80)

        stats = random_results['statistics']
        logger.info(f"\nRandom Episodes (n={random_results['n_episodes']}):")
        logger.info(f"  Mean return: {stats['mean_return']:.4f}")
        logger.info(f"  Median return: {stats['median_return']:.4f}")
        logger.info(f"  Std return: {stats['std_return']:.4f}")
        logger.info(f"  Min return: {stats['min_return']:.4f}")
        logger.info(f"  Max return: {stats['max_return']:.4f}")
        logger.info(f"  Mean length: {stats['mean_length']:.0f}")

        if ppo_results:
            logger.info("\n" + "=" * 80)
            logger.info("PPO vs RANDOM COMPARISON")
            logger.info("=" * 80)

            ppo_mean = ppo_results.get('mean_return', np.mean(ppo_results.get('episode_returns', [])))
            random_mean = stats['mean_return']
            improvement = (ppo_mean - random_mean) / random_mean * 100

            logger.info(f"\nBaseline Performance:")
            logger.info(f"  Random mean: {random_mean:.4f}")
            logger.info(f"  PPO mean: {ppo_mean:.4f}")
            logger.info(f"  Improvement: {improvement:+.1f}%")

            if improvement > 20:
                logger.info(f"\n✓ PPO is significantly better than random (>{improvement:.0f}%)")
                logger.info(f"  RL learning appears to be working")
            elif improvement > 0:
                logger.warning(f"\n⚠️  PPO is only marginally better ({improvement:+.1f}%)")
                logger.warning(f"  RL learning may not be sufficiently effective")
            else:
                logger.error(f"\n❌ PPO is worse than random!")
                logger.error(f"  Something is seriously wrong with the RL setup")

            # Plot comparison
            try:
                runner.plot_comparison(random_results, ppo_results, plot_dir)
            except Exception as e:
                logger.warning(f"Could not create comparison plot: {e}")

        # Save results
        results_file = log_dir / "random_baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'random_results': random_results,
            }, f, indent=2)
        logger.info(f"\nResults saved: {results_file}")

        logger.info("\n" + "=" * 80)
        logger.info("ASSESSMENT & NEXT STEPS")
        logger.info("=" * 80)

        if ppo_results:
            improvement = (ppo_mean - random_mean) / random_mean * 100
            if improvement > 20:
                logger.info("\nRL appears to be learning meaningfully.")
                logger.info("Continue with Phase 3 diagnostics:")
                logger.info("  1. Reward decomposition (which components help?)")
                logger.info("  2. RL metrics (entropy, KL, value loss)")
                logger.info("  3. Full 50-episode training")
            elif improvement > 0:
                logger.info("\nRL is learning but weakly.")
                logger.info("Consider:")
                logger.info("  1. Check reward scaling")
                logger.info("  2. Review action sensitivity maps")
                logger.info("  3. Consider switching to Bayesian Optimization")
            else:
                logger.info("\nRL is not learning effectively.")
                logger.info("Strongly recommend switching to:")
                logger.info("  1. Bayesian Optimization")
                logger.info("  2. CMA-ES")
                logger.info("  3. Evolutionary strategies")
        else:
            logger.info("\nTo compare with PPO:")
            logger.info("1. Save PPO diagnostic results to: diagnostic_logs/ppo_diagnostic_results.json")
            logger.info("2. Re-run this script")
            logger.info("OR provide PPO results as input")

        return 0

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Random baseline diagnostic')
    parser.add_argument('--data-dir', type=Path, default=Path("data"))
    parser.add_argument('--output-dir', type=Path,
                       default=Path("RL/rl_results_diagnostic"))

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    sys.exit(run_random_baseline_diagnostic(args.data_dir, args.output_dir))
