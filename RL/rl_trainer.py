#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL agent training module with fixed metrics tracking and episode detection
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys
from typing import Dict, List
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import BiosensorOptimizationEnv
from logging_config import RLTrainingMonitor

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Thread-safe metrics tracking for vectorized environments"""

    def __init__(self, window_size: int = 100):
        """Initialize tracker"""
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.total_episodes_completed = 0

    def add_episode(self, episode_return: float, episode_length: int):
        """Track a completed episode"""
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.total_episodes_completed += 1

    def get_stats(self) -> Dict:
        """Get current statistics"""
        if not self.episode_returns:
            return {
                'total_episodes': 0,
                'mean_return': 0.0,
                'max_return': 0.0,
                'min_return': 0.0,
                'mean_length': 0,
                'all_returns': []
            }

        return {
            'total_episodes': self.total_episodes_completed,
            'mean_return': float(np.mean(self.episode_returns)),
            'max_return': float(np.max(self.episode_returns)),
            'min_return': float(np.min(self.episode_returns)),
            'median_return': float(np.median(self.episode_returns)),
            'std_return': float(np.std(self.episode_returns)),
            'mean_length': float(np.mean(self.episode_lengths)),
            'all_returns': list(self.episode_returns)
        }


class ComprehensiveProgressCallback(BaseCallback):
    """Callback with comprehensive metrics tracking and logging"""

    def __init__(self,
                 log_interval: int = 100,
                 total_episodes: int = 50,
                 steps_per_episode: int = 2048,
                 n_envs: int = 1,
                 logger_obj=None):
        super().__init__(verbose=0)
        self.log_interval = log_interval
        self.total_episodes = total_episodes
        self.steps_per_episode = steps_per_episode
        self.n_envs = n_envs
        self.app_logger = logger_obj or logger
        self.start_time = datetime.now()
        self.last_log_timesteps = 0

        # Metrics tracking
        self.metrics_tracker = MetricsTracker(window_size=200)

        # Track episodes by monitoring episode_num increments
        self.last_episode_num = {}

        # Diagnostic tracking
        self.episode_detections_per_env = {}
        self.total_episode_detections = 0
        self.diagnostic_initialized = False

    def _on_step(self) -> bool:
        """Called at each training step"""

        # Detect episode completions by monitoring episode_num increments
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
            for env_idx, single_env in enumerate(self.model.env.envs):
                # Initialize tracking
                if env_idx not in self.last_episode_num:
                    self.last_episode_num[env_idx] = single_env.episode_num
                    self.episode_detections_per_env[env_idx] = 0

                # Check if episode_num has incremented (indicates reset was called)
                current_episode_num = single_env.episode_num
                if current_episode_num > self.last_episode_num[env_idx]:
                    # Episode was completed and reset
                    reward = single_env.last_episode_return
                    length = single_env.last_episode_length
                    self.metrics_tracker.add_episode(reward, length)

                    # Diagnostic tracking
                    self.episode_detections_per_env[env_idx] += 1
                    self.total_episode_detections += 1

                    # Log detection with environment info
                    self.app_logger.debug(
                        f"[EPISODE_DETECTED] Env {env_idx} | "
                        f"Episode {current_episode_num} | "
                        f"Return: {reward:.4f} | "
                        f"Length: {length} | "
                        f"Total detections env{env_idx}: {self.episode_detections_per_env[env_idx]}"
                    )
                    self.last_episode_num[env_idx] = current_episode_num

        # Log progress periodically
        if self.num_timesteps >= self.last_log_timesteps + self.log_interval:
            self._log_progress()
            self.last_log_timesteps = self.num_timesteps

        return True

    def _log_progress(self):
        """Log training progress with comprehensive metrics"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0

        # Actual episode counting
        actual_episodes = self.metrics_tracker.total_episodes_completed
        progress_pct = 100.0 * actual_episodes / self.total_episodes if self.total_episodes > 0 else 0

        # Get statistics
        stats = self.metrics_tracker.get_stats()

        # Format progress message
        msg = (
            f"[Episode {actual_episodes:3d}/{self.total_episodes}] ({progress_pct:5.1f}%) | "
            f"Timesteps: {self.num_timesteps:7d} | "
            f"FPS: {fps:6.0f} | "
            f"Time: {elapsed_time:6.1f}s"
        )

        if stats['total_episodes'] > 0:
            msg += (
                f" | Mean Return: {stats['mean_return']:7.4f} | "
                f"Max Return: {stats['max_return']:7.4f}"
            )

        # Add diagnostic info about per-environment episodes
        if self.episode_detections_per_env:
            env_episode_counts = ", ".join(
                [f"Env{i}={self.episode_detections_per_env.get(i, 0)}"
                 for i in range(self.n_envs)]
            )
            msg += f" | [{env_episode_counts}]"

        print(msg, flush=True)
        sys.stdout.flush()
        self.app_logger.info(msg)

    def get_metrics(self) -> Dict:
        """Get final metrics"""
        metrics = self.metrics_tracker.get_stats()
        # Add diagnostic per-environment data
        metrics['episode_detections_per_env'] = self.episode_detections_per_env
        metrics['total_episode_detections'] = self.total_episode_detections
        return metrics


class EpisodeMetricsCallback(BaseCallback):
    """Callback to log episode metrics to monitoring system"""

    def __init__(self, monitor: RLTrainingMonitor, verbose: int = 0):
        super().__init__(verbose)
        self.monitor = monitor
        self.app_logger = logger
        self.last_episode_num = {}

    def _on_step(self) -> bool:
        """Called at each training step"""
        # Detect episode completions by monitoring episode_num increments
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
            for env_idx, single_env in enumerate(self.model.env.envs):
                # Initialize tracking
                if env_idx not in self.last_episode_num:
                    self.last_episode_num[env_idx] = single_env.episode_num

                # Check if episode_num has incremented
                current_episode_num = single_env.episode_num
                if current_episode_num > self.last_episode_num[env_idx]:
                    # Episode was completed and reset
                    reward = single_env.last_episode_return
                    length = single_env.last_episode_length
                    # Log to monitoring system
                    self.monitor.log_episode_stats(
                        episode=current_episode_num,
                        reward=reward,
                        length=length
                    )
                    self.last_episode_num[env_idx] = current_episode_num

        return True


class RLTrainer:
    """Train RL agent for biosensor parameter optimization"""

    def __init__(self, logger_obj=None):
        self.logger = logger_obj or logger
        self.model = None
        self.training_time = 0.0

    def train_agent(self, surrogates: Dict, data: np.ndarray, output_dir: Path,
                   total_episodes: int = 50, steps_per_episode: int = 2048,
                   n_envs: int = 4, surrogate_feature_scaler=None) -> Dict:
        """
        Train PPO agent with comprehensive metrics tracking.

        Args:
            surrogates: Dict of trained surrogate models
            data: Feature matrix for RL observations
            output_dir: Output directory for checkpoints/logs
            total_episodes: Number of training episodes
            steps_per_episode: Steps per episode (n_steps for PPO)
            n_envs: Number of parallel environments (default 4)
            surrogate_feature_scaler: Scaler for surrogate features
        """

        self.logger.info("=" * 80)
        self.logger.info("RL AGENT TRAINING")
        self.logger.info("=" * 80)

        # Create vectorized environments
        def make_env() -> BiosensorOptimizationEnv:
            return BiosensorOptimizationEnv(
                surrogates, data, self.logger,
                surrogate_feature_scaler=surrogate_feature_scaler
            )

        env = DummyVecEnv([make_env for _ in range(n_envs)])
        self.logger.info(f"Vectorized environment created (DummyVecEnv)")
        self.logger.info(f"  Number of parallel environments: {n_envs}")
        single_env = make_env()
        self.logger.info(f"  Observation space: {single_env.observation_space}")
        self.logger.info(f"  Action space: {single_env.action_space}")
        single_env.close()

        # Setup checkpoints
        checkpoint_dir = output_dir / "models" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=steps_per_episode,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_agent",
            save_replay_buffer=False,
        )

        # Create monitoring system
        monitor = RLTrainingMonitor(output_dir / "rl_logs", self.logger)

        # Progress callback with proper metrics tracking
        progress_callback = ComprehensiveProgressCallback(
            log_interval=100,
            total_episodes=total_episodes,
            steps_per_episode=steps_per_episode,
            n_envs=n_envs,
            logger_obj=self.logger
        )

        # Episode metrics callback
        episode_metrics_callback = EpisodeMetricsCallback(
            monitor=monitor,
            verbose=0
        )

        # PPO hyperparameters
        self.logger.info("\nPPO Hyperparameters:")
        self.logger.info(f"  Algorithm: PPO2 (Proximal Policy Optimization)")
        self.logger.info(f"  Policy: MlpPolicy")
        self.logger.info(f"  Learning rate: 3e-4")
        self.logger.info(f"  n_steps: {steps_per_episode}")
        self.logger.info(f"  batch_size: 64")
        self.logger.info(f"  n_epochs: 10")
        self.logger.info(f"  gamma: 0.99")
        self.logger.info(f"  gae_lambda: 0.95")

        # Create model
        tensorboard_log = None
        try:
            import tensorboard
            tensorboard_log = str(output_dir / "rl_logs")
        except ImportError:
            self.logger.warning("TensorBoard not installed")

        self.model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=steps_per_episode,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=0,
            tensorboard_log=tensorboard_log,
            device='auto',
        )

        self.logger.info(f"\nTraining Configuration:")
        self.logger.info(f"  Total episodes to train: {total_episodes}")
        self.logger.info(f"  Steps per episode: {steps_per_episode}")
        self.logger.info(f"  Parallel environments: {n_envs}")
        self.logger.info(f"  Total timesteps: {total_episodes * steps_per_episode * n_envs}")

        # Train
        start_time = datetime.now()
        total_timesteps = total_episodes * steps_per_episode * n_envs

        self.logger.info(f"\nStarting training...")
        self.logger.info(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, progress_callback, episode_metrics_callback],
                log_interval=1,
            )
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            raise

        # Training complete
        training_time = (datetime.now() - start_time).total_seconds()
        fps = total_timesteps / training_time

        # Get final metrics from callback
        final_metrics = progress_callback.get_metrics()

        # Save metrics
        monitor.save_metrics()
        learning_summary = monitor.get_learning_summary()

        # Report results
        self.logger.info(f"\n" + "=" * 80)
        self.logger.info(f"TRAINING COMPLETE")
        self.logger.info(f"=" * 80)
        self.logger.info(f"  Total episodes (detected): {final_metrics['total_episodes']}")
        self.logger.info(f"  Duration: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        self.logger.info(f"  Speed: {fps:.0f} FPS")

        # Diagnostic: per-environment episode counts
        if 'episode_detections_per_env' in final_metrics:
            self.logger.info(f"\n  Episode Detections Per Environment:")
            for env_idx in range(n_envs):
                count = final_metrics['episode_detections_per_env'].get(env_idx, 0)
                self.logger.info(f"    Env {env_idx}: {count} episodes completed")
            self.logger.info(f"  Total detections across all envs: {final_metrics.get('total_episode_detections', 0)}")
        self.logger.info(f"")
        self.logger.info(f"  Episode Returns:")
        self.logger.info(f"    Mean: {final_metrics['mean_return']:.6f}")
        self.logger.info(f"    Max: {final_metrics['max_return']:.6f}")
        self.logger.info(f"    Min: {final_metrics['min_return']:.6f}")
        self.logger.info(f"    Std: {final_metrics['std_return']:.6f}")
        self.logger.info(f"    Median: {final_metrics['median_return']:.6f}")
        self.logger.info(f"  Mean Episode Length: {final_metrics['mean_length']:.0f} steps")

        if learning_summary:
            self.logger.info(f"\n  Learning Dynamics:")
            self.logger.info(f"    Final reward: {learning_summary.get('final_reward', 0):.6f}")
            self.logger.info(f"    Reward trend: {learning_summary.get('reward_trend', 'unknown')}")

        # Evaluate training results
        try:
            from training_evaluator import TrainingEvaluator
            evaluator = TrainingEvaluator()
            results_dict = {
                'total_timesteps': total_timesteps,
                'total_episodes': final_metrics['total_episodes'],
                'training_time': training_time,
                'fps': fps,
                'mean_episode_return': final_metrics['mean_return'],
                'max_episode_return': final_metrics['max_return'],
                'min_episode_return': final_metrics['min_return'],
                'std_return': final_metrics['std_return'],
                'mean_length': final_metrics['mean_length'],
                'episode_returns': final_metrics['all_returns'],
                'learning_summary': learning_summary,
            }
            evaluation = evaluator.evaluate(results_dict)
            evaluator.print_recommendations()
        except Exception as e:
            self.logger.warning(f"Could not run training evaluator: {e}")

        # Save model
        model_path = output_dir / "models" / "ppo_agent"
        self.model.save(str(model_path))
        self.logger.info(f"\n  Agent saved to: {model_path}.zip")

        self.training_time = training_time
        env.close()

        return {
            'total_timesteps': total_timesteps,
            'total_episodes': final_metrics['total_episodes'],
            'training_time': training_time,
            'fps': fps,
            'mean_episode_return': final_metrics['mean_return'],
            'max_episode_return': final_metrics['max_return'],
            'min_episode_return': final_metrics['min_return'],
            'std_return': final_metrics['std_return'],
            'mean_length': final_metrics['mean_length'],
            'episode_returns': final_metrics['all_returns'],
            'learning_summary': learning_summary,
        }

    def get_model(self):
        """Get trained model"""
        return self.model

    def get_training_time(self) -> float:
        """Get training duration in seconds"""
        return self.training_time
