#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RL environment for biosensor parameter optimization
Gymnasium-compatible environment with proper reward shaping
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class BiosensorOptimizationEnv(gym.Env):
    """
    RL environment for optimizing biosensor parameters.

    **DESIGN**: Proper Markovian MDP where actions causally affect state and reward.

    State: Current parameter setting [SNR, scenario, biosensor_type, noise_preset]
    Action: Adjustments to SNR and design parameters [delta_snr, delta_biosensor, delta_noise]
    Reward: Surrogate predictions based on current parameter setting

    Observation: Normalized parameter state (what the agent sees)
    Next State: Parameters modified by action, clipped to valid ranges
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, surrogates: dict, data: np.ndarray, logger_obj=None,
                 surrogate_feature_scaler=None):
        """
        Initialize environment.

        Args:
            surrogates: Dict with 'detection_rate', 'fnr', 'ttd' models
            data: Raw feature matrix [SNR, scenario_encoded, biosensor_encoded, noise_encoded]
            logger_obj: Logger instance
            surrogate_feature_scaler: StandardScaler fitted on surrogate features (SNR, biosensor, noise)
        """
        super().__init__()

        self.surrogates = surrogates
        self.data = data  # Raw data for sampling initial states
        self.surrogate_feature_scaler = surrogate_feature_scaler
        self.logger = logger_obj or logger

        # Observation space: [SNR, scenario_encoded, biosensor_encoded, noise_encoded]
        # Normalized via z-score based on data statistics
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(4,), dtype=np.float32
        )

        # Action space: adjustments to design parameters [delta_snr, delta_biosensor, delta_noise]
        # NOTE: scenario is observable but not actionable (fixed by experiment design)
        # We only adjust SNR, biosensor, and noise
        # [-0.5, 0.5] allows 50% adjustment per step for exploration
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(3,), dtype=np.float32
        )

        # Compute data statistics for clipping (bounds)
        self.data_min = self.data.min(axis=0)
        self.data_max = self.data.max(axis=0)
        self.data_mean = self.data.mean(axis=0)
        self.data_std = self.data.std(axis=0) + 1e-8  # Avoid div by zero

        self.current_step = 0
        self.max_steps = 2048
        self.episode_return = 0.0
        self.episode_rewards = []
        self.episode_num = 0
        self.episodes_completed = 0

        # Store last completed episode stats for callback access
        self.last_episode_return = 0.0
        self.last_episode_length = 0

        # Current state: parameter values (NOT normalized)
        self.state = None

        # Diagnostic tracking for activity monitoring
        self.total_steps = 0
        self.step_actions_sum = np.array([0.0, 0.0, 0.0])  # Track action magnitudes

    def reset(self, seed=None, options=None):
        """Reset environment to random initial parameter setting.

        For reproducibility, always seed with the provided seed.
        This ensures deterministic behavior across parallel environments
        when seeded consistently.
        """
        super().reset(seed=seed)

        # Track completed episodes and save stats
        is_episode_complete = False
        if self.current_step > 0:
            is_episode_complete = True
            self.episodes_completed += 1
            # Store stats before reset so callbacks can access them
            self.last_episode_return = self.episode_return
            self.last_episode_length = self.current_step

        self.current_step = 0
        self.episode_return = 0.0
        self.episode_rewards = []
        old_episode_num = self.episode_num
        self.episode_num += 1

        # Sample random initial parameter setting from dataset
        # Use seeded RNG for reproducibility (super().reset() already set self.np_random)
        idx = self.np_random.integers(0, len(self.data))
        self.state = self.data[idx].astype(np.float32).copy()

        # Diagnostic logging (will be captured by callback system)
        if is_episode_complete:
            self.logger.debug(
                f"[ENV_RESET] Episode {old_episode_num} complete → "
                f"Episode {self.episode_num} starting | "
                f"Episode length: {self.last_episode_length} | "
                f"Episode return: {self.last_episode_return:.4f}"
            )

        return self._normalize_observation(self.state), {}

    def step(self, action):
        """
        Execute one environment step.

        **PROPER MARKOVIAN DYNAMICS**:
        1. Current state is [SNR, scenario, biosensor, noise]
        2. Action adjusts [SNR, biosensor, noise] (scenario is fixed, not controlled)
        3. Next observation is normalized state'
        4. Reward depends on state' (surrogate predictions)

        Args:
            action: [delta_snr, delta_biosensor, delta_noise] ([-0.5, 0.5])
                   3-dimensional (no action on scenario - it's observable but fixed)

        Returns:
            obs_next: Normalized next state (observation for agent)
            reward: Scalar reward based on surrogate predictions
            terminated: True if episode exceeded max_steps
            truncated: False (no timeout besides max_steps)
            info: Episode stats if terminated
        """
        self.current_step += 1
        self.total_steps += 1

        # Apply action: update state parameters (except scenario which is fixed)
        # State layout: [SNR, scenario, biosensor, noise]
        # Action layout: [delta_snr, delta_biosensor, delta_noise]
        state_deltas = np.array([action[0], 0.0, action[1], action[2]], dtype=np.float32)
        self.state = self.state + state_deltas

        # Diagnostic: track action magnitudes for activity monitoring
        self.step_actions_sum += np.array([abs(action[0]), abs(action[1]), abs(action[2])])

        # Clip to valid data ranges (keep parameters within observed data bounds)
        self.state = np.clip(self.state, self.data_min, self.data_max)

        # Compute reward using surrogate features only (SNR, biosensor, noise)
        # NOTE: Exclude scenario to match how surrogates were trained
        surrogate_features = np.array([self.state[0], self.state[2], self.state[3]], dtype=np.float32)
        reward = self._compute_reward(surrogate_features.reshape(1, -1), action)

        self.episode_return += reward
        self.episode_rewards.append(reward)

        # Termination condition
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Info dict for logging
        info = {}
        if terminated:
            # Episode is complete - provide stats to callback
            info['episode'] = {
                'r': float(self.episode_return),
                'l': self.current_step,
                'mean_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                'max_reward': float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0,
            }

        # Return normalized observation
        obs_next = self._normalize_observation(self.state)

        return obs_next, reward, terminated, truncated, info

    def _normalize_observation(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state for neural network input.
        Uses z-score normalization based on actual data statistics.
        """
        # data_mean and data_std are computed from raw data passed to environment
        normalized = ((state - self.data_mean) / self.data_std).astype(np.float32)
        return np.clip(normalized, -3.0, 3.0)  # Clip to valid observation space

    def _compute_reward(self, features: np.ndarray, action: np.ndarray) -> float:
        """
        Compute reward from surrogates using proper reward composition.

        Features must be RAW (unscaled). Scaling is applied here to match
        surrogate training data.

        Reward composition:
        - 50% Detection rate maximization
        - 25% False negative rate minimization
        - 25% Time to detection minimization
        """
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Scale features to match surrogate training
            if self.surrogate_feature_scaler is not None:
                features_scaled = self.surrogate_feature_scaler.transform(features)
            else:
                # Fallback: use raw features (not ideal but works)
                features_scaled = features

            reward = 0.0

            # 1. Detection rate (maximize) - 50% weight
            if 'detection_rate' in self.surrogates:
                dr_pred = np.clip(self.surrogates['detection_rate'].predict(features_scaled)[0], 0, 1)
                reward += 0.50 * float(dr_pred)

            # 2. FNR minimization (minimize) - 25% weight
            if 'fnr' in self.surrogates:
                fnr_pred = np.clip(self.surrogates['fnr'].predict(features_scaled)[0], 0, 1)
                reward += 0.25 * (1.0 - float(fnr_pred))

            # 3. TTD minimization (minimize) - 25% weight
            if 'ttd' in self.surrogates:
                ttd_pred = np.clip(self.surrogates['ttd'].predict(features_scaled)[0], 0, 10000)
                ttd_reward = max(0.0, 1.0 - float(ttd_pred) / 5000.0)
                reward += 0.25 * ttd_reward

            return float(reward)

        except Exception as e:
            self.logger.warning(f"Reward computation error: {e}")
            return 0.0

    def render(self):
        """Render environment (no-op)"""
        pass

    def get_episode_stats(self) -> dict:
        """Get current episode statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'episode_return': float(self.episode_return),
            'mean_reward': float(np.mean(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
        }
