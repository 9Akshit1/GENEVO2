#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production-grade logging configuration for RL pipeline.

COMPREHENSIVE LOGGING ARCHITECTURE:
- Single unified log file capturing ALL output (prints, logging, warnings, exceptions, tracebacks)
- Thread-safe queue handler for parallel environment logging
- sys.stdout/stderr redirection to capture direct prints
- Warning redirection to logging system
- Exception hook for uncaught exceptions
- Clean separation between console output (human-readable) and file output (complete transcript)
"""

import logging
import logging.handlers
import queue
import sys
import io
import warnings
from pathlib import Path
from typing import Optional, Dict, TextIO
import json
from datetime import datetime


class TeeStream(io.TextIOBase):
    """Redirect print output to both original stream and logging system."""

    def __init__(self, original_stream: TextIO, logger: logging.Logger, level: int = logging.INFO):
        self.original_stream = original_stream
        self.logger = logger
        self.level = level
        self.linebuffer = ""

    def write(self, message: str) -> int:
        self.original_stream.write(message)
        self.original_stream.flush()

        # Buffer lines and log complete lines
        self.linebuffer += message
        while '\n' in self.linebuffer:
            line, self.linebuffer = self.linebuffer.split('\n', 1)
            if line.strip():  # Skip empty lines
                self.logger.log(self.level, f"[stdout] {line}")

        return len(message)

    def flush(self):
        self.original_stream.flush()
        # Log any remaining buffered content
        if self.linebuffer.strip():
            self.logger.log(self.level, f"[stdout] {self.linebuffer}")
            self.linebuffer = ""

    def isatty(self):
        return self.original_stream.isatty()


class WarningHandler(logging.Handler):
    """Custom handler to ensure warnings go through logging system."""

    def emit(self, record):
        # Already logged by the logging system, just ensure it appears
        pass


def configure_logging(
    log_dir: Path,
    log_name: str = "pipeline",
    verbose: bool = False,
    suppress_third_party: bool = True,
    capture_prints: bool = True
) -> logging.Logger:
    """
    Configure production-grade logging system with complete output capture.

    Args:
        log_dir: Directory for log files
        log_name: Base name for log files (adds timestamp)
        verbose: If True, set app logger to DEBUG; else INFO
        suppress_third_party: If True, set third-party libs to WARNING
        capture_prints: If True, redirect print() to logging system

    Returns:
        Configured logger for application use
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # =====================================================================
    # Create timestamp for all log files
    # =====================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # =====================================================================
    # MAIN LOG FILE: Single unified file with ALL output
    # =====================================================================
    main_log_file = log_dir / f"{log_name}_{timestamp}.log"

    # Use QueueHandler for thread-safe logging from parallel environments
    log_queue = queue.Queue(-1)  # Unlimited size
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(queue_handler)

    # Listener thread to write queued messages to file
    file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Detailed format for file (includes module name, line number)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()

    # =====================================================================
    # CONSOLE OUTPUT: Clean, human-readable app logs only
    # =====================================================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)

    console_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Filter console to app modules only (cleaner display)
    app_modules = [
        '__main__',
        'data_processor',
        'surrogate_trainer',
        'rl_trainer',
        'rl_environment',
        'rl_pipeline',
        'cache_manager',
        'visualizer',
        'logging_config',
    ]

    class AppFilter(logging.Filter):
        def filter(self, record):
            module = record.name.split('.')[0]
            return any(module.startswith(m) for m in app_modules)

    console_handler.addFilter(AppFilter())
    root_logger.addHandler(console_handler)

    # =====================================================================
    # SUPPRESS THIRD-PARTY LIBRARIES (but still log them to file)
    # =====================================================================
    if suppress_third_party:
        third_party_libs = [
            'matplotlib', 'PIL', 'sklearn', 'joblib', 'numpy',
            'pandas', 'torch', 'gym', 'gymnasium', 'stable_baselines3'
        ]
        for lib in third_party_libs:
            logging.getLogger(lib).setLevel(logging.WARNING)

    # =====================================================================
    # CAPTURE PYTHON WARNINGS
    # =====================================================================
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        logger = logging.getLogger('warnings')
        logger.warning(f"{category.__name__}: {message} ({filename}:{lineno})")

    warnings.showwarning = warning_on_one_line

    # =====================================================================
    # CAPTURE UNCAUGHT EXCEPTIONS
    # =====================================================================
    original_excepthook = sys.excepthook

    def logging_excepthook(exc_type, exc_value, exc_traceback):
        error_logger = logging.getLogger('exceptions')
        error_logger.error(
            f"Uncaught exception: {exc_type.__name__}: {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = logging_excepthook

    # =====================================================================
    # CAPTURE PRINTS (optional)
    # =====================================================================
    if capture_prints:
        print_logger = logging.getLogger('prints')
        sys.stdout = TeeStream(sys.stdout, print_logger, logging.INFO)
        sys.stderr = TeeStream(sys.stderr, print_logger, logging.WARNING)

    # =====================================================================
    # GET APPLICATION LOGGER
    # =====================================================================
    app_logger = logging.getLogger('rl_pipeline')

    # Initial log messages
    app_logger.info("=" * 80)
    app_logger.info("COMPREHENSIVE LOGGING SYSTEM INITIALIZED")
    app_logger.info("=" * 80)
    app_logger.info(f"Main log file: {main_log_file}")
    app_logger.info(f"Console level: {'DEBUG' if verbose else 'INFO'}")
    app_logger.info(f"File level: DEBUG (all output captured)")
    app_logger.info(f"Prints captured: {capture_prints}")
    app_logger.info(f"Warnings captured: YES")
    app_logger.info(f"Exceptions captured: YES")
    app_logger.info(f"Thread-safe queue: YES (for parallel environments)")
    app_logger.info("")

    return app_logger


class MetricsLogger:
    """Structured logging for training metrics (CSV + JSON)"""

    def __init__(self, log_dir: Path):
        """Initialize metrics logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []

    def log_episode(self, episode: int, episode_data: Dict) -> None:
        """Log episode metrics."""
        record = {'episode': episode, **episode_data}
        self.metrics_history.append(record)

    def log_step(self, step: int, step_data: Dict) -> None:
        """Log training step metrics."""
        record = {'step': step, **step_data}
        self.metrics_history.append(record)

    def save_csv(self, filename: str = "metrics.csv") -> Path:
        """Save metrics to CSV file."""
        if not self.metrics_history:
            return None

        import csv
        csv_path = self.log_dir / filename

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            all_keys = set()
            for record in self.metrics_history:
                all_keys.update(record.keys())
            all_keys = sorted(list(all_keys))

            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self.metrics_history)

        return csv_path

    def save_json(self, filename: str = "metrics.json") -> Path:
        """Save metrics to JSON file."""
        if not self.metrics_history:
            return None

        import numpy as np

        def serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")

        json_path = self.log_dir / filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, default=serialize)

        return json_path


class RLTrainingMonitor:
    """Monitor and log RL training dynamics with thread-safe metrics collection."""

    def __init__(self, log_dir: Path, logger: logging.Logger):
        """Initialize RL training monitor."""
        self.log_dir = Path(log_dir)
        self.logger = logger
        self.metrics = MetricsLogger(log_dir)
        self.episode_rewards = []
        self.episode_lengths = []

    def log_episode_stats(self, episode: int, reward: float, length: int, info: Dict = None) -> None:
        """Log episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        # Compute rolling statistics
        window = min(10, len(self.episode_rewards))
        mean_reward = sum(self.episode_rewards[-window:]) / window
        max_reward = max(self.episode_rewards[-max(1, len(self.episode_rewards) - 10):])

        # Log to metrics
        record = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'mean_reward_10': mean_reward,
            'max_reward_10': max_reward,
            'mean_length_10': sum(self.episode_lengths[-window:]) / window,
        }

        if info:
            record.update(info)

        self.metrics.log_episode(episode, record)

        # Log to console/file
        self.logger.info(
            f"[Episode {episode:3d}] Reward: {reward:7.4f} | "
            f"Mean(10): {mean_reward:7.4f} | Length: {length:5d}"
        )

    def log_ppo_stats(self, step: int, stats: Dict) -> None:
        """Log PPO training statistics."""
        record = {'step': step, **stats}
        self.metrics.log_step(step, record)

        if step % 1000 == 0:
            msg = f"[Step {step:6d}] "
            for key, value in stats.items():
                if isinstance(value, float):
                    msg += f"{key}: {value:8.5f} | "
            self.logger.info(msg.rstrip(' | '))

    def save_metrics(self) -> None:
        """Save all collected metrics to CSV and JSON."""
        csv_path = self.metrics.save_csv('episode_metrics.csv')
        json_path = self.metrics.save_json('episode_metrics.json')

        if csv_path:
            self.logger.info(f"Episode metrics saved to: {csv_path}")
        if json_path:
            self.logger.info(f"Episode metrics saved to: {json_path}")

    def get_learning_summary(self) -> Dict:
        """Get summary of learning dynamics."""
        if not self.episode_rewards:
            return {}

        return {
            'total_episodes': len(self.episode_rewards),
            'final_reward': self.episode_rewards[-1],
            'mean_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'max_reward': max(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'mean_length': sum(self.episode_lengths) / len(self.episode_lengths),
            'reward_trend': 'improving' if self.episode_rewards[-1] > self.episode_rewards[0] else 'declining',
        }
