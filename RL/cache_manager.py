#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cache management for RL training pipeline
Handles caching of processed data, surrogates, and trained models
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib


class CacheManager:
    """Manages cache for pipeline artifacts"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_cache_hash(self, key: str, **kwargs) -> str:
        """Generate cache hash from key and parameters"""
        data = f"{key}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def has_cached(self, key: str, **kwargs) -> bool:
        """Check if cache exists"""
        cache_hash = self.get_cache_hash(key, **kwargs)
        cache_path = self.cache_dir / f"{key}_{cache_hash}.pkl"
        return cache_path.exists()

    def get_cached(self, key: str, **kwargs) -> Optional[Any]:
        """Get cached object"""
        cache_hash = self.get_cache_hash(key, **kwargs)
        cache_path = self.cache_dir / f"{key}_{cache_hash}.pkl"

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache from {cache_path}: {e}")
            return None

    def save_cache(self, key: str, obj: Any, **kwargs):
        """Save object to cache"""
        cache_hash = self.get_cache_hash(key, **kwargs)
        cache_path = self.cache_dir / f"{key}_{cache_hash}.pkl"

        with open(cache_path, 'wb') as f:
            pickle.dump(obj, f)

        # Update metadata
        self.metadata[key] = {
            'hash': cache_hash,
            'path': str(cache_path),
            'params': kwargs
        }
        self._save_metadata()

    def clear_cache(self):
        """Clear all cache"""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        self.metadata = {}
        self._save_metadata()

    def list_cached(self) -> Dict:
        """List all cached items"""
        return self.metadata
