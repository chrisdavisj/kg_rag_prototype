import yaml
from pathlib import Path

_config_cache = None
_runtime_overrides = {}


def load_config(path: str = "config.yaml") -> dict:
    global _config_cache
    if _config_cache is None:
        with open(Path(path), "r") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get(key_path: str, default=None):
    """Access nested config using dot notation, e.g., get('thresholds.preferred_confidence')"""
    keys = key_path.split(".")

    # Check runtime overrides first
    cfg = _runtime_overrides
    for key in keys:
        if not isinstance(cfg, dict):
            break
        if key in cfg:
            cfg = cfg[key]
        else:
            cfg = None
            break
    if cfg is not None:
        return cfg

     # Fallback to static config
    cfg = load_config()
    for key in keys:
        if not isinstance(cfg, dict):
            return default
        cfg = cfg.get(key)
    return cfg if cfg is not None else default


def put(key_path: str, value):
    """Set a runtime-only config value."""
    keys = key_path.split(".")
    cfg = _runtime_overrides
    for key in keys[:-1]:
        cfg = cfg.setdefault(key, {})
    cfg[keys[-1]] = value
