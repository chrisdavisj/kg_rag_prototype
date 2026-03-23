import yaml
from pathlib import Path
from errors import ConfigError

_config_cache = None
_runtime_overrides = {}


def load_config(path: str = "config.yaml") -> dict:
    global _config_cache
    if _config_cache is None:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(
                f"Config file must contain a top-level mapping: {config_path}"
            )

        _config_cache = loaded
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


def get_required(key_path: str):
    value = get(key_path)
    if value is None:
        raise ConfigError(f"Missing required config value: {key_path}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.startswith("<") and stripped.endswith(">"):
            raise ConfigError(f"Invalid placeholder config value for: {key_path}")
    return value
