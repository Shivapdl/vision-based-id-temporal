"""
TARA shared utilities.

Import this at the top of every notebook instead of hardcoding paths:

    from src.config import load_config, get_paths
    cfg = load_config()                # reads configs/default.yaml
    paths = get_paths(cfg)             # resolved absolute paths

This keeps the notebooks portable across machines and lets users
override settings in one place.
"""

from pathlib import Path
import yaml

# Project root = parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_name: str = "default.yaml") -> dict:
    """Load a YAML config from configs/."""
    config_path = PROJECT_ROOT / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Expected a YAML file in {PROJECT_ROOT / 'configs'}"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_paths(cfg: dict) -> dict:
    """Resolve config paths to absolute paths and create them if missing."""
    paths = {}
    for key, rel in cfg["paths"].items():
        p = (PROJECT_ROOT / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths
