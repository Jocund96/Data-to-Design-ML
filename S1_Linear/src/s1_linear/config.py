from pathlib import Path
import yaml


def load_config(config_path: str | Path = "configs/week03_linear.yaml") -> dict:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
