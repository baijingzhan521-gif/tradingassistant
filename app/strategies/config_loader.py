from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml

from app.core.config import get_settings
from app.core.exceptions import StrategyConfigError


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class StrategyConfigLoader:
    def __init__(self, config_dir: Optional[str] = None) -> None:
        settings = get_settings()
        self.config_dir = Path(config_dir or settings.strategy_config_dir)

    def load(self, strategy_name: str, defaults: dict[str, Any]) -> dict[str, Any]:
        config_path = self.config_dir / f"{strategy_name}.yaml"
        if not config_path.exists():
            return deepcopy(defaults)

        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise StrategyConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

        if not isinstance(loaded, dict):
            raise StrategyConfigError(f"Strategy config {config_path} must be a YAML mapping")
        return _deep_merge(defaults, loaded)
