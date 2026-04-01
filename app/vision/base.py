from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VisionAdapter(ABC):
    @abstractmethod
    def analyze_chart_images(self, image_paths: list[Path]) -> dict[str, Any]:
        raise NotImplementedError
