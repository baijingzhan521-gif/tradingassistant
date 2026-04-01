from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from app.schemas.analysis import AnalysisResult
from app.schemas.request import AnalyzeRequest


class Strategy(ABC):
    name: str
    required_timeframes: tuple[str, ...]

    @abstractmethod
    def analyze(self, request: AnalyzeRequest, ohlcv_by_timeframe: dict[str, pd.DataFrame]) -> AnalysisResult:
        raise NotImplementedError
