from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.analysis import AnalysisResult


class NotificationService(ABC):
    @abstractmethod
    def send_analysis(self, analysis: AnalysisResult) -> None:
        raise NotImplementedError
