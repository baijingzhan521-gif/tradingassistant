from __future__ import annotations


class TradingAssistantError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UnsupportedExchangeError(TradingAssistantError):
    pass


class UnsupportedMarketTypeError(TradingAssistantError):
    pass


class UnsupportedTimeframeError(TradingAssistantError):
    pass


class SymbolNotFoundError(TradingAssistantError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=404)


class ExternalServiceError(TradingAssistantError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=502)


class StrategyConfigError(TradingAssistantError):
    pass


class PersistenceError(TradingAssistantError):
    def __init__(self, message: str) -> None:
        status_code = 404 if "not found" in message.lower() else 500
        super().__init__(message, status_code=status_code)
