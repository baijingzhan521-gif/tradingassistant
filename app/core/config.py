from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Trading Analysis Assistant"
    app_version: str = "0.1.0"
    default_exchange: str = "binance"
    default_market_type: str = "perpetual"
    default_strategy_profile: str = "swing_trend_long_regime_gate_v1"
    default_lookback: int = 300
    database_url: str = "sqlite:///./trading_assistant.db"
    strategy_config_dir: str = str(BASE_DIR / "config" / "strategies")
    log_level: str = "INFO"
    volatility_low_atr_pct: float = 0.6
    volatility_high_atr_pct: float = 2.0
    ccxt_timeout_ms: int = 15000
    ccxt_max_retries: int = 2
    ccxt_retry_delay_ms: int = 750
    ccxt_trust_env: bool = True
    ccxt_http_proxy: str | None = None
    ccxt_https_proxy: str | None = None
    ccxt_socks_proxy: str | None = None
    bybit_public_linear_ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    binance_use_sandbox: bool = False
    binance_hostname: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
