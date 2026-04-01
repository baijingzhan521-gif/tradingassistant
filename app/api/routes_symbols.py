from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_symbol_exchange_factory
from app.core.config import get_settings
from app.data.exchange_client import ExchangeClientFactory
from app.schemas.common import MarketType
from app.schemas.response import SymbolsResponse


router = APIRouter(tags=["symbols"])
settings = get_settings()


@router.get("/symbols", response_model=SymbolsResponse)
def list_symbols(
    exchange: str = Query(default=settings.default_exchange),
    market_type: MarketType = Query(default=MarketType(settings.default_market_type)),
    limit: int = Query(default=200, ge=1, le=500),
    exchange_factory: ExchangeClientFactory = Depends(get_symbol_exchange_factory),
) -> SymbolsResponse:
    symbols = exchange_factory.list_symbols(exchange.strip().lower(), market_type.value, limit=limit)
    return SymbolsResponse(
        exchange=exchange.strip().lower(),
        market_type=market_type.value,
        count=len(symbols),
        symbols=symbols,
    )
