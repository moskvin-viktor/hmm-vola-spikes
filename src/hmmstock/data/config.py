from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict


class DateFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: str | None = None
    end: str | None = None


class MarketProxyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_market_vola_proxy: str
    type: str = "raw"
    smoothing_window: int = 5


class VolatilityProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str = "zscore"
    normalize: bool = True


class DataConfig(BaseModel):
    """A validated, immutable order describing what data to fetch and prepare."""

    model_config = ConfigDict(frozen=True)

    tickers: list[str]
    period: str
    interval: str
    volatility_windows: list[int]
    market_proxy_processing: MarketProxyConfig
    date_filter: DateFilter = DateFilter()
    volatility_processing: VolatilityProcessingConfig = VolatilityProcessingConfig()

    @property
    def all_tickers(self) -> list[str]:
        return sorted(
            set(self.tickers) | {self.market_proxy_processing.default_market_vola_proxy}
        )

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "DataConfig":
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))
