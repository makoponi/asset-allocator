"""Configuration and constants for transatlantic equity portfolio allocation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

RegionBloc = Literal["NA", "EU"]

VALID_REGION_BLOCS: tuple[RegionBloc, ...] = ("NA", "EU")
BASE_CURRENCY: str = "EUR"
FINANCIALS_GICS_SECTOR: int = 40

DEFAULT_INDEX_TICKERS: dict[RegionBloc, tuple[str, ...]] = {
    "NA": ("SPX Index", "SPTSX60 Index"),
    "EU": ("SXXP Index",),
}

DEFAULT_HEDGE_FUTURES: dict[RegionBloc, tuple[str, ...]] = {
    "NA": ("ES", "MES"),
    "EU": ("FXXP", "FESX"),
}

CORE_FUNDAMENTAL_METRICS: tuple[str, ...] = (
    "PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "EV_TO_T12M_EBITDA",
    "FREE_CASH_FLOW_YIELD",
    "RETURN_COM_EQY",
    "OPER_MARGIN",
    "ASSET_TURNOVER",
    "TOT_DEBT_TO_TOT_EQY",
    "SALES_REV_TURN_GROWTH_1YR",
    "ACCRUALS_RATIO",
)

CORE_TECHNICAL_METRICS: tuple[str, ...] = (
    "MOM_1M",
    "MOM_12M_1M",
    "IDIO_VOL_252D",
    "BETA_REGIONAL_252D",
    "AMIHUD_ILLIQUIDITY_21D",
)

ALL_CORE_FEATURES: tuple[str, ...] = CORE_FUNDAMENTAL_METRICS + CORE_TECHNICAL_METRICS


class FeatureConfig(BaseModel):
    """Configuration for cross-sectional feature computation and normalization."""

    winsor_lower_quantile: float = Field(default=0.01, ge=0.0, le=0.10)
    winsor_upper_quantile: float = Field(default=0.99, ge=0.90, le=1.0)
    min_observations_for_zscore: int = Field(default=2, ge=1)
    min_history_days_beta: int = Field(default=63, ge=20)
    full_history_days_beta: int = Field(default=252, ge=63)
    momentum_reversal_window: int = Field(default=21, ge=5)
    momentum_trend_window: int = Field(default=252, ge=21)
    amihud_window: int = Field(default=21, ge=5)
    accounting_lag_days: int = Field(default=90, ge=0)


class StoragePaths(BaseModel):
    """File system paths for data storage."""

    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    raw_bql_dir: Path = Field(default_factory=lambda: Path("data/raw_bql_exports"))
    db_path: Path = Field(default_factory=lambda: Path("data/db/quant_store.db"))
    parquet_dir: Path = Field(default_factory=lambda: Path("data/parquet"))

    def resolve_paths(self, base: Path | None = None) -> "StoragePaths":
        """Resolve all paths relative to a specified base directory."""
        anchor = base or self.root_dir
        return StoragePaths(
            root_dir=anchor,
            data_dir=(anchor / self.data_dir).resolve(),
            raw_bql_dir=(anchor / self.raw_bql_dir).resolve(),
            db_path=(anchor / self.db_path).resolve(),
            parquet_dir=(anchor / self.parquet_dir).resolve(),
        )


class AllocationConstraints(BaseModel):
    """Allocation constraints for stocks and regional sleeves."""

    min_stock_weight: float = Field(default=0.005, ge=0.0, le=1.0)
    max_stock_weight: float = Field(default=0.035, ge=0.0, le=1.0)

    # Sleeve weights bounds: NA in [50%, 65%], EU in [35%, 50%]
    na_weight_min: float = Field(default=0.50, ge=0.0, le=1.0)
    na_weight_max: float = Field(default=0.65, ge=0.0, le=1.0)
    eu_weight_min: float = Field(default=0.35, ge=0.0, le=1.0)
    eu_weight_max: float = Field(default=0.50, ge=0.0, le=1.0)

    # Number of selected assets
    na_top_stocks: int = Field(default=35, ge=1)
    eu_top_stocks: int = Field(default=25, ge=1)


MACRO_SERIES_NA: tuple[str, ...] = ("VIX", "BAMLH0A0HYM2", "T10Y2Y", "TURBULENCE_NA")
MACRO_SERIES_EU: tuple[str, ...] = ("V2TX", "BAMLHE00EHYIOAS", "BUND10Y2Y", "TURBULENCE_EU")

FUTURE_MULTIPLIERS: dict[str, float] = {
    "ES": 50.0,
    "MES": 5.0,
    "FXXP": 50.0,
    "FESX": 10.0,
}

FUTURE_CURRENCIES: dict[str, str] = {
    "ES": "USD",
    "MES": "USD",
    "FXXP": "EUR",
    "FESX": "EUR",
}


class RegimeHedgingConfig(BaseModel):
    """Parameters for dual-sleeve HMM regime sensors and futures overlay."""

    n_states: int = Field(default=3, ge=2, le=5)
    covariance_type: str = Field(default="diag")
    min_history_days: int = Field(default=126, ge=63)
    recalibration_freq_days: int = Field(default=21, ge=1)
    inertia_days: int = Field(default=3, ge=1)
    stress_open_prob: float = Field(default=0.70, ge=0.0, le=1.0)
    stress_close_prob: float = Field(default=0.40, ge=0.0, le=1.0)
    default_na_future: str = Field(default="ES")
    default_eu_future: str = Field(default="FXXP")
    macro_series_na: tuple[str, ...] = MACRO_SERIES_NA
    macro_series_eu: tuple[str, ...] = MACRO_SERIES_EU
    hedge_ratios: dict[int, float] = Field(
        default_factory=lambda: {
            0: 0.0,  # Calm: 0% hedge
            1: 0.5,  # Alert / Transition: 50% * Beta
            2: 1.0,  # Severe Stress: 100% * Beta
        }
    )


class AlphaConfig(BaseModel):
    """Configuration for Phase 4 Machine Learning model and Purged CV."""

    forward_horizon_days: int = Field(default=21, ge=1)
    purge_days: int = Field(default=21, ge=0)
    embargo_days: int = Field(default=5, ge=0)
    min_train_periods: int = Field(default=252, ge=20)

    # LightGBM anti-overfitting hyperparameters
    lgbm_max_depth: int = Field(default=4, ge=1)
    lgbm_min_child_samples: int = Field(default=30, ge=2)
    lgbm_n_estimators: int = Field(default=100, ge=10)
    lgbm_learning_rate: float = Field(default=0.03, gt=0.0)
    lgbm_reg_alpha: float = Field(default=0.1, ge=0.0)
    lgbm_reg_lambda: float = Field(default=1.0, ge=0.0)

    # Random Forest hyperparameters
    rf_max_depth: int = Field(default=4, ge=1)
    rf_min_samples_leaf: int = Field(default=30, ge=1)
    rf_n_estimators: int = Field(default=100, ge=10)

    # Ensemble parameters
    ensemble_lgbm_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    random_state: int = Field(default=42)


class RiskConfig(BaseModel):
    """Configuration for multi-day synchronization, RMT denoising, and portfolio allocation."""

    resampling_days: int = Field(default=3, ge=1, le=10)
    history_days: int = Field(default=504, ge=63)
    cvar_confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    cvar_lambda_return: float = Field(default=0.0, ge=0.0)
    hrp_linkage_method: str = Field(default="ward")
    default_optimizer: Literal["hrp", "cvar", "ensemble"] = Field(default="hrp")


class PortfolioConfig(BaseModel):
    """Global system configuration."""

    base_currency: str = BASE_CURRENCY
    financials_gics_sector: int = FINANCIALS_GICS_SECTOR
    paths: StoragePaths = Field(default_factory=StoragePaths)
    allocation: AllocationConstraints = Field(default_factory=AllocationConstraints)
    regime: RegimeHedgingConfig = Field(default_factory=RegimeHedgingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    alpha: AlphaConfig = Field(default_factory=AlphaConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
