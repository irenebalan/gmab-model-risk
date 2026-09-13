from pathlib import Path


# ---------------------------------------------------------------------------
# CARTELLE
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR


# ---------------------------------------------------------------------------
# SOTTOSTANTE
# ---------------------------------------------------------------------------

ETF_TICKER = "CSUS.SW"
ETF_ISIN = "IE00B52SFT06"

LONG_INDEX_TICKER = "^SP500TR"

ETF_CURRENCY = "USD"
INDEX_CURRENCY = "USD"
CONTRACT_CURRENCY = "EUR"

ETF_START_DATE = "2010-01-12"
LONG_INDEX_START_DATE = "1999-01-04"

END_DATE = "2026-08-21"


# ---------------------------------------------------------------------------
# CAMBIO EUR/USD
# ---------------------------------------------------------------------------

FX_SERIES_KEY = "EXR.D.USD.EUR.SP00.A"

FILE_FX_EURUSD = (
    BASE_DIR / "eurusd_ecb.csv"
)


# ---------------------------------------------------------------------------
# VOLATILITA' IMPLICITA
# ---------------------------------------------------------------------------

VIX_TICKER = "^VIX"


# ---------------------------------------------------------------------------
# CONTRATTO / GMAB
# ---------------------------------------------------------------------------

PREMIO_LORDO = 100.0
CARICAMENTO = 0.015

INITIAL_FUND_VALUE = (
    PREMIO_LORDO
    * (1.0 - CARICAMENTO)
)

GMAB_MATURITY_YEARS = 10

GUARANTEE_LEVEL = 1.00
G = PREMIO_LORDO * GUARANTEE_LEVEL

GUARANTEE_LEVELS = (
    0.90,
    1.00,
    1.10,
)


# ---------------------------------------------------------------------------
# COMMISSIONI
# ---------------------------------------------------------------------------

ETF_TER_ANNUAL = 0.0003
POLICY_FEE_ANNUAL = 0.0160

TOTAL_FEE_ANNUAL = (
    ETF_TER_ANNUAL
    + POLICY_FEE_ANNUAL
)


# ---------------------------------------------------------------------------
# TASSO RISK-FREE
# ---------------------------------------------------------------------------

RISK_FREE_CURRENCY = "EUR"

RISK_FREE_SERIES_TEMPLATE = (
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor}Y"
)

RISK_FREE_TENORS_YEARS = tuple(
    range(
        1,
        GMAB_MATURITY_YEARS + 1,
    )
)

FILE_RISK_FREE_RATES = (
    BASE_DIR / "risk_free_eur_ecb.csv"
)


# ---------------------------------------------------------------------------
# BACKTEST / HEDGING
# ---------------------------------------------------------------------------

BACKTEST_WINDOW_YEARS = (
    GMAB_MATURITY_YEARS
)

BACKTEST_STEP_YEARS = 1

TRADING_DAYS_PER_YEAR = 252

REBALANCE_FREQUENCIES = {
    "giornaliera": 1,
    "mensile": 21,
    "trimestrale": 63,
}

BASE_REBALANCE_FREQUENCY = "mensile"

TRANSACTION_COST_RATE = 0.0004

TRANSACTION_COST_SCENARIOS = (
    0.0,
    TRANSACTION_COST_RATE,
)

# ---------------------------------------------------------------------------
# DRAG DELLA PROXY
# ---------------------------------------------------------------------------

PROXY_DRAG_CONTINUOUS = 0.004858385498415007

PROXY_DRAG_SCENARIOS = {
    "senza_drag_proxy": 0.0,
    "con_drag_proxy": PROXY_DRAG_CONTINUOUS,
}

# ---------------------------------------------------------------------------
# FILE DI INPUT E OUTPUT
# ---------------------------------------------------------------------------

FILE_DATI_MERCATO = (
    BASE_DIR / "dati_mercato_eur.csv"
)

FILE_VOLATILITA_FINESTRE = (
    BASE_DIR / "volatilita_storica_per_finestra.csv"
)

FILE_DATI_VOLATILITA = (
    BASE_DIR / "dati_volatilita.csv"
)

FILE_PREZZO_GMAB = (
    BASE_DIR / "prezzo_gmab_per_scenario.csv"
)
