from datetime import datetime, timedelta

# --------- ACTIONS : séparées par indice --------- #
EQUITY_INDICES = {
    "CAC 40": {
        "ENGI.PA": "Engie (CAC 40)",
        "BNP.PA": "BNP Paribas (CAC 40)",
        "AIR.PA": "Airbus (CAC 40)",
    },
    "S&P 500": {
        "AAPL": "Apple (S&P 500)",
        "MSFT": "Microsoft (S&P 500)",
        "GOOGL": "Alphabet (S&P 500)",
        "AMZN": "Amazon (S&P 500)",
    },
    # tu peux rajouter "NASDAQ 100", etc.
}

# --------- AUTRES CLASSES D'ACTIFS --------- #
FOREX_PAIRS = {
    "EURUSD=X": "EUR / USD",
    "GBPUSD=X": "GBP / USD",
    "USDJPY=X": "USD / JPY",
}

COMMODITIES = {
    "GC=F": "Or (Gold futures)",
    "SI=F": "Argent (Silver futures)",
    "CL=F": "Pétrole (Crude Oil)",
}

# --------- STRUCTURE GLOBALE --------- #
ASSET_CLASSES = {
    "Actions": EQUITY_INDICES,     # attention: dict d'indices
    "Forex": FOREX_PAIRS,          # dict ticker -> label
    "Matières premières": COMMODITIES,
}

DEFAULT_ASSET_CLASS = "Actions"
DEFAULT_EQUITY_INDEX = "S&P 500"
DEFAULT_SINGLE_ASSET = "AAPL"

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_INTERVAL = "1d"


def default_start_end():
    end = datetime.today()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start, end
