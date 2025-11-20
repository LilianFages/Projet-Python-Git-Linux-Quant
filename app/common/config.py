from datetime import datetime, timedelta

# Univers d'actifs organisé par classes
ASSET_CLASSES = {
    "Actions": {
        "AAPL": "Apple (AAPL)",
        "MSFT": "Microsoft (MSFT)",
        "GOOGL": "Alphabet (GOOGL)",
        "ENGI.PA": "Engie (ENGI.PA)",
    },
    "Forex": {
        "EURUSD=X": "EUR / USD",
        "GBPUSD=X": "GBP / USD",
        "USDJPY=X": "USD / JPY",
    },
    "Matières premières": {
        "GC=F": "Or (Gold futures)",
        "SI=F": "Argent (Silver futures)",
        "CL=F": "Pétrole (Crude Oil)",
    },
}

DEFAULT_ASSET_CLASS = "Actions"
DEFAULT_SINGLE_ASSET = "AAPL"

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_INTERVAL = "1d"


def default_start_end():
    end = datetime.today()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start, end
