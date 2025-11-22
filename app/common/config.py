from datetime import datetime, timedelta

# --------- ACTIONS : séparées par indice --------- #
EQUITY_INDICES = {
    "CAC 40": {
        "MC.PA": "LVMH",
        "OR.PA": "L'Oréal",
        "AIR.PA": "Airbus",
        "SAN.PA": "Sanofi",
        "BNP.PA": "BNP Paribas",
        "GLE.PA": "Société Générale",
        "TTE.PA": "TotalEnergies",
        "AI.PA": "Air Liquide",
        "ENGI.PA": "Engie",
        "DG.PA": "Vinci",
    },
    "S&P 500": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet (Google)",
        "AMZN": "Amazon",
        "META": "Meta Platforms",
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
        "JPM": "JPMorgan Chase",
        "XOM": "Exxon Mobil",
        "JNJ": "Johnson & Johnson",
    }
    # tu peux rajouter "NASDAQ 100", etc.
}

# --------- AUTRES CLASSES D'ACTIFS --------- #
FOREX_PAIRS = {
    "EURUSD=X": "EUR / USD",
    "GBPUSD=X": "GBP / USD",
    "USDJPY=X": "USD / JPY",
}

COMMODITIES = {
    "GC=F":  "Gold (COMEX)",
    "SI=F":  "Silver (COMEX)",
    "CL=F":  "Crude Oil WTI",
    "BZ=F":  "Brent Crude Oil",
    "NG=F":  "Natural Gas",
    "HG=F":  "Copper",
    "ZS=F":  "Soybean",
    "ZC=F":  "Corn",
    "ZW=F":  "Wheat",
}

# ========= CRYPTO ========= #
# paires contre USD, très liquides

CRYPTO_ASSETS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "BNB",
    "SOL-USD": "Solana",
    "XRP-USD": "XRP",
    "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin",
    "LTC-USD": "Litecoin",
}

# ========= ETF ========= #
# ETF actions globaux/sectoriels très connus

ETF_ASSETS = {
    "SPY":   "SPDR S&P 500 ETF",
    "QQQ":   "Invesco QQQ (Nasdaq 100)",
    "DIA":   "Dow Jones (DIA)",
    "IWM":   "Russell 2000 (IWM)",

    "EWJ":   "MSCI Japan (EWJ)",
    "EWZ":   "MSCI Brazil (EWZ)",
    "EEM":   "MSCI Emerging Markets (EEM)",

    "GLD":   "SPDR Gold Trust",
    "SLV":   "iShares Silver Trust",
    "USO":   "United States Oil Fund",
}


# --------- STRUCTURE GLOBALE --------- #
ASSET_CLASSES = {
    "Actions": EQUITY_INDICES,          # dict d'indices -> dict symbol -> label
    "Forex": FOREX_PAIRS,               # dict symbol -> label
    "Matières premières": COMMODITIES,  # dict symbol -> label
    "Crypto": CRYPTO_ASSETS,            # dict symbol -> label
    "ETF": ETF_ASSETS,                  # dict symbol -> label
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
