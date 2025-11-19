from datetime import datetime, timedelta

ASSET_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

DEFAULT_SINGLE_ASSET = "AAPL"

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_INTERVAL = "1d"  # format yfinance


def default_start_end():
    end = datetime.today()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return start, end
