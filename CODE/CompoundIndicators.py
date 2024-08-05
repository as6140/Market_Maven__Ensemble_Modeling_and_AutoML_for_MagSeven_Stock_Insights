def bollinger_bands(prices, window=20):
    SMA = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = SMA + (rolling_std * 2)
    lower_band = SMA - (rolling_std * 2)
    BB_percentage = (prices - lower_band) / (upper_band - lower_band)
    return BB_percentage

def ichimoku_cloud(high, low):
    conversion_line = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    base_line = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    leading_span_A = ((conversion_line + base_line) / 2).shift(26)
    leading_span_B = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    ratio = leading_span_A / leading_span_B
    indicator = ratio.where(leading_span_A > leading_span_B, -1 / ratio)
    return indicator

def volume_oscillator(volume, short_window=12, long_window=26):
    SMA_short = volume.rolling(window=short_window).mean()
    SMA_long = volume.rolling(window=long_window).mean()
    return ((SMA_short - SMA_long) / SMA_long) * 100

def MACD(prices, n_fast=12, n_slow=26, n_signal=9):
    EMA_fast = prices.ewm(span=n_fast, min_periods=n_fast).mean()
    EMA_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD_line = EMA_fast - EMA_slow
    Signal_line = MACD_line.ewm(span=n_signal, min_periods=n_signal).mean()
    MACD_hist = MACD_line - Signal_line
    return MACD_hist