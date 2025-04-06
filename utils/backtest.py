import numpy as np
import pandas as pd

def apply_signal_smoothing(signals: pd.Series, window: int = 3, persist: int = 2) -> pd.Series:
    smoothed = signals.rolling(window=window, min_periods=1).mean()
    binary = (smoothed > 0.5).astype(int)
    final_signal = binary.copy()
    binary = binary.reset_index(drop=True)  # Ensure integer index for access
    for i in range(persist, len(binary)):
        if not all(binary.iloc[i - persist:i] == binary.iloc[i]):
            final_signal.iloc[i] = final_signal.iloc[i - 1]
    return final_signal

def compute_market_regime(df: pd.DataFrame, vol_window: int = 5, trend_window: int = 5):
    df = df.copy()
    df['volatility'] = df['target'].rolling(vol_window).std()
    df['trend'] = df['target'].rolling(trend_window).mean().diff()

    df['vol_regime'] = np.where(df['volatility'] > df['volatility'].median(), 'high', 'low')
    df['trend_regime'] = np.where(df['trend'] < 0, 'down', 'up')
    return df

def compute_atr(df: pd.DataFrame, atr_window: int = 5) -> pd.Series:
    return df['target'].abs().rolling(atr_window).mean()

def backtest_portfolio(df: pd.DataFrame, raw_signals: pd.Series, initial_value: float = 1.0) -> pd.DataFrame:
    df = df.sort_values("date").copy().reset_index(drop=True)
    df['signal'] = apply_signal_smoothing(raw_signals.reset_index(drop=True))
    df = compute_market_regime(df)
    df['atr'] = compute_atr(df)

    portfolio = [initial_value]
    peak = initial_value
    exposures = []

    for i in range(len(df) - 1):
        signal = df.loc[i, 'signal']
        vol_regime = df.loc[i, 'vol_regime']
        trend_regime = df.loc[i, 'trend_regime']

        if vol_regime == 'high' and trend_regime == 'down':
            exposure = 1.5 * signal
        elif vol_regime == 'low' and trend_regime == 'down':
            exposure = 0.5 * signal
        else:
            exposure = 1.0 * signal

        today_val = portfolio[-1]
        next_ret = df.loc[i, 'target']
        next_val = today_val * (1 + exposure * next_ret)

        atr_threshold = 2.0 * df.loc[i, 'atr']
        drawdown = peak - next_val
        if drawdown > atr_threshold * peak:
            next_val = today_val
            exposure = 0

        peak = max(peak, next_val)
        portfolio.append(next_val)
        exposures.append(exposure)

    exposures.append(exposures[-1] if exposures else 0)
    df['portfolio'] = portfolio
    df['exposure'] = exposures
    return df