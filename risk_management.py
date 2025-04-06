import numpy as np
import pandas as pd

def add_regime_features(df,
                        vol_window: int = 20,
                        trend_short: int = 50,
                        trend_long: int = 200) -> pd.DataFrame:
    """
    Adds regime flags to df (must have 'date' & 'close'):
      - ret: next-day return
      - vol_regime: 1=high-vol, 0=low-vol
      - trend_regime: 1=uptrend, -1=downtrend
    """
    df = df.sort_values("date").copy().reset_index(drop=True)
    # next-day return
    df['ret'] = df['close'].pct_change().shift(-1)
    # rolling vol & mean
    df['vol']      = df['ret'].rolling(vol_window).std()
    df['vol_mean'] = df['vol'].rolling(vol_window).mean()
    df['vol_regime'] = np.where(df['vol'] > df['vol_mean'], 1, 0)
    # SMAs for trend
    df['sma_short']   = df['close'].rolling(trend_short).mean()
    df['sma_long']    = df['close'].rolling(trend_long).mean()
    df['trend_regime'] = np.where(df['sma_short'] > df['sma_long'], 1, -1)
    return df

def adjust_exposure_by_regime(df) -> pd.DataFrame:
    """
    Takes raw signals and:
      - FLIPS them (×−1) when in Low Vol & Downtrend
      - SCALES them by 1.5× when in High Vol & Uptrend
      - leaves them untouched in other regimes
    """
    # align
    signals = df['combo_signal'].reindex(df.index).fillna(0).astype(float)
    df['raw_signal'] = signals
    df['exposure']   = df['raw_signal']
    
    # 1) Low Vol (0) & Downtrend (-1) → flip
    mask_flip = (df['vol_regime'] == 0) & (df['trend_regime'] == -1)
    df.loc[mask_flip, 'exposure'] *= -1
    
    # 2) High Vol (1) & Uptrend (1) → boost 1.5×
    mask_boost = (df['vol_regime'] == 1) & (df['trend_regime'] == 1)
    df.loc[mask_boost, 'exposure'] *= 1.5
    
    return df