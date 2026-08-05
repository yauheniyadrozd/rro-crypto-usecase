"""Wskaźniki techniczne i statystyki."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Stats:
    open:        float
    close:       float
    high:        float
    low:         float
    change_pct:  float
    change_abs:  float
    avg_price:   float
    avg_volume:  float
    volatility:  float
    ma7:         float
    ma25:        float
    rsi:         float

    @property
    def is_up(self) -> bool:
        return self.change_pct >= 0


def compute_stats(df: pd.DataFrame) -> Stats:
    p   = df["price"]
    ret = p.pct_change().dropna()
    open_  = float(p.iloc[0])
    close  = float(p.iloc[-1])
    ind    = add_indicators(df)
    return Stats(
        open       = open_,
        close      = close,
        high       = float(p.max()),
        low        = float(p.min()),
        change_pct = (close - open_) / open_ * 100 if open_ else 0.0,
        change_abs = close - open_,
        avg_price  = float(p.mean()),
        avg_volume = float(df["volume"].mean()),
        volatility = float(ret.std() * 100) if len(ret) > 1 else 0.0,
        ma7        = float(ind["ma7"].iloc[-1]),
        ma25       = float(ind["ma25"].iloc[-1]),
        rsi        = float(ind["rsi"].iloc[-1]),
    )


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje ma7, ma25, bb_upper, bb_lower, rsi."""
    df = df.copy()
    p  = df["price"]

    df["ma7"]  = p.rolling(7,  min_periods=1).mean()
    df["ma25"] = p.rolling(25, min_periods=1).mean()

    roll          = p.rolling(20, min_periods=5)
    df["bb_mid"]   = roll.mean()
    df["bb_upper"] = df["bb_mid"] + 2 * roll.std()
    df["bb_lower"] = df["bb_mid"] - 2 * roll.std()

    delta = p.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    return df
