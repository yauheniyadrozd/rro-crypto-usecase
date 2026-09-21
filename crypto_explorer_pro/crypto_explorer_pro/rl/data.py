import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ..api import CoinGecko, APIError


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wejście: timestamp, price, volume -> wyjście: te + wskaźniki."""
    out = df.copy()
    out["return_1"] = out["price"].pct_change().fillna(0)
    out["log_return"] = np.log1p(out["return_1"])
    out["ma_7"] = out["price"].rolling(7, min_periods=1).mean()
    out["ma_21"] = out["price"].rolling(21, min_periods=1).mean()
    out["ma_ratio"] = out["ma_7"] / out["ma_21"] - 1
    out["volatility_7"] = out["return_1"].rolling(7, min_periods=1).std().fillna(0)
    out["volatility_21"] = out["return_1"].rolling(21, min_periods=1).std().fillna(0)
    out["rsi_14"] = _rsi(out["price"])
    out["vol_change"] = out["volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    return out


def fetch_multi_asset_dataset(
    coin_ids: list[str],
    days: int = 365,
    cache_dir: str | None = None,
    progress: callable | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Pobiera dane dla listy aktywów, cache'ując każdą monetę osobno (inkrementalnie).
    """
    cg = CoinGecko()
    frames = {}
    total = len(coin_ids)
    for i, cid in enumerate(coin_ids, start=1):
        raw = None
        if cache_dir:
            path = os.path.join(cache_dir, f"{cid}_{days}.csv")
            if os.path.exists(path):
                try:
                    raw = pd.read_csv(path, parse_dates=["timestamp"])
                except Exception:
                    raw = None

        if raw is None:
            try:
                raw = cg.get_history(cid, days=days)
            except APIError as e:
                raise RuntimeError(f"Nie udało się pobrać danych dla {cid}: {e}") from e
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                raw.to_csv(os.path.join(cache_dir, f"{cid}_{days}.csv"), index=False)

        frames[cid] = build_asset_features(raw)
        if progress:
            progress(i, total, cid)

    min_len = min(len(f) for f in frames.values())
    for cid in frames:
        frames[cid] = frames[cid].iloc[-min_len:].reset_index(drop=True)

    return frames


def load_cached_dataset(cache_dir: str, coin_ids: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Wczytuje cache (dla danego `days`) i przelicza cechy — bez wywołań API."""
    frames = {}
    for cid in coin_ids:
        path = os.path.join(cache_dir, f"{cid}_{days}.csv")
        frames[cid] = build_asset_features(pd.read_csv(path, parse_dates=["timestamp"]))

    min_len = min(len(f) for f in frames.values())
    for cid in frames:
        frames[cid] = frames[cid].iloc[-min_len:].reset_index(drop=True)
    return frames