import time
import requests
import pandas as pd
from datetime import datetime
from .config import COINGECKO_BASE


class APIError(Exception):
    pass


class CoinGecko:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._last = 0.0

    def get_history(self, coin_id: str, days: int) -> pd.DataFrame:
        """Zwraca DataFrame z kolumnami: timestamp, price, volume."""
        self._throttle()
        url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        return self._fetch_and_parse(url, params)

    def get_history_range(
        self, coin_id: str, date_from: datetime, date_to: datetime
    ) -> pd.DataFrame:
        """Pobiera dane dla niestandardowego zakresu dat."""
        self._throttle()
        url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": int(date_from.timestamp()),
            "to":   int(date_to.timestamp()),
        }
        return self._fetch_and_parse(url, params)

    def get_current_prices(self, coin_ids: list[str]) -> dict[str, float]:
        """Zwraca {coin_id: bieżąca cena USD} jednym zapytaniem."""
        self._throttle()
        url = f"{COINGECKO_BASE}/simple/price"
        params = {"ids": ",".join(coin_ids), "vs_currencies": "usd"}
        try:
            r = self._session.get(url, params=params, timeout=20)
            r.raise_for_status()
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 429:
                raise APIError("Przekroczono limit API CoinGecko (429). Odczekaj ~60 s i spróbuj ponownie.")
            raise APIError(f"Błąd HTTP {code}") from e
        except requests.RequestException as e:
            raise APIError(f"Błąd połączenia: {e}") from e

        data = r.json()
        return {cid: float(data[cid]["usd"]) for cid in coin_ids if cid in data}

    def _fetch_and_parse(self, url: str, params: dict) -> pd.DataFrame:
        try:
            r = self._session.get(url, params=params, timeout=20)
            r.raise_for_status()
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 429:
                raise APIError("Przekroczono limit API CoinGecko (429). Odczekaj ~60 s i spróbuj ponownie.")
            raise APIError(f"Błąd HTTP {code}") from e
        except requests.RequestException as e:
            raise APIError(f"Błąd połączenia: {e}") from e

        data = r.json()
        if not data.get("prices"):
            raise APIError("Brak danych dla wybranego zakresu.")

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                [p[0] for p in data["prices"]], unit="ms"
            ),
            "price":  [p[1] for p in data["prices"]],
            "volume": [v[1] for v in data["total_volumes"]],
        })
        return df.sort_values("timestamp").reset_index(drop=True)

    def _throttle(self):
        wait = 2.0 - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
