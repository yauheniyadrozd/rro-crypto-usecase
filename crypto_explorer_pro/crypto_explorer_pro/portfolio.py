"""Portfel startowy (~1000 USD, kupiony drogo) i planowanie rebalancingu.

Portfel to zestaw monet "już kupionych" po ustalonej cenie zakupu (cost basis),
celowo powyżej bieżącego rynku — czyli portfel jest na stracie. Aktualna wartość
liczona jest z cen live, a rebalancing z wag agentów RL (z opłatami i podatkiem).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rl.policies import blended_weights, agent_pol_for_state, blend_weights
from .rl._math import trade_costs, FEE_RATE, TAX_RATE

# (coin_id, etykieta, ilość, cena_zakupu)  ->  łącznie dokładnie 1000 USD.
# Ceny zakupu są zawyżone (zakup na "górce"), więc portfel jest na minusie.
INITIAL_HOLDINGS = [
    ("bitcoin",     "BTC",  0.003, 120_000.0),  # 360.00
    ("ethereum",    "ETH",  0.03,    8_000.0),  # 240.00
    ("binancecoin", "BNB",  0.1,     1_000.0),  # 100.00
    ("solana",      "SOL",  0.3,       300.0),  #  90.00
    ("cardano",     "ADA",  30.0,       2.00),  #  60.00
    ("avalanche-2", "AVAX", 0.75,       80.0),  #  60.00
    ("chainlink",   "LINK", 1.5,        40.0),  #  60.00
    ("uniswap",     "UNI",  1.0,        30.0),  #  30.00
]

COIN_IDS = [h[0] for h in INITIAL_HOLDINGS]
LABELS = {h[0]: h[1] for h in INITIAL_HOLDINGS}


@dataclass
class Holding:
    coin_id: str
    label: str
    quantity: float
    buy_price: float

    def invested(self) -> float:
        return self.quantity * self.buy_price


class Portfolio:
    def __init__(self, holdings: list[Holding] | None = None):
        self.holdings = holdings or [Holding(*h) for h in INITIAL_HOLDINGS]

    @property
    def coin_ids(self) -> list[str]:
        return [h.coin_id for h in self.holdings]

    def quantity(self, coin_id: str) -> float:
        for h in self.holdings:
            if h.coin_id == coin_id:
                return h.quantity
        return 0.0

    def buy_price(self, coin_id: str) -> float:
        for h in self.holdings:
            if h.coin_id == coin_id:
                return h.buy_price
        return 0.0

    def total_invested(self) -> float:
        return sum(h.invested() for h in self.holdings)

    def snapshot(self, prices: dict[str, float]) -> dict:
        rows = []
        total_value = 0.0
        for h in self.holdings:
            px = float(prices.get(h.coin_id, h.buy_price))
            value = h.quantity * px
            pnl = value - h.invested()
            total_value += value
            rows.append({
                "coin_id": h.coin_id,
                "label": h.label,
                "quantity": h.quantity,
                "buy_price": h.buy_price,
                "price": px,
                "value": value,
                "pnl": pnl,
            })

        invested = self.total_invested()
        for r in rows:
            r["weight"] = (r["value"] / total_value) if total_value else 0.0
            r["pnl_pct"] = (r["pnl"] / r["buy_price"] / r["quantity"] * 100) if r["quantity"] else 0.0

        return {
            "rows": rows,
            "total_value": total_value,
            "total_invested": invested,
            "pnl": total_value - invested,
            "pnl_pct": (total_value - invested) / invested * 100 if invested else 0.0,
        }


def rebalance_plan(portfolio: Portfolio, prices: dict[str, float], vol: float, enabled: dict | None = None, models: dict | None = None, frames: dict | None = None) -> dict:
    """Plan transakcji przesuwający portfel do alokacji docelowej agentów.

    Do każdej transakcji dolicza opłatę giełdową (FEE_RATE) i podatek od
    zrealizowanego zysku (TAX_RATE, tylko przy sprzedaży z zyskiem). Gdy
    podano `models` + `frames`, wagi liczy z modeli RL (inaczej deterministycznie).
    """
    snap = portfolio.snapshot(prices)
    n = len(portfolio.coin_ids)

    if models is not None and frames is not None:
        t = len(frames[portfolio.coin_ids[0]]) - 1
        agent_pol = agent_pol_for_state(frames, portfolio, portfolio.coin_ids, t, prices, models)
        target, _coeffs = blend_weights(agent_pol, vol, enabled)
    else:
        target = blended_weights(vol, n, enabled=enabled)["final_weights"]

    total = snap["total_value"]
    trades = []
    total_fee = 0.0
    total_tax = 0.0
    for i, h in enumerate(portfolio.holdings):
        target_value = target[i] * total
        px = float(prices.get(h.coin_id, h.buy_price))
        current_value = h.quantity * px
        delta = target_value - current_value
        fee, tax = trade_costs(delta, px, h.buy_price)
        total_fee += fee
        total_tax += tax
        trades.append({
            "coin_id": h.coin_id,
            "label": h.label,
            "delta": delta,
            "action": "KUP" if delta >= 0 else "SPRZEDAJ",
            "target_weight": target[i],
            "fee": fee,
            "tax": tax,
        })

    cash_target = target[-1] * total
    return {
        "snapshot": snap,
        "target_weights": target,
        "trades": trades,
        "cash_target": cash_target,
        "volatility": vol,
        "total_fee": total_fee,
        "total_tax": total_tax,
        "total_cost": total_fee + total_tax,
    }
