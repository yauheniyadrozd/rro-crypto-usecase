import numpy as np

FEATURE_COLS = [
    "return_1", "ma_ratio", "volatility_7", "volatility_21", "rsi_14", "vol_change",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# Koszty rebalancingu
FEE_RATE = 0.001   # 0.1% opłata giełdowa od wartości transakcji
TAX_RATE = 0.19    # 19% podatek od zrealizowanego zysku


def trade_costs(delta: float, price: float, buy_price: float) -> tuple[float, float]:
    """Zwraca (opłata, podatek) dla transakcji o wartości `delta` USD.

    `delta < 0` oznacza sprzedaż; podatek naliczany jest tylko od
    zrealizowanego zysku (sprzedaż powyżej ceny zakupu).
    """
    fee = abs(delta) * FEE_RATE
    tax = 0.0
    if delta < 0 and price > 0:
        sold_qty = abs(delta) / price
        realized = sold_qty * (price - buy_price)
        if realized > 0:
            tax = realized * TAX_RATE
    return fee, tax