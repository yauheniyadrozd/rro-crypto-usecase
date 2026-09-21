import numpy as np

from ._math import _softmax, trade_costs, FEATURE_COLS
from .blend import volatility_to_weights


def agent_weights(n_assets: int) -> dict[str, np.ndarray]:
    """Wagi (N aktywów + gotówka) dla trzech deterministycznych agentów (fallback).

    Pesymista trzyma najwięcej gotówki, optymista najwięcej aktywów.
    """
    def _policy(asset_logit: float, cash_logit: float) -> np.ndarray:
        logits = [asset_logit] * n_assets + [cash_logit]
        return _softmax(np.asarray(logits, dtype=np.float64))

    return {
        "pessimist": _policy(0.6, 1.8),
        "realist":   _policy(1.0, 1.0),
        "optimist":  _policy(1.6, 0.5),
    }


def build_observation(frames, coin_ids: list[str], t: int, asset_weights: np.ndarray) -> np.ndarray:
    """Buduje obserwację (taką jak w TradingEnv) dla dnia `t`.

    Obserwacja = cechy rynku (N aktywów × FEATURE_COLS) + aktualne wagi portfela.
    """
    feat = np.stack(
        [frames[c][FEATURE_COLS].iloc[t].to_numpy() for c in coin_ids]
    ).flatten()
    port = np.concatenate([np.asarray(asset_weights, dtype=np.float64), [0.0]])
    return np.concatenate([feat, port]).astype(np.float32)


def agent_weights_from_models(models: dict, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Wagi trzech agentów z wytrenowanych modeli PPO (softmax z akcji)."""
    out = {}
    for key in ("pessimist", "realist", "optimist"):
        action, _ = models[key].predict(obs, deterministic=True)
        out[key] = _softmax(np.asarray(action, dtype=np.float64))
    return out


def agent_pol_for_state(frames, portfolio, coin_ids: list[str], t: int, prices: dict, models: dict | None = None) -> dict[str, np.ndarray]:
    """Wagi trzech agentów dla danego stanu (dzień `t`, ceny `prices`).

    Gdy `models` jest dostępne — używa modeli; w przeciwnym razie fallback
    deterministyczny.
    """
    n_assets = len(coin_ids)
    holdings_value = {c: portfolio.quantity(c) * float(prices.get(c, 0.0)) for c in coin_ids}
    total = sum(holdings_value.values())
    asset_w = (
        np.array([holdings_value[c] / total for c in coin_ids])
        if total else np.zeros(n_assets)
    )
    if models is not None:
        obs = build_observation(frames, coin_ids, t, asset_w)
        return agent_weights_from_models(models, obs)
    return agent_weights(n_assets)


def blend_weights(agent_pol: dict, vol: float, enabled: dict | None = None) -> tuple[np.ndarray, dict]:
    """Blenduje wagi trzech agentów zmiennością rynku -> (final, coeffs).

    `enabled` pozwala wyłączyć agentów; ich waga przechodzi na pozostałych.
    """
    if enabled is None:
        enabled = {}

    w_pess, w_real, w_opt = volatility_to_weights(vol)
    coeffs = {
        "pessimist": w_pess if enabled.get("pessimist", True) else 0.0,
        "realist":   w_real if enabled.get("realist", True) else 0.0,
        "optimist":  w_opt if enabled.get("optimist", True) else 0.0,
    }
    total = sum(coeffs.values())
    if total <= 0:
        coeffs = {"pessimist": 0.0, "realist": 1.0, "optimist": 0.0}
        total = 1.0
    coeffs = {k: v / total for k, v in coeffs.items()}

    final = (
        coeffs["pessimist"] * agent_pol["pessimist"]
        + coeffs["realist"] * agent_pol["realist"]
        + coeffs["optimist"] * agent_pol["optimist"]
    )
    final = final / final.sum()
    return final, coeffs


def blended_weights(vol: float, n_assets: int, enabled: dict | None = None) -> dict:
    """Deterministyczna alokacja (fallback bez modeli)."""
    pol = agent_weights(n_assets)
    final, coeffs = blend_weights(pol, vol, enabled)
    return {
        "final_weights": final,
        "agent_weights": pol,
        "blend_coeffs": coeffs,
        "volatility": float(vol),
    }


def build_rebalance_timeline(frames, portfolio, coin_ids: list[str], enabled: dict | None = None, models: dict | None = None) -> list[dict]:
    """Oś czasu rebalancingu: dla każdego dnia zwraca zmienność, wagi i transakcje.

    Na każdym kroku liczymy zmienność rynku, wagi docelowe agentów (modele RL
    lub fallback), a potem transakcje (z opłatami i podatkiem) potrzebne do
    przesunięcia portfela do alokacji docelowej.
    """
    prices = {c: frames[c]["price"].to_numpy() for c in coin_ids}
    n = min(len(prices[c]) for c in coin_ids)
    timestamps = frames[coin_ids[0]]["timestamp"]

    timeline: list[dict] = []
    for t in range(n):
        px = {c: float(prices[c][t]) for c in coin_ids}
        vol = float(np.mean([frames[c]["volatility_21"].iloc[t] for c in coin_ids]))

        agent_pol = agent_pol_for_state(frames, portfolio, coin_ids, t, px, models)
        target, coeffs = blend_weights(agent_pol, vol, enabled)

        holdings_value = {c: portfolio.quantity(c) * px[c] for c in coin_ids}
        total = sum(holdings_value.values())

        trades = []
        total_fee = 0.0
        total_tax = 0.0
        for i, c in enumerate(coin_ids):
            target_value = target[i] * total
            current_value = holdings_value[c]
            delta = target_value - current_value
            fee, tax = trade_costs(delta, px[c], portfolio.buy_price(c))
            total_fee += fee
            total_tax += tax
            trades.append({
                "coin": c,
                "delta": delta,
                "action": "KUP" if delta >= 0 else "SPRZEDAJ",
                "fee": fee,
                "tax": tax,
            })
        cash_target = target[-1] * total  # ile przenieść do gotówki

        timeline.append({
            "date": timestamps.iloc[t],
            "volatility": vol,
            "weights": target,
            "agent_weights": agent_pol,
            "blend_coeffs": coeffs,
            "total_value": total,
            "trades": trades,
            "cash_target": cash_target,
            "total_fee": total_fee,
            "total_tax": total_tax,
            "total_cost": total_fee + total_tax,
        })

    return timeline
