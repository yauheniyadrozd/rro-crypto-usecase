import numpy as np
from ._math import _softmax


def volatility_to_weights(vol: float, low=0.01, high=0.05) -> tuple[float, float, float]:
    """
    Zamienia zmienność rynku na wagi (pesymista, realista, optymista).
    vol <= low   -> głównie optymista
    vol >= high  -> głównie pesymista
    pomiędzy     -> interpolacja liniowa, realista dopełnia
    """
    vol = float(np.clip(vol, low, high))
    t = (vol - low) / (high - low)  # 0 = spokojny rynek, 1 = burza
    w_pessimist = 0.10 + 0.55 * t
    w_optimist = 0.65 - 0.55 * t
    w_realist = 1.0 - w_pessimist - w_optimist
    return w_pessimist, w_realist, w_optimist


def blended_action(agents: dict, obs: np.ndarray, vol: float) -> dict:
    """
    Zwraca finalny wektor wag portfela (softmax po aktywach + cash),
    będący ważoną kombinacją trzech agentów, ważoną zmiennością rynku.
    """
    a1, _ = agents["realist"].predict(obs, deterministic=True)
    a2, _ = agents["pessimist"].predict(obs, deterministic=True)
    a3, _ = agents["optimist"].predict(obs, deterministic=True)

    w_pess, w_real, w_opt = volatility_to_weights(vol)

    w1 = _softmax(np.asarray(a1, dtype=np.float64))
    w2 = _softmax(np.asarray(a2, dtype=np.float64))
    w3 = _softmax(np.asarray(a3, dtype=np.float64))

    final = w_pess * w2 + w_real * w1 + w_opt * w3
    final = final / final.sum()

    return {
        "final_weights": final,          
        "realist_weights": w1,
        "pessimist_weights": w2,
        "optimist_weights": w3,
        "blend_coeffs": {"pessimist": w_pess, "realist": w_real, "optimist": w_opt},
        "volatility": vol,
    }


def apply_user_split(capital: float, pct_module1: float, pct_module2: float):
    """
    Rozdziela kapitał użytkownika na dwa moduły z opisu:
      - pct_module1: część kierowana do optymalizacji z korektami (na razie zaślepka)
      - pct_module2: część kierowana do RL (agenci 1/2/3)
    pct_* w procentach (0..100), suma nie może przekroczyć 100.
    """
    assert 0 <= pct_module1 + pct_module2 <= 100 + 1e-6, "Suma % nie może przekraczać 100"
    cash_module1 = capital * pct_module1 / 100.0
    cash_module2 = capital * pct_module2 / 100.0
    return {"module1_optimization": cash_module1, "module2_rl": cash_module2}
