import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


from ._math import FEATURE_COLS, _softmax


class TradingEnv(gym.Env):
    """Środowisko wielo-aktywowe dla Agenta 1 (Realista / maksymalizacja zysku)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
        window: int = 1,
    ):
        super().__init__()
        self.coin_ids = list(frames.keys())
        self.n_assets = len(self.coin_ids)
        self.frames = frames
        self.prices = np.stack(
            [frames[c]["price"].to_numpy() for c in self.coin_ids], axis=1
        )  # shape (T, n_assets)
        self.features = np.stack(
            [frames[c][FEATURE_COLS].to_numpy() for c in self.coin_ids], axis=1
        )  # shape (T, n_assets, n_features)
        self.T = self.prices.shape[0]
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.window = window

        # akcja: N+1 logitów (N aktywów + cash) -> softmax -> target weights
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        n_feat = self.n_assets * len(FEATURE_COLS)
        obs_dim = n_feat + self.n_assets + 1  # cechy rynku + aktualne wagi portfela
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        weights = _softmax(np.asarray(action, dtype=np.float64))  # size n_assets+1
        target_asset_w = weights[:-1]
        target_cash_w = weights[-1]

        prev_value = self._portfolio_value(self.t)

        # koszty transakcyjne proporcjonalne do zmiany alokacji
        turnover = np.abs(target_asset_w - self._current_asset_weights()).sum()
        cost = turnover * self.transaction_cost * prev_value

        self.asset_weights = target_asset_w
        self.cash_weight = target_cash_w

        self.t += 1
        done = self.t >= self.T - 1

        new_value = self._portfolio_value(self.t) - cost
        new_value = max(new_value, 1e-6)

        reward = np.log(new_value / max(prev_value, 1e-6))  # log-return portfela
        self.value_history.append(new_value)
        self._last_value = new_value

        info = {
            "portfolio_value": new_value,
            "weights": target_asset_w.copy(),
            "volatility": self._market_volatility(),
        }
        return self._obs(), float(reward), done, False, info

    def _reset_state(self):
        self.t = 0
        self.asset_weights = np.zeros(self.n_assets)
        self.cash_weight = 1.0
        self._last_value = self.initial_cash
        self.value_history = [self.initial_cash]

    def _current_asset_weights(self):
        return self.asset_weights

    def _portfolio_value(self, t):
        # wartość portfela przy cenach z chwili t, zakładając wagi ustawione w self.asset_weights
        if t == 0:
            return self.initial_cash
        price_rel = self.prices[t] / self.prices[t - 1]
        asset_value = self._last_value * self.asset_weights * price_rel
        cash_value = self._last_value * self.cash_weight
        return float(asset_value.sum() + cash_value)

    def _market_volatility(self) -> float:
        # średnia zmienność 21-dniowa po aktywach - używana też przez blend.py
        vol_col_idx = FEATURE_COLS.index("volatility_21")
        return float(self.features[self.t, :, vol_col_idx].mean())

    def _obs(self) -> np.ndarray:
        feat = self.features[self.t].flatten()
        port = np.concatenate([self.asset_weights, [self.cash_weight]])
        return np.concatenate([feat, port]).astype(np.float32)


class ShadowTradingEnv(TradingEnv):
    """
    Środowisko dla Agenta 2 (Pesymista, bias<0) / Agenta 3 (Optymista, bias>0).

    Nagroda = kombinacja:
      (a) własny zysk portfela (żeby propozycja była sama w sobie sensowna),
      (b) "przydatność jako korekta": im bliżej decyzji Agenta 1 przesuniętej
          o `bias * zmienność_rynku`, tym lepiej. Dzięki mnożeniu biasu przez
          zmienność, przy dużej zmienności różnica względem Agenta 1 rośnie
          (silniejsza korekta w dół/górę), a przy małej zmienności agent
          "prawie zgadza się" z Agentem 1 - to dokładnie mechanizm, o którym
          pisałeś (duża zmienność -> bardziej słuchamy Pesymisty, mała ->
          Optymisty przy blendowaniu w blend.py).
    """

    def __init__(self, *args, agent1_model, bias: float, mimic_weight: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent1_model = agent1_model  # wytrenowany model Agenta 1 (stable-baselines3)
        self.bias = bias  # < 0 dla Pesymisty, > 0 dla Optymisty
        self.mimic_weight = mimic_weight  # 0..1, ile wagi na imitację vs własny zysk

    def step(self, action: np.ndarray):
        obs_before = self._obs()
        agent1_action, _ = self.agent1_model.predict(obs_before, deterministic=True)
        agent1_weights = _softmax(np.asarray(agent1_action, dtype=np.float64))

        vol = self._market_volatility()
        shift = self.bias * (0.3 + vol * 5.0)  # skala korekty rośnie ze zmiennością
        target = agent1_weights.copy()
        target[:-1] = np.clip(target[:-1] + shift, 0.0, 1.0)  # przesunięcie wag aktywów
        target = target / target.sum()

        own_weights = _softmax(np.asarray(action, dtype=np.float64))
        mimic_reward = -float(np.square(own_weights - target).sum())

        # zwykły krok środowiska liczy prawdziwy zysk portfela z własnej akcji
        _, profit_reward, done, truncated, info = super().step(action)

        reward = self.mimic_weight * mimic_reward + (1 - self.mimic_weight) * profit_reward
        info["agent1_weights"] = agent1_weights
        info["target_weights"] = target
        return self._obs(), float(reward), done, truncated, info