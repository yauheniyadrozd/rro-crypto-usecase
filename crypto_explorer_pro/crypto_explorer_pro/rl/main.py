from .data import fetch_multi_asset_dataset
from .agents import train_all, load_agents
from .env import TradingEnv
from .blend import blended_action, apply_user_split

COIN_IDS = ["bitcoin", "ethereum", "solana"]


def main():
    print("1) Pobieranie danych z CoinGecko...")
    frames = fetch_multi_asset_dataset(COIN_IDS, days=365, cache_dir="data_cache")

    print("2) Trening 3 agentów (Realista / Pesymista / Optymista)...")
    agents = train_all(frames, timesteps_per_agent=20_000, out_dir="models")

    print("3) Symulacja podejmowania decyzji na ostatnim kroku danych...")
    env = TradingEnv(frames)
    obs, _ = env.reset()
    for _ in range(env.T - 2):
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        if done:
            break
    vol = info["volatility"]

    result = blended_action(agents, obs, vol)
    print("Zmienność rynku:", vol)
    print("Wagi Realisty:  ", result["realist_weights"])
    print("Wagi Pesymisty: ", result["pessimist_weights"])
    print("Wagi Optymisty: ", result["optimist_weights"])
    print("Wagi FINALNE:   ", result["final_weights"])
    print("Współczynniki blendu:", result["blend_coeffs"])

    split = apply_user_split(capital=10_000, pct_module1=40, pct_module2=60)
    print("Podział kapitału użytkownika:", split)


if __name__ == "__main__":
    main()