from crypto_explorer_pro.portfolio import COIN_IDS
from crypto_explorer_pro.rl.data import fetch_multi_asset_dataset
from crypto_explorer_pro.rl.agents import train_all

TIMESTEPS = 20_000
CACHE_DIR = "data_cache"
DAYS = 365
OUT_DIR = "models"


def main():
    print("1) Wczytywanie/pobieranie danych...")
    frames = fetch_multi_asset_dataset(
        COIN_IDS, days=DAYS, cache_dir=CACHE_DIR,
        progress=lambda d, t, c: print(f"   {d}/{t} {c}"),
    )

    print(f"2) Trening {TIMESTEPS} kroków na agenta...")
    agents = train_all(frames, timesteps_per_agent=TIMESTEPS, out_dir=OUT_DIR)
    print("3) Gotowe. Modele:", list(agents.keys()))


if __name__ == "__main__":
    main()
