import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .env import TradingEnv, ShadowTradingEnv


def train_realist(frames, total_timesteps=50_000, save_path="models/agent1_realist.zip", **env_kwargs):
    env = Monitor(TradingEnv(frames, **env_kwargs))
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64,
                gamma=0.99, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    return model


def train_shadow(frames, agent1_model, bias, total_timesteps=50_000,
                  save_path="models/agentX_shadow.zip", mimic_weight=0.7, **env_kwargs):
    env = Monitor(ShadowTradingEnv(
        frames, agent1_model=agent1_model, bias=bias, mimic_weight=mimic_weight, **env_kwargs
    ))
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64,
                gamma=0.99, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    return model


def train_all(frames, timesteps_per_agent=50_000, out_dir="models"):
    """Trenuje wszystkich trzech agentów po kolei i zwraca ich modele."""
    agent1 = train_realist(
        frames, total_timesteps=timesteps_per_agent,
        save_path=os.path.join(out_dir, "agent1_realist.zip"),
    )
    agent2 = train_shadow(
        frames, agent1_model=agent1, bias=-0.15,
        total_timesteps=timesteps_per_agent,
        save_path=os.path.join(out_dir, "agent2_pessimist.zip"),
    )
    agent3 = train_shadow(
        frames, agent1_model=agent1, bias=+0.15,
        total_timesteps=timesteps_per_agent,
        save_path=os.path.join(out_dir, "agent3_optimist.zip"),
    )
    return {"realist": agent1, "pessimist": agent2, "optimist": agent3}


def load_agents(model_dir="models"):
    return {
        "realist": PPO.load(os.path.join(model_dir, "agent1_realist.zip")),
        "pessimist": PPO.load(os.path.join(model_dir, "agent2_pessimist.zip")),
        "optimist": PPO.load(os.path.join(model_dir, "agent3_optimist.zip")),
    }