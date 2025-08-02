import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from stable_baselines3 import A2C
from environment.custom_env import SeedDeliveryEnv
from logger_utils import save_hyperparameters, init_csv_logger, append_to_csv

def train_a2c():
    env = SeedDeliveryEnv()

    hyperparams = {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "gae_lambda": 1.0
    }

    metric_dir = "metric/a2c/"
    save_hyperparameters(hyperparams, os.path.join(metric_dir, "hyperparameters.json"))
    csv_path = os.path.join(metric_dir, "training_metrics.csv")
    init_csv_logger(csv_path, ["episode", "reward"])

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **hyperparams
    )

    episode_rewards = []

    def reward_callback(_locals, _globals):
        if len(_locals["infos"]) > 0 and "episode" in _locals["infos"][0]:
            reward = _locals["infos"][0]["episode"]["r"]
            episode_rewards.append(reward)
            append_to_csv(csv_path, [len(episode_rewards), reward])
        return True

    model.learn(total_timesteps=200000, callback=reward_callback)
    model.save("models/pg/a2c_seed_delivery")

if __name__ == "__main__":
    train_a2c()