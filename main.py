import time
from stable_baselines3 import DQN
from environment.custom_env import SeedDeliveryEnv

def evaluate_dqn(model_path="models/dqn/dqn_seed_delivery", render=True, delay=0.5, num_episodes=10):
    env = SeedDeliveryEnv(render_mode="human")
    model = DQN.load(model_path)

    for episode_num in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render()
                time.sleep(delay)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        print(f"Episode {episode_num} finished with total reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate_dqn()