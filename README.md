# 🚀 Seed Delivery RL Agent

**Author:** Mahamat Hissein Ahmat  
**Video Demo:** [Watch on Google Drive](https://drive.google.com/file/d/1bQxk_E-Usw-OhJAP6ZjndJ2OTHptnAOD/view?usp=sharing)  
**GitHub Repo:** [View on GitHub](https://github.com/MAHAMAT263/Mahamat__rl_summative.git)

---

## 🌟 Project Overview

This project develops and compares multiple reinforcement learning agents **DQN**, **PPO**, **REINFORCE**, and **A2C** to solve a custom **seed delivery task** in a grid-based environment.  
The agent must:

- Navigate from a **depot** to a **farm**
- Pick up and deliver seeds
- Avoid obstacles (rocks, dry land)

The environment uses **reward shaping** to guide learning, with penalties for inefficient or invalid actions.

---

## 🌾 Environment Description

| Feature            | Description                                                      |
|--------------------|------------------------------------------------------------------|
| **Agent**          | Drone that moves in four directions and delivers/picks up seeds |
| **Action Space**   | 5 discrete actions: `up`, `down`, `left`, `right`, `deliver`    |
| **Observation**    | Flattened 5x5 grid × 7 channels (multi-hot encoded)             |
| **Rewards**        |                                                                  |
|                    | ➕ +2 for picking up seed at depot                               |
|                    | ➕ +10 for successful delivery to farm                           |
|                    | ➖ -1 for invalid delivery or stepping on dry land               |
|                    | 🔄 Step penalty & shaped rewards to guide toward goals          |

---

## 🧠 Implemented Algorithms

| Algorithm   | Framework           | Highlights                                      |
|-------------|---------------------|-------------------------------------------------|
| **DQN**     | Stable-Baselines3   | Replay buffer, target network, ε-greedy policy |
| **PPO**     | Stable-Baselines3   | Stable and sample-efficient                    |
| **REINFORCE** | SB3-Contrib (LSTM) | Handles sequence memory                        |
| **A2C**     | Stable-Baselines3   | Good stability, low variance                   |

---

## 🔧 Hyperparameter Summary

| Method     | Learning Rate | Batch Size | Gamma | Notes                         |
|------------|---------------|------------|-------|-------------------------------|
| **DQN**    | 1e-3          | 64         | 0.99  | Simple and fast convergence   |
| **PPO**    | 3e-4          | 64         | 0.99  | Best overall performance      |
| **REINFORCE** | 3e-4       | 64         | 0.99  | LSTM-based, needs more time   |
| **A2C**    | 7e-4          | 64         | 0.99  | Most stable post-training     |

---

## 📊 Training Insights

- ✅ **DQN** showed **reliable convergence** with simple architecture.
- 🚀 **PPO** reached **highest reward** fastest (~1800 episodes).
- 📈 **A2C** delivered **consistent and stable** learning curves.
- 🧠 **REINFORCE** trained slower but benefited from recurrent policy.
- ⚖️ Overall: **PPO for performance**, **DQN for simplicity and speed**.

---

## 🎯 Generalization & Efficiency

- **PPO** and **A2C** achieved slightly higher rewards but required more compute.
- **DQN** was:
  - Fast to train
  - Lightweight on resources
  - Easier to tune
- For **real-world use cases** like delivery or robotics, DQN’s balance of **efficiency** and **effectiveness** makes it ideal.
- Future enhancements (e.g., Double DQN, Prioritized Replay) can further improve DQN.

---

## ✅ Conclusion

> **DQN** emerged as the most pragmatic choice offering fast convergence, minimal tuning, and competitive performance.  
> While **PPO** remains top for performance benchmarks, **DQN’s simplicity and robustness** make it a standout for scalable and real-time deployment in environments like seed delivery, robotics, or logistics management.

---
