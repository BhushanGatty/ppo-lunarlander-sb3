# ppo-lunarlander-sb3
PPO reinforcement learning agent trained on LunarLander-v2 using Stable-Baselines3.


# PPO LunarLander-v2 (Stable-Baselines3)

This repository contains a Proximal Policy Optimization (PPO) agent trained on the LunarLander-v2 environment using Stable-Baselines3.

The project demonstrates a complete deep reinforcement learning workflow including training, evaluation, visualization, and model publishing.

---

## 🚀 Environment

- LunarLander-v2 (Gymnasium / Box2D)

---

## 🧠 Algorithm

- Proximal Policy Optimization (PPO)
- Actor-Critic MLP policy
- Implemented using Stable-Baselines3 (PyTorch)

---

## 🔧 Training Details

- Total timesteps: ~800,000
- Vectorized environments: 8
- Learning rate: 3e-4
- Discount factor (γ): 0.99

---

## 📊 Results

Final evaluation over 20 episodes:

**Mean reward:** 253 ± 25  

This exceeds the standard solved threshold for LunarLander-v2.

---

## 🎥 Demo

Trained agent rollout:

