
---

# 🧠 train.py (core training script)

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v2", n_envs=8)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=800_000)
model.save("models/ppo_lunarlander")

env.close()
