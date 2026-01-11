import gymnasium as gym
import imageio
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="rgb_array")
model = PPO.load("models/ppo_lunarlander")

frames = []
obs, _ = env.reset()

for _ in range(600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    frames.append(env.render())
    if done or truncated:
        break

env.close()

imageio.mimsave("videos/lunarlander_solved.mp4", frames, fps=30, macro_block_size=1)
