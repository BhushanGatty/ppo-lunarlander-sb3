import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v2")
model = PPO.load("models/ppo_lunarlander")

mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(mean, std)
