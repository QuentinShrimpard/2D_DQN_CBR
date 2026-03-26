
import gymnasium as gym
import numpy as np

def make_env():
    env = gym.make("CartPole-v1")
    return env

envs = gym.vector.SyncVectorEnv([make_env for _ in range(1)])
obs, _ = envs.reset()
terminated = False
truncated = False
steps = 0
while not (terminated or truncated) and steps < 200:
    actions = np.array([envs.single_action_space.sample()])
    obs, rewards, terminations, truncations, infos = envs.step(actions)
    terminated = any(terminations)
    truncated = any(truncations)
    steps += 1

print(f"Terminated: {terminated}, Truncated: {truncated}")
print(f"Infos keys: {infos.keys()}")
if terminated or truncated:
    print(f"Unwrapped keys: {infos.keys()}")
