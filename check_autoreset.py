
import gymnasium as gym
import numpy as np

# Check if Autoreset wrapper exists and how it works
if hasattr(gym.wrappers, "Autoreset"):
    print("Autoreset found")
    # Wrap standard env
    def make_env():
        env = gym.make("CartPole-v1", max_episode_steps=2)
        env = gym.wrappers.Autoreset(env)
        return env

    env = make_env()
    obs, info = env.reset()
    print(f"Reset info: {info.keys()}")

    # Step 1
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Step 1 trunc={truncated}, term={terminated}, info keys={info.keys()}")

    # Step 2 (should truncate here)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Step 2 trunc={truncated}, term={terminated}, info keys={info.keys()}")

    if "final_observation" not in info:
        print("Final observation MISSING in Autoreset wrapper!")
    else:
        print("Final observation present.")

else:
    print("Autoreset NOT found")

# Check vector env behavior with manual Autoreset
print("\nChecking SyncVectorEnv behaviour...")
envs = gym.vector.SyncVectorEnv([make_env])
obs, info = envs.reset()

actions = np.array([env.action_space.sample()])
obs, rewards, terminations, truncations, infos = envs.step(actions) # Step 1
obs, rewards, terminations, truncations, infos = envs.step(actions) # Step 2 (truncated)

print(f"Vector Step 2 trunc={truncations}, infos keys={infos.keys()}")
if "final_observation" in infos:
    print("Vector final_observation present.")
else:
    print("Vector final_observation MISSING!")
