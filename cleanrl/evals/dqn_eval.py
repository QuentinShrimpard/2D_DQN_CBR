import random
import time
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False,
    render: bool = False,
):
    if render:
        return _evaluate_render(model_path, env_id, eval_episodes, Model, device, epsilon)

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None or not isinstance(info, dict):
                    continue
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


def _evaluate_render(
    model_path: str,
    env_id: str,
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device,
    epsilon: float,
):
    """Run evaluation with a human-rendered window (non-vectorized)."""
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Build a fake single-env wrapper so Model(envs) works
    class _FakeVecEnv:
        def __init__(self, e):
            self.single_observation_space = e.observation_space
            self.single_action_space = e.action_space
            self.observation_space = e.observation_space
            self.action_space = e.action_space

    model = Model(_FakeVecEnv(env)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episodic_returns = []
    obs, _ = env.reset()
    episode_reward = 0.0

    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(torch.Tensor(obs).unsqueeze(0).to(device))
            action = int(torch.argmax(q_values, dim=1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        episode_reward += float(reward)

        if terminated or truncated:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={episode_reward:.4f}")
            episodic_returns.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()

    env.close()
    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn import QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth")
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
    )
