"""
Evaluation script for dqn_cbr.py checkpoints on TurtlebotMiniGrid.
"""
import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch

import turtlebot_minigrid_env  # noqa: F401
from dqn_cbr import FlatDQNNet, SuperDQNNet, is_minigrid_env, process_raw_obs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a D2CBRL-DQN model on Turtlebot MiniGrid")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a .pth checkpoint from dqn_cbr.py")
    parser.add_argument("--env-id", type=str, default=None, help="Env ID (default: read from checkpoint)")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon for greedy action selection")
    parser.add_argument("--render", action="store_true", help="Render the environment visually")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def evaluate_cbr(model_path: str, env_id: str, eval_episodes: int, epsilon: float, device: torch.device, render: bool):
    # ---- Load checkpoint ----
    ckpt = torch.load(model_path, map_location=device)
    obs_shape = ckpt["obs_shape"]
    q_state_dict = ckpt["q_net"]
    cfg_dict = ckpt.get("config", {})

    # Infer n_actions from the last layer of saved weights
    last_key = [k for k in q_state_dict if "weight" in k][-1]
    n_actions = q_state_dict[last_key].shape[0]

    # Detect architecture: SuperDQNNet (image) vs FlatDQNNet (flat obs)
    is_image = any("conv" in k for k in q_state_dict)

    if is_image:
        # obs_shape is (H, W, C) for image envs
        h, w, c = obs_shape
        agent_view_size = cfg_dict.get("agent_view_size", h)
        model = SuperDQNNet(n_actions, sensory_size=agent_view_size, channel=c).to(device)
    else:
        obs_dim = int(np.prod(obs_shape))
        model = FlatDQNNet(obs_dim, n_actions).to(device)
    model.load_state_dict(q_state_dict)
    model.eval()

    # ---- Build env ----
    render_mode = "human" if render else None
    if is_image:
        from minigrid.wrappers import ViewSizeWrapper, ImgObsWrapper
        agent_view_size = cfg_dict.get("agent_view_size", 7)
        max_steps = cfg_dict.get("max_steps", 300)
        env = gym.make(env_id, render_mode=render_mode, max_steps=max_steps)
        env = ViewSizeWrapper(env, agent_view_size=agent_view_size)
        env = ImgObsWrapper(env)
    else:
        env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    episodic_returns = []
    obs_raw, _ = env.reset()
    obs = process_raw_obs(obs_raw, is_image)
    episode_reward = 0.0

    while len(episodic_returns) < eval_episodes:
        if render:
            env.render()
            time.sleep(1.0 / 8)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                if is_image:
                    # obs is HWC uint8 → CHW float tensor
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = model(obs_t)
            action = int(torch.argmax(q_values, dim=1).item())

        obs_raw, reward, terminated, truncated, info = env.step(action)
        obs = process_raw_obs(obs_raw, is_image)
        episode_reward += float(reward)

        if terminated or truncated:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={episode_reward:.4f}")
            episodic_returns.append(episode_reward)
            episode_reward = 0.0
            obs_raw, _ = env.reset()
            obs = process_raw_obs(obs_raw, is_image)

    env.close()
    return episodic_returns


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # If env_id not specified, read it from the checkpoint
    env_id = args.env_id
    if env_id is None:
        ckpt_peek = torch.load(args.model_path, map_location="cpu")
        env_id = ckpt_peek.get("config", {}).get("env_id", turtlebot_minigrid_env.ENV_ID)
        print(f"[INFO] Using env_id from checkpoint: {env_id}")

    returns = evaluate_cbr(
        model_path=args.model_path,
        env_id=env_id,
        eval_episodes=args.eval_episodes,
        epsilon=args.epsilon,
        device=device,
        render=args.render,
    )

    mean_return = float(sum(returns) / len(returns)) if returns else 0.0
    print(f"\nEvaluated {len(returns)} episodes")
    print(f"Mean return : {mean_return:.4f}")
    print(f"Returns     : {[round(r, 4) for r in returns]}")


if __name__ == "__main__":
    main()
