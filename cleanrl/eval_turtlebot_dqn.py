import argparse

import torch

import turtlebot_minigrid_env  # noqa: F401
from dqn import QNetwork, make_env
from evals.dqn_eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DQN model on Turtlebot MiniGrid")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a .cleanrl_model file")
    parser.add_argument("--env-id", type=str, default=turtlebot_minigrid_env.ENV_ID)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render the environment visually")
    parser.add_argument("--run-name", type=str, default="turtlebot-dqn-eval")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    returns = evaluate(
        model_path=args.model_path,
        make_env=make_env,
        env_id=args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=args.run_name,
        Model=QNetwork,
        device=device,
        epsilon=args.epsilon,
        capture_video=args.capture_video,
        render=args.render,
    )

    mean_return = float(sum(returns) / len(returns)) if returns else 0.0
    print(f"Evaluated {len(returns)} episodes")
    print(f"Mean return: {mean_return:.4f}")
    print(f"Returns: {returns}")


if __name__ == "__main__":
    main()
