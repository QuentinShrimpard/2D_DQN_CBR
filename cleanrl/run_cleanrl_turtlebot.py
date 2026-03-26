import argparse
import os
import runpy
import sys

import turtlebot_minigrid_env  # noqa: F401


def _contains_env_flag(args_list):
    return "--env-id" in args_list or "--env_id" in args_list


def main():
    parser = argparse.ArgumentParser(description="Run any CleanRL algorithm on the Turtlebot MiniGrid environment")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn.py",
        # default="dqn_cbr.py",
        help="CleanRL algorithm script in the current folder (e.g. dqn.py, ppo.py)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=turtlebot_minigrid_env.ENV_ID,
        help="Gymnasium env id to inject if the algo supports --env-id",
    )

    args, passthrough = parser.parse_known_args()

    script_path = os.path.join(os.path.dirname(__file__), args.algo)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Algorithm script not found: {script_path}")

    forwarded = [script_path]
    if not _contains_env_flag(passthrough):
        forwarded += ["--env-id", args.env_id]
    forwarded += passthrough

    sys.argv = forwarded
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
