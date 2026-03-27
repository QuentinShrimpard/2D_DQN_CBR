# Turtlebot MiniGrid with CleanRL

This mini-framework adds a minigrid-style Gymnasium environment for a robot and allows for easy execution of CleanRL scripts on it.

## Useful Files

- `cleanrl/turtlebot_minigrid_env.py`: 2D environment (grid, orientation, static and dynamic obstacles).
- `cleanrl/run_cleanrl_turtlebot.py`: Generic launcher to execute a CleanRL script on this environment.
- `cleanrl/eval_turtlebot_dqn.py`: Evaluation of a saved `dqn.py` model (`.cleanrl_model`).
- `cleanrl/eval_turtlebot_cbr.py`: Evaluation of a saved `dqn_cbr.py` model (`.pth`).

## Registered Environments

Importing `turtlebot_minigrid_env` automatically registers two Gymnasium IDs:

- `TurtlebotMiniGrid-StaticObstacles-v0`
- `TurtlebotMiniGrid-DynamicObstacles-v0`

The dynamic version uses moving obstacles (`num_moving_obstacles=5`, `moving_obstacle_freq=3`).

## State and Action Space

- **Discrete Actions** (`Discrete(3)`):
  - `0`: turn left
  - `1`: turn right
  - `2`: move forward
- **Observation** (`Box` of size `grid_size * grid_size + 8`):
  - flattened grid (static obstacles, moving obstacles, robot, goal)
  - visibility masking around the robot (`VISIBILITY_RADIUS = 7`)
  - compact state:
    - normalized robot position `(x, y)`
    - normalized goal position `(x, y)`
    - one-hot orientation (N/E/S/W)

## Rewards

In the current implementation (`TurtlebotMiniGridEnv.__init__`):

- step penalty: `-0.01`
- collision penalty: `-0.55`
- progress bonus: `+0.01 * (distance_old - distance_new)`
- goal reward: `+1.2` and episode termination

## Run a DQN training

From the root of the repository:

```bash

python cleanrl/run_cleanrl_turtlebot.py --algo dqn.py --total-timesteps 200000 --learning-starts 1000 --buffer-size 50000 --batch-size 128 --save-model

```

By default, the launcher injects `--env-id TurtlebotMiniGrid-StaticObstacles-v0` if none is provided


## Run a D2CBRL training

```bash

python cleanrl/run_cleanrl_turtlebot.py --algo dqn_cbr.py --env-id TurtlebotMiniGrid-DynamicObstacles-v0

```


## Evaluate a DQN model (`dqn.py`)

```bash

python cleanrl/eval_turtlebot_dqn.py --model-path runs/<run_name>/dqn.cleanrl_model --eval-episodes 20 --capture-video

```


## Evaluate a D2CBRL model (`dqn_cbr.py`)

```bash

python cleanrl/eval_turtlebot_cbr.py --model-path checkpoints/<exp>/model.pth --eval-episodes 20 --render

```


## Notes

- Video rendering works with `render_mode="rgb_array"` (useful for`--capture-video`).

- Interactive rendering is available with `--render` (Pygame window).

- We removed the other cleanRL scripts but they should work on our framework if they accept `--env-id` and a discrete action space.
