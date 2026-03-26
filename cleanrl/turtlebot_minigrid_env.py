from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random as rand


ENV_ID = "TurtlebotMiniGrid-StaticObstacles-v0"
ENV_ID_DYNAMIC = "TurtlebotMiniGrid-DynamicObstacles-v0"


@dataclass(frozen=True)
class TurtlebotMiniGridConfig:
    grid_size: int = 20
    max_steps: int = 300
    step_penalty: float = -0.01
    collision_penalty: float = -0.10
    goal_reward: float = 1.0
    num_moving_obstacles: int = 0
    moving_obstacle_freq: int = 3  # move every N steps


class TurtlebotMiniGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__( # FLAG
        self,
        render_mode: Optional[str] = None,
        grid_size: int = 20,
        max_steps: int = 300,
        step_penalty: float = -0.01,
        collision_penalty: float = -0.55,
        goal_reward: float = 1.2,
        num_moving_obstacles: int = 3,
        moving_obstacle_freq: int = 2,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.config = TurtlebotMiniGridConfig(
            grid_size=grid_size,
            max_steps=max_steps,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            goal_reward=goal_reward,
            num_moving_obstacles=num_moving_obstacles,
            moving_obstacle_freq=moving_obstacle_freq,
        )

        self.action_space = spaces.Discrete(3)
        obs_dim = self.config.grid_size * self.config.grid_size + 8
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._orientation = 1
        self._robot_pos = (1, 1)
        self._goal_pos = (self.config.grid_size - 2, self.config.grid_size - 2)
        self._static_obstacles = self._build_static_obstacles(self.config.grid_size)
        self._obstacles = self._static_obstacles
        self._moving_obstacles: list = []  # list of [x, y, direction]
        self._step_count = 0
        self._last_collision = False  # True if robot collided last step

        self._window = None
        self._clock = None

    @staticmethod
    def _build_static_obstacles(grid_size: int):
        obstacles = set()
        center = grid_size // 2

        for y in range(2, grid_size - 2):
            if y not in (center - 1, center + 1):
                obstacles.add((center, y))

        for x in range(2, grid_size - 2):
            if x not in (center - 2, center + 2):
                obstacles.add((x, center))

        obstacles.update(
            {
                (2, grid_size - 3),
                (3, grid_size - 3),
                (grid_size - 3, 2),
                (grid_size - 4, 2),
            }
        )
        return frozenset(obstacles)

    @staticmethod
    def _heading_to_delta(orientation: int):
        if orientation == 0:
            return (0, -1)
        if orientation == 1:
            return (1, 0)
        if orientation == 2:
            return (0, 1)
        return (-1, 0)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size

    def _is_static_blocked(self, x: int, y: int) -> bool:
        """Check walls and static obstacles only (used for moving obstacle pathfinding)."""
        if not self._in_bounds(x, y):
            return True
        return (x, y) in self._static_obstacles

    def _is_blocked(self, x: int, y: int) -> bool:
        """Check walls, static obstacles, and moving obstacles."""
        if not self._in_bounds(x, y):
            return True
        if (x, y) in self._static_obstacles:
            return True
        return any(m[0] == x and m[1] == y for m in self._moving_obstacles)

    def _spawn_moving_obstacles(self) -> list:
        """Place moving obstacles randomly, avoiding walls, static obstacles, robot start, goal."""
        forbidden = set(self._static_obstacles)
        forbidden.add((1, 1))  # robot start
        forbidden.add((self.config.grid_size - 2, self.config.grid_size - 2))  # goal
        # also keep a safety margin of 1 cell around robot start
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                forbidden.add((1 + dx, 1 + dy))

        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random.default_rng()
        candidates = [
            (x, y)
            for x in range(1, self.config.grid_size - 1)
            for y in range(1, self.config.grid_size - 1)
            if (x, y) not in forbidden
        ]
        n = min(self.config.num_moving_obstacles, len(candidates))
        indices = rng.choice(len(candidates), size=n, replace=False)
        obstacles = []
        for i in indices:
            x, y = candidates[int(i)]
            direction = int(rng.integers(0, 4))
            obstacles.append([x, y, direction])
        return obstacles

    def _step_moving_obstacles(self) -> None:
        """
        Move each moving obstacle forward; bounce off walls/static obstacles.
        
        """
        if self._step_count % self.config.moving_obstacle_freq != 0:
            return
        for mob in self._moving_obstacles:
            dx, dy = self._heading_to_delta(mob[2])
            nx, ny = mob[0] + dx, mob[1] + dy
            if self._is_static_blocked(nx, ny) or rand.random() < 0.3:
                # bounce: pick a random new direction that is free
                free_dirs = [
                    d for d in range(4)
                    if not self._is_static_blocked(
                        mob[0] + self._heading_to_delta(d)[0],
                        mob[1] + self._heading_to_delta(d)[1],
                    )
                ]
                if free_dirs:
                    mob[2] = int(self.np_random.choice(free_dirs))
            else:
                mob[0], mob[1] = nx, ny

    def _distance_to_goal(self) -> int:
        return abs(self._robot_pos[0] - self._goal_pos[0]) + abs(self._robot_pos[1] - self._goal_pos[1])

    VISIBILITY_RADIUS: int = 7

    def _get_observation(self) -> np.ndarray:
        grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        for ox, oy in self._static_obstacles:
            grid[oy, ox] = 1.0

        # Moving obstacles shown as 0.5
        for mob in self._moving_obstacles:
            grid[mob[1], mob[0]] = 0.5

        gx, gy = self._goal_pos
        rx, ry = self._robot_pos
        grid[gy, gx] = 0.66
        grid[ry, rx] = 0.33

        # Apply visibility mask: hide cells outside the radius
        ys, xs = np.ogrid[0:self.config.grid_size, 0:self.config.grid_size]
        dist = np.sqrt((xs - rx) ** 2 + (ys - ry) ** 2)
        grid[dist > self.VISIBILITY_RADIUS] = 0.0

        orientation_one_hot = np.zeros(4, dtype=np.float32)
        orientation_one_hot[self._orientation] = 1.0

        norm = float(self.config.grid_size - 1)
        state = np.array(
            [
                rx / norm,
                ry / norm,
                gx / norm,
                gy / norm,
            ],
            dtype=np.float32,
        )

        return np.concatenate([grid.flatten(), state, orientation_one_hot], dtype=np.float32)

    def _get_info(self):
        return {
            "robot_x": self._robot_pos[0],
            "robot_y": self._robot_pos[1],
            "orientation": self._orientation,
            "distance_to_goal": self._distance_to_goal(),
            "collision": self._last_collision,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._robot_pos = (1, 1)
        self._goal_pos = (self.config.grid_size - 2, self.config.grid_size - 2)
        self._orientation = 1
        self._step_count = 0
        self._moving_obstacles = self._spawn_moving_obstacles()
        self._last_collision = False
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        reward = self.config.step_penalty
        terminated = False
        truncated = False
        self._last_collision = False

        old_distance = self._distance_to_goal()

        if action == 0:
            self._orientation = (self._orientation - 1) % 4
        elif action == 1:
            self._orientation = (self._orientation + 1) % 4
        else:
            dx, dy = self._heading_to_delta(self._orientation)
            nx = self._robot_pos[0] + dx
            ny = self._robot_pos[1] + dy
            if self._is_blocked(nx, ny):
                reward += self.config.collision_penalty
                self._last_collision = True
            else:
                self._robot_pos = (nx, ny)

        new_distance = self._distance_to_goal()
        reward += 0.01 * (old_distance - new_distance) # FLAG - reward for getting closer to the goal

        if self._robot_pos == self._goal_pos:
            reward += self.config.goal_reward
            terminated = True

        self._step_count += 1
        if self._step_count >= self.config.max_steps:
            truncated = True

        # Move dynamic obstacles after robot has acted
        self._step_moving_obstacles()

        # Check if a moving obstacle has walked onto the robot
        if any(m[0] == self._robot_pos[0] and m[1] == self._robot_pos[1] for m in self._moving_obstacles):
            reward += self.config.collision_penalty
            self._last_collision = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self):
        frame = self._render_rgb_array(scale=32)
        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            import pygame

            if self._window is None:
                pygame.init()
                pygame.display.init()
                self._window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
                pygame.display.set_caption("Turtlebot MiniGrid")
            if self._clock is None:
                self._clock = pygame.time.Clock()

            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            self._window.blit(surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])

    def _render_rgb_array(self, scale: int = 32):
        size = self.config.grid_size
        img = np.ones((size * scale, size * scale, 3), dtype=np.uint8) * 255

        for y in range(size):
            for x in range(size):
                y0, y1 = y * scale, (y + 1) * scale
                x0, x1 = x * scale, (x + 1) * scale
                img[y0:y1, x0:x1] = (245, 245, 245)
                if (x, y) in self._static_obstacles:
                    img[y0:y1, x0:x1] = (60, 60, 60)

        # Draw moving obstacles in orange
        for mob in self._moving_obstacles:
            mx, my = mob[0], mob[1]
            my0, my1 = my * scale, (my + 1) * scale
            mx0, mx1 = mx * scale, (mx + 1) * scale
            img[my0:my1, mx0:mx1] = (230, 120, 20)

        gx, gy = self._goal_pos
        rx, ry = self._robot_pos

        gy0, gy1 = gy * scale, (gy + 1) * scale
        gx0, gx1 = gx * scale, (gx + 1) * scale
        img[gy0:gy1, gx0:gx1] = (70, 200, 70)

        ry0, ry1 = ry * scale, (ry + 1) * scale
        rx0, rx1 = rx * scale, (rx + 1) * scale
        img[ry0:ry1, rx0:rx1] = (60, 130, 255)

        cx = rx0 + scale // 2
        cy = ry0 + scale // 2
        dx, dy = self._heading_to_delta(self._orientation)
        tip_x = int(cx + dx * (scale * 0.35))
        tip_y = int(cy + dy * (scale * 0.35))
        rr = max(2, scale // 8)

        for yy in range(max(0, tip_y - rr), min(img.shape[0], tip_y + rr + 1)):
            for xx in range(max(0, tip_x - rr), min(img.shape[1], tip_x + rr + 1)):
                if (xx - tip_x) ** 2 + (yy - tip_y) ** 2 <= rr * rr:
                    img[yy, xx] = (255, 255, 255)

        for k in range(0, size * scale, scale):
            img[:, k : k + 1] = (200, 200, 200)
            img[k : k + 1, :] = (200, 200, 200)

        return img

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None


def register_env() -> None:
    if ENV_ID not in gym.envs.registry:
        gym.register(
            id=ENV_ID,
            entry_point="turtlebot_minigrid_env:TurtlebotMiniGridEnv",
            max_episode_steps=TurtlebotMiniGridConfig.max_steps,
        )

    if ENV_ID_DYNAMIC not in gym.envs.registry:
        gym.register(
            id=ENV_ID_DYNAMIC,
            entry_point="turtlebot_minigrid_env:TurtlebotMiniGridEnv",
            kwargs={"num_moving_obstacles": 5, "moving_obstacle_freq": 3},
            max_episode_steps=TurtlebotMiniGridConfig.max_steps,
        )


register_env()


if __name__ == "__main__":
    env = gym.make(ENV_ID_DYNAMIC, render_mode="human") # FLAG
    obs, info = env.reset(seed=0)
    _ = (obs, info)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        env.render()

    env.close()
