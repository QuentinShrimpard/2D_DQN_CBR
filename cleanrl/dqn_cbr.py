"""
D2CBRL training
"""

import os
import sys
import csv
import random
import pickle
import argparse
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym

try:
    import minigrid  # noqa: F401  (registers MiniGrid env IDs)
    from minigrid.wrappers import ViewSizeWrapper, ImgObsWrapper
    _HAS_MINIGRID = True
except ImportError:
    _HAS_MINIGRID = False

try:
    import turtlebot_minigrid_env  # noqa: F401  (registers TurtlebotMiniGrid env IDs)
except ImportError:
    pass

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import trange


# =============================
# Helpers
# =============================
def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_id_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"


def is_minigrid_env(env_id: str) -> bool:
    """Detect whether env_id refers to a MiniGrid environment."""
    return env_id.startswith("MiniGrid-")


# =============================
# Config
# =============================
@dataclass
class Config:
    # ---- ENV ----
    env_id: str = "MiniGrid-LavaGapS7-v0"
    seed: int = 0
    agent_view_size: int = 7
    total_timesteps: int = 600_000
    render_mode: str = "rgb_array"  # "human" or "rgb_array"
    max_steps: int = 300  # per episode

    # ---- DQN ----
    gamma: float = 0.99
    lr: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 2_000
    train_freq: int = 4
    target_update_interval: int = 1_000

    # ---- Epsilon-greedy ----
    eps_start: float = 1.0
    eps_final: float = 0.1
    eps_fraction: float = 0.5

    # ---- Shaping ----
    step_cost: float = 0.01
    turn_cost: float = 0.001
    turn_actions: tuple = (0, 1)  # left/right (after action subset mapping)

    # ---- Actions ----
    # LavaGap action space is Discrete(7) but only left/right/forward are useful:
    # MiniGrid.Actions: left=0, right=1, forward=2
    allowed_actions: list | None = (0, 1, 2)

    # ---- Vanilla-style reseed ----
    reseed_every_n_steps: int = 0  # if >0, reseed env reset seed every N global steps

    # ---- Output ----
    out_dir: str = "./logs_csv/train/d2cbrl_dqn_lavagap"
    ckpt_path: str = "./checkpoints/d2cbrl_dqn_lavagap/d2cbrl_dqn_lavagap.pth"
    case_base_path: str = "./checkpoints/d2cbrl_dqn_lavagap/case_bases.pkl"
    tqdm_refresh: int = 200

    # =============================
    # D2CBRL params
    # =============================
    p_cbr: float = 0.4
    min_trust: float = 0.3
    max_case_dist: float = 1.0

    # window-based “successful experience” detection
    window_size: int = 10
    window_stride: int = 1
    min_windows_for_stats: int = 5
    win_delta: float = 0.02
    win_abs_floor: float = 0.0

    # negative (bad) case base veto
    neg_min_badness: float = 0.5


# =============================
# Wrappers
# =============================
class StepCostWrapper(gym.Wrapper):
    def __init__(self, env, step_cost=0.01, turn_cost=0.001, turn_actions=(0, 1)):
        super().__init__(env)
        self.step_cost = float(step_cost)
        self.turn_cost = float(turn_cost)
        self.turn_actions = set(int(a) for a in turn_actions)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) - self.step_cost
        if int(action) in self.turn_actions:
            reward -= self.turn_cost
        return obs, reward, terminated, truncated, info


class SubsetActionWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError(f"Only supports Discrete action spaces, got {type(env.action_space)}")

        self.allowed_actions = [int(a) for a in allowed_actions]
        n = env.action_space.n
        for a in self.allowed_actions:
            if not (0 <= a < n):
                raise ValueError(f"Action {a} out of range for base action space Discrete({n})")

        self.action_space = gym.spaces.Discrete(len(self.allowed_actions))

    def action(self, a):
        return self.allowed_actions[int(a)]


# =============================
# Obs helpers (uint8 -> float CHW, no /255)
# =============================
def obs_to_hwc_uint8(obs) -> np.ndarray:
    """
    Compatible with:
    - ImgObsWrapper: obs is dict with key "image"
    - Already-image obs: obs is np.ndarray (H,W,C)
    """
    if isinstance(obs, dict):
        if "image" not in obs:
            raise KeyError(f"obs is dict but missing 'image' key. keys={list(obs.keys())}")
        return np.asarray(obs["image"], dtype=np.uint8)

    if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[1], dict):
        return obs_to_hwc_uint8(obs[0])

    arr = np.asarray(obs)
    if arr.ndim != 3:
        raise ValueError(f"Expected obs as HWC image (3D), got shape={arr.shape} type={type(obs)}")
    return arr.astype(np.uint8, copy=False)


def obs_to_tensor(obs_array: np.ndarray, device: str, is_image: bool = True) -> th.Tensor:
    x = th.as_tensor(obs_array, device=device)
    if is_image:
        return x.permute(2, 0, 1).unsqueeze(0).float()
    else:
        return x.unsqueeze(0).float()


def process_raw_obs(obs, is_image: bool) -> np.ndarray:
    """Convert raw env observation to numpy array.

    - Image envs (MiniGrid): returns HWC uint8
    - Flat envs (TurtlebotMiniGrid): returns 1D float32
    """
    if is_image:
        return obs_to_hwc_uint8(obs)
    else:
        if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[1], dict):
            obs = obs[0]
        if isinstance(obs, dict):
            for key in ("image", "observation"):
                if key in obs:
                    obs = obs[key]
                    break
        return np.asarray(obs, dtype=np.float32).flatten()


# =============================
# Replay Buffer
# =============================
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, device: str, is_image: bool = True):
        self.device = device
        self.capacity = int(capacity)
        self.is_image = is_image

        dtype = np.uint8 if is_image else np.float32
        self.obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        if self.is_image:
            obs = th.as_tensor(self.obs[idx], device=self.device).permute(0, 3, 1, 2).float()
            next_obs = th.as_tensor(self.next_obs[idx], device=self.device).permute(0, 3, 1, 2).float()
        else:
            obs = th.as_tensor(self.obs[idx], device=self.device).float()
            next_obs = th.as_tensor(self.next_obs[idx], device=self.device).float()
        actions = th.as_tensor(self.actions[idx], device=self.device)
        rewards = th.as_tensor(self.rewards[idx], device=self.device)
        dones = th.as_tensor(self.dones[idx].astype(np.float32), device=self.device)

        return obs, actions, rewards, next_obs, dones


# =============================
# Q Network
# =============================
class SuperDQNNet(nn.Module):
    def __init__(self, action_dim: int, sensory_size: int, channel: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        with th.no_grad():
            dummy_input = th.zeros(1, channel, sensory_size, sensory_size)
            conv_out = self.conv2(F.relu(self.conv1(dummy_input)))
            conv_out_size = conv_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FlatDQNNet(nn.Module):
    """MLP Q-network for flat (1-D) observation spaces."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


# =============================
# Case base logic
# =============================
class Case:
    # -------- Positive (pos) trust updates --------
    INCREMENT_SUCCESS = 0.1
    DECREMENT_UNSUCCESS = 0.1
    DECREMENT_SUCCESS_NO_CASE = 0.1
    DEL_THRESHOLD = 0.1
    MATCH_THRESHOLD = 1.0

    # -------- Negative (neg) trust updates (badness) --------
    NEG_INC_ON_FAIL = 0.2
    NEG_DEC_ON_SUCCESS = 0.1
    NEG_DEC_ON_SUCCESS_NO_CASE = 0.05
    NEG_DEL_THRESHOLD = 0.1

    def __init__(
        self,
        problem_chw_f32,
        solution: int,
        trust_value=0.5,
        reward=None,
        reward_mean=0.0,
        is_negative=False,
    ):
        self.problem = np.asarray(problem_chw_f32, dtype=np.float32)  # (C,H,W)
        self.solution = int(solution)
        self.trust_value = float(trust_value)
        self.reward = reward
        self.reward_mean = float(reward_mean)
        self.is_negative = bool(is_negative)

    @staticmethod
    def _dist(p1, p2) -> float:
        return float(np.linalg.norm(p1.reshape(-1) - p2.reshape(-1)))

    @staticmethod
    def retrieve(case_base, current_problem, min_trust=0.0, max_dist=1.0):
        best, best_d = None, float("inf")
        for c in case_base:
            if c.trust_value < min_trust:
                continue
            d = Case._dist(c.problem, current_problem)
            if d < best_d:
                best, best_d = c, d
        return best if best_d < max_dist else None

    @staticmethod
    def retrieve_negative(case_base_neg, current_problem, max_dist=1.0):
        best, best_d = None, float("inf")
        for c in case_base_neg:
            d = Case._dist(c.problem, current_problem)
            if d < best_d:
                best, best_d = c, d
        return best if best_d < max_dist else None

    # ---------------- POSITIVE revise/retain ----------------
    @staticmethod
    def revise(case_base, temporary_case_base, successful_episode: bool):
        for case in case_base:
            case_found = False
            for temp_case in temporary_case_base:
                d = Case._dist(case.problem, temp_case.problem)
                if d < Case.MATCH_THRESHOLD and temp_case.solution == case.solution:
                    case_found = True
                    break

            if case_found:
                if successful_episode:
                    case.trust_value += Case.INCREMENT_SUCCESS
                else:
                    case.trust_value -= Case.DECREMENT_UNSUCCESS
            else:
                if successful_episode:
                    case.trust_value -= Case.DECREMENT_SUCCESS_NO_CASE

            case.trust_value = max(0.0, min(1.0, case.trust_value))

    @staticmethod
    def retain(case_base, own_temp_case_base, successful_episode: bool, threshold=DEL_THRESHOLD):
        temp_case_dict = {}
        for temp_case in own_temp_case_base:
            key = tuple(np.round(temp_case.problem.reshape(-1), 4))
            temp_case_dict[key] = temp_case

        if successful_episode:
            for _, new_case in temp_case_dict.items():
                found = False
                new_key = tuple(np.round(new_case.problem.reshape(-1), 4))
                for i, old_case in enumerate(case_base):
                    old_key = tuple(np.round(old_case.problem.reshape(-1), 4))
                    if old_key == new_key:
                        found = True
                        if new_case.reward_mean >= old_case.reward_mean:
                            case_base[i] = new_case
                        break
                if not found:
                    case_base.append(new_case)

        case_base = [c for c in case_base if c.trust_value >= threshold]
        return case_base

    # ---------------- NEGATIVE revise/retain ----------------
    @staticmethod
    def revise_negative(case_base_neg, temp_case_base_neg, successful_episode: bool):
        for case in case_base_neg:
            case_found = False
            for temp_case in temp_case_base_neg:
                d = Case._dist(case.problem, temp_case.problem)
                if d < Case.MATCH_THRESHOLD and temp_case.solution == case.solution:
                    case_found = True
                    break

            if case_found:
                if successful_episode:
                    case.trust_value -= Case.NEG_DEC_ON_SUCCESS
                else:
                    case.trust_value += Case.NEG_INC_ON_FAIL
            else:
                if successful_episode:
                    case.trust_value -= Case.NEG_DEC_ON_SUCCESS_NO_CASE

            case.trust_value = max(0.0, min(1.0, case.trust_value))

    @staticmethod
    def retain_negative(case_base_neg, temp_case_base_neg, successful_episode: bool, threshold=NEG_DEL_THRESHOLD):
        temp_case_dict = {}
        for temp_case in temp_case_base_neg:
            key = (tuple(np.round(temp_case.problem.reshape(-1), 4)), int(temp_case.solution))
            temp_case_dict[key] = temp_case

        if not successful_episode:
            for (kprob, ksol), new_case in temp_case_dict.items():
                found = False
                for i, old_case in enumerate(case_base_neg):
                    old_key = (tuple(np.round(old_case.problem.reshape(-1), 4)), int(old_case.solution))
                    if old_key == (kprob, ksol):
                        found = True
                        if new_case.reward_mean >= old_case.reward_mean:
                            case_base_neg[i] = new_case
                        break
                if not found:
                    case_base_neg.append(new_case)

        case_base_neg = [c for c in case_base_neg if c.trust_value >= threshold]
        return case_base_neg


# =============================
# CSV Logger
# =============================
class CSVLogger:
    def __init__(self, out_dir: str, run_id: str, cfg: Config):
        os.makedirs(out_dir, exist_ok=True)
        self.rew_dir = os.path.join(out_dir, f"rewsteps_{cfg.seed}_{cfg.env_id.replace(':', '_')}")
        self.loss_dir = os.path.join(out_dir, f"loss_{cfg.seed}_{cfg.env_id.replace(':', '_')}")
        os.makedirs(self.rew_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)

        self.rewards_csv = os.path.join(self.rew_dir, f"train_rewards_{run_id}.csv")
        self.loss_csv = os.path.join(self.loss_dir, f"train_loss_{run_id}.csv")

        with open(self.rewards_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "agent", "global_step", "episode_return", "episode_length", "seed_used", "success_flag"]
            )

        with open(self.loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["global_step", "agent", "loss"])

    def log_episode(self, episode, agent, step, ep_return, ep_len, seed_used, success_flag):
        with open(self.rewards_csv, "a", newline="") as f:
            csv.writer(f).writerow([episode, agent, step, ep_return, ep_len, int(seed_used), int(success_flag)])

    def log_loss(self, step, agent, loss):
        with open(self.loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([step, agent, loss])


# =============================
# Environment call
# =============================
def make_env(cfg: Config):
    if is_minigrid_env(cfg.env_id):
        env = gym.make(
            cfg.env_id,
            render_mode=cfg.render_mode,
            max_steps=cfg.max_steps,
        )

        env = ViewSizeWrapper(env, agent_view_size=cfg.agent_view_size)
        env = ImgObsWrapper(env)

        if cfg.allowed_actions is not None:
            env = SubsetActionWrapper(env, cfg.allowed_actions)

        env = StepCostWrapper(env, cfg.step_cost, cfg.turn_cost, cfg.turn_actions)
    else:
        # Generic gym env (e.g. TurtlebotMiniGrid)
        env = gym.make(
            cfg.env_id,
            render_mode=cfg.render_mode,
        )
    return env


# =============================
# D2CBRL-DQN Agent
# =============================
class D2CBRL_DQNAgent:
    def __init__(self, cfg: Config, obs_shape, n_actions: int, device: str, is_image: bool = True):
        self.cfg = cfg
        self.device = device
        self.n_actions = int(n_actions)
        self.is_image = is_image

        if is_image:
            h, w, c = obs_shape
            self.view_size = int(cfg.agent_view_size)
            self.channel = int(c)
            self.q = SuperDQNNet(self.n_actions, self.view_size, self.channel).to(device)
            self.q_tgt = SuperDQNNet(self.n_actions, self.view_size, self.channel).to(device)
        else:
            obs_dim = int(np.prod(obs_shape))
            self.q = FlatDQNNet(obs_dim, self.n_actions).to(device)
            self.q_tgt = FlatDQNNet(obs_dim, self.n_actions).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.opt = th.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.buffer_size, obs_shape, device, is_image=is_image)

        self.gamma = float(cfg.gamma)
        self.batch_size = int(cfg.batch_size)
        self.learning_starts = int(cfg.learning_starts)
        self.train_freq = int(cfg.train_freq)
        self.target_update_interval = int(cfg.target_update_interval)

        # epsilon schedule
        self.epsilon = float(cfg.eps_start)
        self.eps_min = float(cfg.eps_final)

        anneal_steps = int(max(1, float(cfg.eps_fraction) * int(cfg.total_timesteps)))
        ratio = max(1e-12, min(1.0, self.eps_min / max(1e-12, self.epsilon)))
        self.eps_decay = float(ratio ** (1.0 / anneal_steps))

        self.global_step = 0

        # D2CBRL storages
        self.case_base: list[Case] = []
        self.case_base_neg: list[Case] = []
        self.episode_temp_cases: list[Case] = []
        self.episode_temp_cases_neg: list[Case] = []

        # window stats
        self.window_size = int(cfg.window_size)
        self.window_stride = int(cfg.window_stride)
        self.min_windows_for_stats = int(cfg.min_windows_for_stats)
        self.win_delta = float(cfg.win_delta)
        self.win_abs_floor = float(cfg.win_abs_floor)

        self.online_window_r: list[float] = []
        self.online_window_cases: list[Case] = []
        self.window_tick = 0
        self.global_w_r_mean = 0.0
        self.global_w_r_count = 0
        self.share_budget = 0  # single-agent no sharing

    def problem_from_obs(self, obs_array: np.ndarray) -> np.ndarray:
        if self.is_image:
            return obs_array.transpose(2, 0, 1).astype(np.float32)
        else:
            return np.asarray(obs_array, dtype=np.float32).flatten()

    def act(self, obs_array: np.ndarray, action_space: gym.spaces.Discrete):
        """
        - Negative veto applies to ALL candidate actions (random/cbr/dqn).
        - If veto triggers, ALWAYS pick a RANDOM action instead.
        """
        cur_problem = self.problem_from_obs(obs_array)

        def is_veto(proposed_a: int) -> tuple[bool, Case | None]:
            neg_case = Case.retrieve_negative(self.case_base_neg, cur_problem, max_dist=self.cfg.max_case_dist)
            if (
                neg_case is not None
                and int(neg_case.solution) == int(proposed_a)
                and float(neg_case.trust_value) >= float(self.cfg.neg_min_badness)
            ):
                return True, neg_case
            return False, None

        # (1) epsilon random
        if np.random.rand() < self.epsilon:
            proposed = int(action_space.sample())
            veto, neg = is_veto(proposed)
            if veto:
                for _ in range(10):
                    ra = int(action_space.sample())
                    if ra != proposed:
                        return ra, "veto_random", neg
                return int(action_space.sample()), "veto_random", neg
            return proposed, "random", None

        # (2) maybe CBR
        if random.random() < float(self.cfg.p_cbr):
            chosen_case = Case.retrieve(
                self.case_base,
                cur_problem,
                min_trust=self.cfg.min_trust,
                max_dist=self.cfg.max_case_dist,
            )
            if chosen_case is not None:
                proposed = int(chosen_case.solution)
                veto, neg = is_veto(proposed)
                if veto:
                    for _ in range(10):
                        ra = int(action_space.sample())
                        if ra != proposed:
                            return ra, "veto_random", neg
                    return int(action_space.sample()), "veto_random", neg
                return proposed, "cbr", chosen_case

        # (3) DQN greedy
        with th.no_grad():
            qvals = self.q(obs_to_tensor(obs_array, self.device, is_image=self.is_image))
            proposed = int(th.argmax(qvals, dim=1).item())

        veto, neg = is_veto(proposed)
        if veto:
            for _ in range(10):
                ra = int(action_space.sample())
                if ra != proposed:
                    return ra, "veto_random", neg
            return int(action_space.sample()), "veto_random", neg

        return proposed, "dqn", None

    def store(self, obs, action, reward, next_obs, done):
        self.rb.add(obs, action, reward, next_obs, done)

    def can_learn(self) -> bool:
        return self.rb.size >= self.batch_size and self.global_step >= self.learning_starts

    def learn_step(self):
        if (not self.can_learn()) or (self.global_step % self.train_freq != 0):
            return None

        o, a, r, no, d = self.rb.sample(self.batch_size)

        with th.no_grad():
            best_a = self.q(no).argmax(dim=1)
            next_q = self.q_tgt(no).gather(1, best_a[:, None]).squeeze(1)
            tgt = r + self.gamma * (1.0 - d) * next_q

        qv = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(qv, tgt)

        self.opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        if self.global_step % self.target_update_interval == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def update_epsilon(self):
        self.global_step += 1
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def add_or_update_negative_case(self, new_case: Case):
        self.episode_temp_cases_neg.append(new_case)

        new_key = (tuple(np.round(new_case.problem.reshape(-1), 4)), int(new_case.solution))
        for old in self.case_base_neg:
            old_key = (tuple(np.round(old.problem.reshape(-1), 4)), int(old.solution))
            if old_key == new_key:
                old.trust_value = min(1.0, old.trust_value + 0.05)
                if new_case.reward_mean >= old.reward_mean:
                    old.reward_mean = new_case.reward_mean
                    old.reward = new_case.reward
                return
        self.case_base_neg.append(new_case)

    def window_tick_step(self, total_steps_in_episode: int):
        self.window_tick += 1

        if self.window_tick % self.window_stride != 0:
            return
        if len(self.online_window_r) < self.window_size:
            return

        w_r = float(np.mean(self.online_window_r[-self.window_size:]))
        window_cases = self.online_window_cases[-self.window_size:]

        cnt = self.global_w_r_count
        mu_prev = self.global_w_r_mean

        eps = 1e-6
        good_window = (
            (cnt >= self.min_windows_for_stats)
            and (w_r > mu_prev + self.win_delta)
            and (w_r > self.win_abs_floor + eps)
        )

        mu_new = (mu_prev * cnt + w_r) / (cnt + 1)
        self.global_w_r_mean = mu_new
        self.global_w_r_count = cnt + 1

        if good_window:
            Case.revise(self.case_base, window_cases, successful_episode=True)
            self.case_base = Case.retain(self.case_base, window_cases, successful_episode=True)

    def episode_end_update(self, success_flag: bool):
        if len(self.episode_temp_cases) > 0:
            Case.revise(self.case_base, self.episode_temp_cases, successful_episode=success_flag)
            self.case_base = Case.retain(self.case_base, self.episode_temp_cases, successful_episode=success_flag)

        if len(self.episode_temp_cases_neg) > 0 or len(self.case_base_neg) > 0:
            Case.revise_negative(self.case_base_neg, self.episode_temp_cases_neg, successful_episode=success_flag)
            self.case_base_neg = Case.retain_negative(
                self.case_base_neg, self.episode_temp_cases_neg, successful_episode=success_flag
            )

        self.episode_temp_cases.clear()
        self.episode_temp_cases_neg.clear()
        self.online_window_r.clear()
        self.online_window_cases.clear()
        self.window_tick = 0

    def save(self, path: str, cfg: Config, obs_shape):
        ensure_parent_dir(path)
        th.save(
            {
                "config": {
                    "env_id": cfg.env_id,
                    "seed": cfg.seed,
                    "render_mode": cfg.render_mode,
                    "agent_view_size": cfg.agent_view_size,
                    "allowed_actions": cfg.allowed_actions,
                    "max_steps": cfg.max_steps,
                    "reseed_every_n_steps": cfg.reseed_every_n_steps,
                    "step_cost": cfg.step_cost,
                    "turn_cost": cfg.turn_cost,
                    "turn_actions": cfg.turn_actions,
                    "target_update_interval": cfg.target_update_interval,
                    "eps_start": cfg.eps_start,
                    "eps_final": cfg.eps_final,
                    "eps_fraction": cfg.eps_fraction,
                    "p_cbr": cfg.p_cbr,
                    "min_trust": cfg.min_trust,
                    "max_case_dist": cfg.max_case_dist,
                },
                "obs_shape": obs_shape,
                "q_net": self.q.state_dict(),
                "q_target": self.q_tgt.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )


# =============================
# Train
# =============================
def train(cfg: Config):
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    th.set_grad_enabled(True)
    random.seed(cfg.seed)
    device = "cuda" if th.cuda.is_available() else "cpu"
    print("USING DEVICE ", device)

    is_image = is_minigrid_env(cfg.env_id)

    run_id = run_id_tag()
    logger = CSVLogger(cfg.out_dir, run_id, cfg)

    env = make_env(cfg)

    def step_seed(step_idx: int) -> int:
        if cfg.reseed_every_n_steps is None or int(cfg.reseed_every_n_steps) <= 0:
            return int(cfg.seed)
        return int(cfg.seed + (int(step_idx) // int(cfg.reseed_every_n_steps)))

    seed_used = step_seed(0)
    obs_dict, info = env.reset(seed=seed_used)
    obs = process_raw_obs(obs_dict, is_image)

    obs_shape = obs.shape
    n_actions = env.action_space.n
    agent = D2CBRL_DQNAgent(cfg, obs_shape=obs_shape, n_actions=n_actions, device=device, is_image=is_image)

    episode = 0
    ep_return = 0.0
    ep_len = 0

    cnt_rand = 0
    cnt_cbr = 0
    cnt_dqn = 0

    hist_returns = []
    hist_success = []
    hist_losses = []

    pbar = trange(1, cfg.total_timesteps + 1, dynamic_ncols=True)

    for step in pbar:
        # reseed-by-steps
        new_seed = step_seed(step)
        if new_seed != seed_used:
            seed_used = new_seed
            obs_dict, info = env.reset(seed=seed_used)
            obs = process_raw_obs(obs_dict, is_image)
            ep_return, ep_len = 0.0, 0

        # act
        a, src, chosen_case_or_neg = agent.act(obs, env.action_space)

        if src in ("random", "veto_random"):
            cnt_rand += 1
        elif src == "cbr":
            cnt_cbr += 1
        else:
            cnt_dqn += 1

        # env step
        next_obs_dict, reward, terminated, truncated, info = env.step(a)
        next_obs = process_raw_obs(next_obs_dict, is_image)
        done = bool(terminated or truncated)

        # reward_mean tag (window-based context)
        if len(agent.online_window_r) > 0:
            tail = agent.online_window_r[-agent.window_size:] if len(agent.online_window_r) >= agent.window_size else agent.online_window_r
            reward_mean_tag = float(np.mean(tail))
        else:
            reward_mean_tag = float(reward)

        agent.store(obs, a, reward, next_obs, done)

        ep_return += float(reward)
        ep_len += 1

        # success flag (MiniGrid: success gives reward > 0)
        success_flag = bool(float(reward) > 0.0) and (not bool(truncated))

        # build/store case
        cur_problem = agent.problem_from_obs(obs)


        # check last bad step
        # is_bad_terminal = bool(done) and (not success_flag)
        is_bad_terminal = terminated and (not success_flag)  # only consider "real" terminations as bad, not timeouts

        if is_bad_terminal:
            bad_case = Case(
                problem_chw_f32=cur_problem.copy(),
                solution=a,
                trust_value=0.7,
                reward=float(reward),
                reward_mean=reward_mean_tag,
                is_negative=True,
            )
            agent.add_or_update_negative_case(bad_case)
        else:
            c = Case(
                problem_chw_f32=cur_problem.copy(),
                solution=a,
                trust_value=0.5,
                reward=float(reward),
                reward_mean=reward_mean_tag,
                is_negative=False,
            )
            agent.episode_temp_cases.append(c)
            agent.online_window_cases.append(c)

        # window rewards
        agent.online_window_r.append(float(reward))
        agent.window_tick_step(total_steps_in_episode=ep_len)

        # --- Negative case: collision with static or dynamic obstacle ---
        # info["collision"] is set by TurtlebotMiniGridEnv; always False for MiniGrid.
        if info.get("collision", False):
            collision_case = Case(
                problem_chw_f32=cur_problem.copy(),
                solution=a,
                trust_value=0.6,
                reward=float(reward),
                reward_mean=reward_mean_tag,
                is_negative=True,
            )
            agent.add_or_update_negative_case(collision_case)

        # learn
        loss = agent.learn_step()
        if loss is not None:
            logger.log_loss(step=step, agent=0, loss=loss)
            hist_losses.append(loss)

        # epsilon update
        agent.update_epsilon()

        # advance obs
        obs = next_obs

        if cfg.render_mode == "human":
            env.render()

        if done:
            hist_returns.append(ep_return)
            hist_success.append(1 if success_flag else 0)

            logger.log_episode(
                episode=episode,
                agent=0,
                step=step,
                ep_return=ep_return,
                ep_len=ep_len,
                seed_used=seed_used,
                success_flag=success_flag,
            )

            agent.episode_end_update(success_flag=success_flag)

            episode += 1
            obs_dict, info = env.reset(seed=seed_used)
            obs = process_raw_obs(obs_dict, is_image)
            ep_return, ep_len = 0.0, 0
            cnt_rand = cnt_cbr = cnt_dqn = 0

        if step % cfg.tqdm_refresh == 0:
            pbar.set_description(
                f"ep={episode} step={step} eps={agent.epsilon:.3f} ret={ep_return:.2f} "
                f"cb={len(agent.case_base)} neg={len(agent.case_base_neg)}"
            )

    # save
    agent.save(cfg.ckpt_path, cfg, obs_shape)

    ensure_parent_dir(cfg.case_base_path)
    with open(cfg.case_base_path, "wb") as f:
        pickle.dump({"pos": agent.case_base, "neg": agent.case_base_neg}, f)

    w_ret = min(100, max(1, len(hist_returns)))
    w_suc = min(100, max(1, len(hist_success)))
    w_loss = min(1000, max(1, len(hist_losses)))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    if len(hist_returns) > 0:
        axs[0].plot(np.convolve(hist_returns, np.ones(w_ret)/w_ret, mode='valid'))
    axs[0].set_title("Return")

    if len(hist_success) > 0:
        axs[1].plot(np.convolve(hist_success, np.ones(w_suc)/w_suc, mode='valid'))
    axs[1].set_title("Success")

    if len(hist_losses) > 0:
        axs[2].plot(np.convolve(hist_losses, np.ones(w_loss)/w_loss, mode='valid'))
    axs[2].set_title("Loss")

    fig.tight_layout()
    plot_path = os.path.join(os.path.dirname(cfg.ckpt_path), "training_stats.png")
    fig.savefig(plot_path)
    plt.close(fig)

    env.close()
    print(f"[DONE] Saved log to: {cfg.out_dir}")
    print(f"[DONE] Saved checkpoint to: {cfg.ckpt_path}")
    print(f"[DONE] Saved case bases to: {cfg.case_base_path}")


# =============================
# Main
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D2CBRL-DQN training")
    parser.add_argument("--env-id", "--env_id", type=str, default="MiniGrid-LavaGapS7-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", "--total_timesteps", type=int, default=2_200_000)
    parser.add_argument("--agent-view-size", "--agent_view_size", type=int, default=7)
    parser.add_argument("--render-mode", "--render_mode", type=str, default="rgb_array")
    parser.add_argument("--render", action="store_true", help="Render the environment visually (sets render-mode to human)")
    parser.add_argument("--max-steps", "--max_steps", type=int, default=300)
    parser.add_argument("--reseed-every-n-steps", "--reseed_every_n_steps", type=int, default=0)
    cli = parser.parse_args()
    if cli.render:
        cli.render_mode = "human"

    # For MiniGrid envs, use the restricted action set; for others, use the full action space
    allowed_actions = [0, 1, 2] if is_minigrid_env(cli.env_id) else None

    print(f"[INFO] Running training with env_id={cli.env_id}, seed={cli.seed}")

    # Build output paths from env_id so each environment gets its own folder
    env_slug = cli.env_id.replace(":", "_").replace("/", "_")

    train(
        Config(
            env_id=cli.env_id,
            agent_view_size=cli.agent_view_size,
            allowed_actions=allowed_actions,
            total_timesteps=cli.total_timesteps,
            seed=cli.seed,
            render_mode=cli.render_mode,
            max_steps=cli.max_steps,
            reseed_every_n_steps=cli.reseed_every_n_steps,
            out_dir=f"./logs_csv/train/{env_slug}",
            ckpt_path=f"./checkpoints/{env_slug}/{env_slug}.pth",
            case_base_path=f"./checkpoints/{env_slug}/case_bases.pkl",
        )
    )