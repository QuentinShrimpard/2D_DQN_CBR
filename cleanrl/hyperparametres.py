"""
Print all hyperparameters used in dqn.py and dqn_cbr.py side by side.
"""


def print_dqn_hyperparameters():
    """Hyperparameters from dqn.py (vanilla DQN)."""
    params = {
        # ---- ENV ----
        "env_id": "MiniGrid-LavaGapS7-v0",
        "seed": 0,
        "agent_view_size": 7,
        "total_timesteps": 600_000,
        "render_mode": "rgb_array",
        "max_steps": 300,

        # ---- DQN ----
        "gamma": 0.99,
        "lr": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "learning_starts": 2_000,
        "train_freq": 4,
        "target_update_interval": 1_000,

        # ---- Epsilon-greedy ----
        "eps_start": 1.0,
        "eps_final": 0.1,
        "eps_fraction": 0.5,

        # ---- Shaping ----
        "step_cost": 0.01,
        "turn_cost": 0.001,
        "turn_actions": (0, 1),

        # ---- Actions ----
        "allowed_actions": [0, 1, 2],

        # ---- Vanilla-style reseed ----
        "reseed_every_n_steps": 0,

        # ---- Output ----
        "out_dir": "./logs_csv/train/dqn_lavagap",
        "ckpt_path": "./checkpoints/dqn_lavagap/dqn_lavagap.pth",
        "tqdm_refresh": 200,
    }

    print("=" * 60)
    print("  DQN (dqn.py) — Hyperparameters")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key:<30s} = {value}")
    print()


def print_dqn_cbr_hyperparameters():
    """Hyperparameters from dqn_cbr.py (D2CBRL-DQN)."""
    params = {
        # ---- ENV ----
        "env_id": "MiniGrid-LavaGapS7-v0",
        "seed": 0,
        "agent_view_size": 7,
        "total_timesteps": 600_000,
        "render_mode": "rgb_array",
        "max_steps": 300,

        # ---- DQN ----
        "gamma": 0.99,
        "lr": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "learning_starts": 2_000,
        "train_freq": 4,
        "target_update_interval": 1_000,

        # ---- Epsilon-greedy ----
        "eps_start": 1.0,
        "eps_final": 0.1,
        "eps_fraction": 0.5,

        # ---- Shaping ----
        "step_cost": 0.01,
        "turn_cost": 0.001,
        "turn_actions": (0, 1),

        # ---- Actions ----
        "allowed_actions": [0, 1, 2],

        # ---- Vanilla-style reseed ----
        "reseed_every_n_steps": 0,

        # ---- Output ----
        "out_dir": "./logs_csv/train/d2cbrl_dqn_lavagap",
        "ckpt_path": "./checkpoints/d2cbrl_dqn_lavagap/d2cbrl_dqn_lavagap.pth",
        "case_base_path": "./checkpoints/d2cbrl_dqn_lavagap/case_bases.pkl",
        "tqdm_refresh": 200,

        # ---- D2CBRL params ----
        "p_cbr": 0.4,
        "min_trust": 0.3,
        "max_case_dist": 1.0,

        # ---- Window-based success detection ----
        "window_size": 10,
        "window_stride": 1,
        "min_windows_for_stats": 5,
        "win_delta": 0.02,
        "win_abs_floor": 0.0,

        # ---- Negative case base veto ----
        "neg_min_badness": 0.5,

        # ---- Case trust update constants (positive) ----
        "Case.INCREMENT_SUCCESS": 0.1,
        "Case.DECREMENT_UNSUCCESS": 0.1,
        "Case.DECREMENT_SUCCESS_NO_CASE": 0.1,
        "Case.DEL_THRESHOLD": 0.1,
        "Case.MATCH_THRESHOLD": 1.0,

        # ---- Case trust update constants (negative) ----
        "Case.NEG_INC_ON_FAIL": 0.2,
        "Case.NEG_DEC_ON_SUCCESS": 0.1,
        "Case.NEG_DEC_ON_SUCCESS_NO_CASE": 0.05,
        "Case.NEG_DEL_THRESHOLD": 0.1,
    }

    print("=" * 60)
    print("  D2CBRL-DQN (dqn_cbr.py) — Hyperparameters")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key:<40s} = {value}")
    print()


def print_comparison():
    """Print a side-by-side comparison of shared vs unique params."""

    shared = {
        "env_id": "MiniGrid-LavaGapS7-v0",
        "seed": 0,
        "agent_view_size": 7,
        "total_timesteps": 600_000,
        "render_mode": "rgb_array",
        "max_steps": 300,
        "gamma": 0.99,
        "lr": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "learning_starts": 2_000,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "eps_start": 1.0,
        "eps_final": 0.1,
        "eps_fraction": 0.5,
        "step_cost": 0.01,
        "turn_cost": 0.001,
        "turn_actions": (0, 1),
        "allowed_actions": [0, 1, 2],
        "reseed_every_n_steps": 0,
        "tqdm_refresh": 200,
    }

    cbr_only = {
        "p_cbr": 0.4,
        "min_trust": 0.3,
        "max_case_dist": 1.0,
        "window_size": 10,
        "window_stride": 1,
        "min_windows_for_stats": 5,
        "win_delta": 0.02,
        "win_abs_floor": 0.0,
        "neg_min_badness": 0.5,
        "case_base_path": "./checkpoints/d2cbrl_dqn_lavagap/case_bases.pkl",
        "Case.INCREMENT_SUCCESS": 0.1,
        "Case.DECREMENT_UNSUCCESS": 0.1,
        "Case.DECREMENT_SUCCESS_NO_CASE": 0.1,
        "Case.DEL_THRESHOLD": 0.1,
        "Case.MATCH_THRESHOLD": 1.0,
        "Case.NEG_INC_ON_FAIL": 0.2,
        "Case.NEG_DEC_ON_SUCCESS": 0.1,
        "Case.NEG_DEC_ON_SUCCESS_NO_CASE": 0.05,
        "Case.NEG_DEL_THRESHOLD": 0.1,
    }

    print("=" * 60)
    print("  SHARED hyperparameters (DQN & D2CBRL-DQN)")
    print("=" * 60)
    for key, value in shared.items():
        print(f"  {key:<30s} = {value}")

    print()
    print("=" * 60)
    print("  D2CBRL-ONLY hyperparameters (unique to dqn_cbr.py)")
    print("=" * 60)
    for key, value in cbr_only.items():
        print(f"  {key:<40s} = {value}")

    print()


if __name__ == "__main__":
    print_dqn_hyperparameters()
    print_dqn_cbr_hyperparameters()
    print("-" * 60)
    print_comparison()