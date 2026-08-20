from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import imageio.v2 as imageio
import numpy as np

from .environment import PhysicsConfig, SensorNoiseConfig, make_swingup_env


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass
class EpisodeMetrics:
    episode_return: float
    survival_s: float
    completed_episode: float
    captured_upright: float
    time_to_capture_s: float | None
    final_stable: float
    upright_ratio: float
    rms_angle_deg: float
    rms_cart_position_m: float
    rms_force_n: float


@dataclass
class AggregateMetrics:
    mean_return: float
    std_return: float
    mean_survival_s: float
    completion_rate: float
    capture_rate: float
    mean_time_to_capture_s: float | None
    final_stable_rate: float
    upright_ratio: float
    rms_angle_deg: float
    rms_cart_position_m: float
    rms_force_n: float

    def to_dict(self) -> dict:
        return asdict(self)


def _action(policy: PredictPolicy | None, env, observation: np.ndarray) -> np.ndarray:
    if policy is None:
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    action, _ = policy.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def evaluate_episode(
    policy: PredictPolicy | None,
    seed: int,
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
    video_path: Path | None = None,
) -> EpisodeMetrics:
    render_mode = "rgb_array" if video_path is not None else None
    env = make_swingup_env(
        physics=physics,
        sensor_noise=sensor_noise,
        reset_mode="evaluation_downward",
        render_mode=render_mode,
    )
    env.action_space.seed(seed + 10_000)
    observation, info = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    angles: list[float] = []
    positions: list[float] = []
    forces: list[float] = []
    stable_flags: list[bool] = []
    episode_return = 0.0
    capture_run = 0
    capture_required = max(1, int(round(0.5 / physics.dt_s)))
    capture_time: float | None = None

    terminated = truncated = False
    while not (terminated or truncated):
        action = _action(policy, env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        x, _, theta, theta_dot = [float(v) for v in info["true_state"]]
        angles.append(theta)
        positions.append(x)
        forces.append(float(info["actual_force_n"]))

        capture_condition = abs(theta) <= math.radians(12.0) and abs(theta_dot) <= 1.5
        capture_run = capture_run + 1 if capture_condition else 0
        if capture_time is None and capture_run >= capture_required:
            capture_time = float(info["time_s"] - (capture_required - 1) * physics.dt_s)

        stable_flags.append(
            abs(theta) <= math.radians(10.0)
            and abs(x) <= 0.50
            and abs(theta_dot) <= 1.2
        )

        if video_path is not None and len(angles) % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    env.close()
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=25, macro_block_size=1)

    angle_a = np.asarray(angles, dtype=np.float64)
    pos_a = np.asarray(positions, dtype=np.float64)
    force_a = np.asarray(forces, dtype=np.float64)
    last_steps = max(1, int(round(2.0 / physics.dt_s)))
    final_stable = float(np.mean(stable_flags[-last_steps:]) >= 0.80)
    return EpisodeMetrics(
        episode_return=episode_return,
        survival_s=len(angles) * physics.dt_s,
        completed_episode=float(truncated and not terminated),
        captured_upright=float(capture_time is not None),
        time_to_capture_s=capture_time,
        final_stable=final_stable,
        upright_ratio=float(np.mean(np.abs(angle_a) <= math.radians(10.0))),
        rms_angle_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(angle_a))))),
        rms_cart_position_m=float(np.sqrt(np.mean(np.square(pos_a)))),
        rms_force_n=float(np.sqrt(np.mean(np.square(force_a)))),
    )


def evaluate_policy(
    policy: PredictPolicy | None,
    seeds: list[int],
    physics: PhysicsConfig,
    sensor_noise: SensorNoiseConfig,
) -> AggregateMetrics:
    episodes = [evaluate_episode(policy, s, physics, sensor_noise) for s in seeds]
    captures = [e.time_to_capture_s for e in episodes if e.time_to_capture_s is not None]
    returns = np.asarray([e.episode_return for e in episodes], dtype=np.float64)
    return AggregateMetrics(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_survival_s=float(np.mean([e.survival_s for e in episodes])),
        completion_rate=float(np.mean([e.completed_episode for e in episodes])),
        capture_rate=float(np.mean([e.captured_upright for e in episodes])),
        mean_time_to_capture_s=float(np.mean(captures)) if captures else None,
        final_stable_rate=float(np.mean([e.final_stable for e in episodes])),
        upright_ratio=float(np.mean([e.upright_ratio for e in episodes])),
        rms_angle_deg=float(np.mean([e.rms_angle_deg for e in episodes])),
        rms_cart_position_m=float(np.mean([e.rms_cart_position_m for e in episodes])),
        rms_force_n=float(np.mean([e.rms_force_n for e in episodes])),
    )
