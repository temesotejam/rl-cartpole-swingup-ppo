from __future__ import annotations

import math

import numpy as np
from stable_baselines3 import PPO

from src.environment import PhysicsConfig, SensorNoiseConfig, make_swingup_env
from src.evaluation import evaluate_policy


def test_downward_reset_does_not_terminate_for_angle() -> None:
    env = make_swingup_env(
        physics=PhysicsConfig(),
        sensor_noise=SensorNoiseConfig(enabled=False),
        reset_mode="evaluation_downward",
    )
    obs, info = env.reset(seed=123)
    theta = info["true_state"][2]
    assert abs(abs(theta) - math.pi) < math.radians(6.0)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert obs.shape == (5,)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    env.close()


def test_sensor_noise_is_seed_reproducible() -> None:
    env1 = make_swingup_env(reset_mode="evaluation_downward")
    env2 = make_swingup_env(reset_mode="evaluation_downward")
    o1, _ = env1.reset(seed=55)
    o2, _ = env2.reset(seed=55)
    np.testing.assert_allclose(o1, o2)
    env1.close(); env2.close()


def test_short_ppo_training_path() -> None:
    env = make_swingup_env(reset_mode="near_upright")
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, seed=3, device="cpu", verbose=0)
    model.learn(total_timesteps=128)
    metrics = evaluate_policy(model, [10], PhysicsConfig(), SensorNoiseConfig())
    assert np.isfinite(metrics.mean_return)
    assert 0.0 <= metrics.capture_rate <= 1.0
    env.close()
