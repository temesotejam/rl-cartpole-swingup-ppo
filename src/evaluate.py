from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from .environment import PhysicsConfig, SensorNoiseConfig
from .evaluation import evaluate_episode, evaluate_policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/normal.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--video", type=Path)
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    physics = PhysicsConfig(**config["physics"])
    noise = SensorNoiseConfig(**config["sensor_noise"])
    model = PPO.load(args.model, device="cpu")
    metrics = evaluate_policy(model, [args.seed + i for i in range(args.episodes)], physics, noise)
    if args.video:
        evaluate_episode(model, args.seed, physics, noise, video_path=args.video)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
