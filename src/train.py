from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .environment import PhysicsConfig, SensorNoiseConfig, make_swingup_env
from .evaluation import evaluate_episode, evaluate_policy
from .reporting import create_plots, write_metrics_csv, write_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = [
    ("25_percent", "near_upright"),
    ("50_percent", "wide"),
    ("75_percent", "full"),
    ("100_percent", "downward_mix"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def load_config(preset: str) -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / f"{preset}.yaml").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def config_objects(config: dict) -> tuple[PhysicsConfig, SensorNoiseConfig]:
    return PhysicsConfig(**config["physics"]), SensorNoiseConfig(**config["sensor_noise"])


def build_vec_env(config: dict, mode: str, seed: int):
    physics, noise = config_objects(config)
    n_envs = int(config["ppo"]["n_envs"])
    return make_vec_env(
        lambda: make_swingup_env(physics=physics, sensor_noise=noise, reset_mode=mode),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
    )


def evaluate_stage(records, stage, mode, model, timesteps, eval_seeds, video_seed, output_dir, physics, noise):
    metrics = evaluate_policy(model, eval_seeds, physics, noise)
    evaluate_episode(
        model,
        video_seed,
        physics,
        noise,
        video_path=output_dir / "videos" / f"{len(records):02d}_{stage}.mp4",
    )
    record = {"stage": stage, "curriculum_mode": mode, "timesteps": int(timesteps), **metrics.to_dict()}
    records.append(record)
    print(
        f"[{stage}/{mode}] steps={timesteps:,} return={metrics.mean_return:.1f} "
        f"capture={100*metrics.capture_rate:.1f}% final={100*metrics.final_stable_rate:.1f}%"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.preset)
    set_seed(args.seed)
    physics, noise = config_objects(config)
    out = args.output_dir.resolve()
    for name in ["models", "videos", "plots"]:
        (out / name).mkdir(parents=True, exist_ok=True)

    eval_seeds = [args.seed + 100 + i for i in range(int(config["evaluation_episodes"]))]
    video_seed = args.seed + 999
    records: list[dict] = []
    evaluate_stage(records, "random", "evaluation_downward", None, 0, eval_seeds, video_seed, out, physics, noise)

    total = int(config["total_timesteps"])
    quarter = max(1, total // 4)
    model = None
    actual_steps = 0
    for index, (stage, mode) in enumerate(CURRICULUM):
        env = build_vec_env(config, mode, args.seed + 10 * index)
        if model is None:
            ppo = config["ppo"]
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=float(ppo["learning_rate"]),
                n_steps=int(ppo["n_steps"]),
                batch_size=int(ppo["batch_size"]),
                n_epochs=int(ppo["n_epochs"]),
                gamma=float(ppo["gamma"]),
                gae_lambda=float(ppo["gae_lambda"]),
                clip_range=float(ppo["clip_range"]),
                ent_coef=float(ppo["ent_coef"]),
                use_sde=bool(ppo["use_sde"]),
                sde_sample_freq=int(ppo["sde_sample_freq"]),
                policy_kwargs={"net_arch": [128, 128], "activation_fn": torch.nn.Tanh},
                seed=args.seed,
                device="cpu",
                verbose=1,
            )
        else:
            model.set_env(env)
        before = model.num_timesteps
        model.learn(total_timesteps=quarter, reset_num_timesteps=False, progress_bar=False)
        actual_steps += model.num_timesteps - before
        model.save(out / "models" / f"{stage}.zip")
        evaluate_stage(records, stage, mode, model, model.num_timesteps, eval_seeds, video_seed, out, physics, noise)
        env.close()

    write_metrics_csv(records, out / "metrics.csv")
    create_plots(records, out / "plots")
    write_summary(records, out / "summary.md", args.preset, args.seed)
    metadata = {
        "environment": "Continuous Cart-Pole Swing-up",
        "algorithm": "PPO",
        "preset": args.preset,
        "seed": args.seed,
        "requested_total_timesteps": total,
        "actual_training_timesteps": actual_steps,
        "evaluation_start": "near downward (+/-5 deg)",
        "curriculum": [{"checkpoint": a, "reset_mode": b} for a, b in CURRICULUM],
        "physics": physics.to_dict(),
        "sensor_noise": noise.to_dict(),
        "config": config,
        "versions": {name: importlib.metadata.version(name) for name in ["gymnasium", "stable-baselines3", "torch", "numpy"]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
