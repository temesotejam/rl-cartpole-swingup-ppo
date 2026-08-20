from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def write_metrics_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict], key: str, ylabel: str, path: Path, scale: float = 1.0) -> None:
    x = [r["timesteps"] for r in records]
    y = [float(r[key] or 0.0) * scale for r in records]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_plots(records: list[dict], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot(records, "mean_return", "Mean return", plots_dir / "learning_curve.png")
    _plot(records, "capture_rate", "Swing-up capture rate [%]", plots_dir / "capture_rate.png", 100.0)
    _plot(records, "final_stable_rate", "Final stable rate [%]", plots_dir / "final_stable_rate.png", 100.0)
    _plot(records, "upright_ratio", "Time within +/-10 deg [%]", plots_dir / "upright_ratio.png", 100.0)
    _plot(records, "rms_cart_position_m", "RMS cart position [m]", plots_dir / "cart_position.png")


def write_summary(records: list[dict], path: Path, preset: str, seed: int) -> None:
    best = max(records[1:], key=lambda r: (r["final_stable_rate"], r["capture_rate"], r["mean_return"]))
    lines = [
        "# Cart-Pole Swing-up PPO training result",
        "",
        f"- Preset: `{preset}`",
        f"- Seed: `{seed}`",
        f"- Best checkpoint: `{best['stage']}`",
        f"- Best final-stable rate: `{100*best['final_stable_rate']:.1f}%`",
        "",
        "## Downward-start evaluation",
        "",
        "| Stage | Timesteps | Return | Capture | Time to capture | Final stable | Upright +/-10 deg | RMS cart x |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        ttc = "-" if r["mean_time_to_capture_s"] is None else f"{r['mean_time_to_capture_s']:.2f}s"
        lines.append(
            f"| {r['stage']} | {r['timesteps']:,} | {r['mean_return']:.1f} | "
            f"{100*r['capture_rate']:.1f}% | {ttc} | {100*r['final_stable_rate']:.1f}% | "
            f"{100*r['upright_ratio']:.1f}% | {r['rms_cart_position_m']:.3f}m |"
        )
    lines += [
        "",
        "Capture means the pole remained within +/-12 deg with low angular velocity for at least 0.5 s.",
        "Final stable means at least 80% of the final 2 s was within +/-10 deg and +/-0.50 m.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
