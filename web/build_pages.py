from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path


def pct(value) -> str:
    return f"{100*float(value):.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-url", default="")
    args = p.parse_args()
    source = args.input
    out = args.output
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    for folder in ["videos", "plots"]:
        shutil.copytree(source / folder, out / folder)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    with (source / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    trs = []
    for r in rows:
        ttc = "-" if not r.get("mean_time_to_capture_s") else f"{float(r['mean_time_to_capture_s']):.2f}s"
        trs.append(
            f"<tr><td>{html.escape(r['stage'])}</td><td>{int(float(r['timesteps'])):,}</td>"
            f"<td>{float(r['mean_return']):.1f}</td><td>{pct(r['capture_rate'])}</td><td>{ttc}</td>"
            f"<td>{pct(r['final_stable_rate'])}</td><td>{pct(r['upright_ratio'])}</td>"
            f"<td>{float(r['rms_cart_position_m']):.3f} m</td></tr>"
        )
    videos = [
        ("学習前", "00_random.mp4"), ("25%", "01_25_percent.mp4"),
        ("50%", "02_50_percent.mp4"), ("75%", "03_75_percent.mp4"),
        ("100%", "04_100_percent.mp4"),
    ]
    video_html = "".join(
        f'<section class="card"><h3>{label}</h3><video controls preload="metadata" src="videos/{name}"></video></section>'
        for label, name in videos
    )
    plot_html = "".join(
        f'<section class="card"><img src="plots/{name}" alt="{name}"></section>'
        for name in ["learning_curve.png", "capture_rate.png", "final_stable_rate.png", "upright_ratio.png", "cart_position.png"]
    )
    run_link = f'<a href="{html.escape(args.run_url)}">GitHub Actions run</a>' if args.run_url else ""
    sensor = metadata["sensor_noise"]
    physics = metadata["physics"]
    page = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cart-Pole Swing-up PPO</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f5f7fa;color:#172033}}main{{max-width:1200px;margin:auto;padding:28px}}
.hero,.card{{background:white;border-radius:16px;padding:22px;margin-bottom:18px;box-shadow:0 4px 16px #0001}}h1{{margin-top:0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
video,img{{width:100%;border-radius:10px}}table{{width:100%;border-collapse:collapse;overflow:auto}}th,td{{padding:9px;border-bottom:1px solid #dde3eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eef2f7;padding:2px 5px;border-radius:5px}}
</style></head><body><main>
<section class="hero"><h1>Cart-Pole Swing-up PPO</h1><p>真下から振り上げ、倒立を捕捉し、台車を中央へ戻す強化学習実験。</p>
<p><b>{metadata['preset']}</b> / seed {metadata['seed']} / requested {metadata['requested_total_timesteps']:,} steps / {run_link}</p></section>
<section class="card"><h2>評価結果（全て真下スタート）</h2><table><thead><tr><th>段階</th><th>steps</th><th>return</th><th>capture</th><th>capture time</th><th>final stable</th><th>upright</th><th>RMS cart</th></tr></thead><tbody>{''.join(trs)}</tbody></table></section>
<h2>学習進行動画</h2><div class="grid">{video_html}</div>
<h2>学習曲線</h2><div class="grid">{plot_html}</div>
<section class="card"><h2>実験条件</h2><p>Cart {physics['cart_mass_kg']} kg / Pole {physics['pole_mass_kg']} kg / Force ±{physics['max_force_n']} N / Track ±{physics['track_limit_m']} m / {1/physics['dt_s']:.0f} Hz / episode {physics['max_episode_s']} s</p>
<p>角度ノイズ σ={sensor['angle_noise_std_deg']}°、角度バイアス σ={sensor['angle_bias_std_deg']}°、ジャイロノイズ σ={sensor['gyro_noise_std_dps']}°/s。</p>
<p>Curriculum: <code>near_upright → wide → full → downward_mix</code></p></section>
</main></body></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__": main()
