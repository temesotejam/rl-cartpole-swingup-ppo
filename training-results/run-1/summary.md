# Cart-Pole Swing-up PPO training result

- Preset: `normal`
- Seed: `42`
- Best checkpoint: `100_percent`
- Best final-stable rate: `91.7%`

## Downward-start evaluation

| Stage | Timesteps | Return | Capture | Time to capture | Final stable | Upright +/-10 deg | RMS cart x |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0 | 9.4 | 0.0% | - | 0.0% | 0.0% | 1.062m |
| 25_percent | 102,400 | 456.5 | 0.0% | - | 0.0% | 5.1% | 0.200m |
| 50_percent | 204,800 | 644.5 | 8.3% | 2.52s | 0.0% | 14.2% | 0.297m |
| 75_percent | 307,200 | 1206.8 | 100.0% | 2.90s | 8.3% | 51.3% | 0.418m |
| 100_percent | 409,600 | 1612.4 | 100.0% | 2.77s | 91.7% | 79.2% | 0.299m |

Capture means the pole remained within +/-12 deg with low angular velocity for at least 0.5 s.
Final stable means at least 80% of the final 2 s was within +/-10 deg and +/-0.50 m.
