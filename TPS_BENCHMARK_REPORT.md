# HST Model TPS Benchmark Report

## Overall Summary

| Model | Class | Params (M) | Avg TPS | Best TPS |
|-------|-------|------------|---------|----------|
| hst_v4_unified | HSTv4Unified | 4.04 | 7902.06 | 8945.53 |
| hst_v3_ultra | HSTv3Ultra | 4.04 | 7618.35 | 8871.94 |
| hst_v5_2_unified | HSTv5_2Unified | 7.12 | 6955.09 | 7690.17 |
| hst_v6_giga | HSTv6Giga | 7.12 | 6830.04 | 7526.98 |
| hst.v6.1 | HSTv6Giga | 16.89 | 5584.01 | 6113.53 |
| hst.v7.1_ultimate | HSTv7_1Ultimate | 57.29 | 2952.98 | 3551.75 |
| HST-chaos_logic_ai | HSTv7Agile | 21.21 | 2816.05 | 3094.85 |
| hst.v7.0_agile | HSTv7Agile | 21.21 | 2688.72 | 2926.12 |

## Detailed Results by Sequence Length

### Sequence Length: 128

| Model | TPS | Time (ms) | Min (ms) | Max (ms) |
|-------|-----|-----------|----------|----------|
| hst_v4_unified | 7119.44 | 17.98 | 17.49 | 18.49 |
| hst_v3_ultra | 6263.34 | 20.44 | 17.60 | 27.45 |
| hst_v5_2_unified | 5728.28 | 22.35 | 21.55 | 23.40 |
| hst_v6_giga | 5706.49 | 22.43 | 21.58 | 23.16 |
| hst.v6.1 | 4584.08 | 27.92 | 26.80 | 30.07 |
| hst.v7.0_agile | 2647.90 | 48.34 | 46.59 | 51.02 |
| HST-chaos_logic_ai | 2527.45 | 50.64 | 47.18 | 56.64 |
| hst.v7.1_ultimate | 2174.03 | 58.88 | 57.45 | 61.37 |

### Sequence Length: 256

| Model | TPS | Time (ms) | Min (ms) | Max (ms) |
|-------|-----|-----------|----------|----------|
| hst_v4_unified | 8945.53 | 28.62 | 27.87 | 29.11 |
| hst_v3_ultra | 8871.94 | 28.86 | 27.76 | 30.03 |
| hst_v5_2_unified | 7690.17 | 33.29 | 32.65 | 33.88 |
| hst_v6_giga | 7526.98 | 34.01 | 32.66 | 35.06 |
| hst.v6.1 | 6113.53 | 41.87 | 40.25 | 43.80 |
| hst.v7.1_ultimate | 3133.14 | 81.71 | 81.27 | 82.34 |
| HST-chaos_logic_ai | 3094.85 | 82.72 | 82.17 | 83.52 |
| hst.v7.0_agile | 2926.12 | 87.49 | 80.83 | 92.18 |

### Sequence Length: 512

| Model | TPS | Time (ms) | Min (ms) | Max (ms) |
|-------|-----|-----------|----------|----------|
| hst_v3_ultra | 7719.76 | 66.32 | 61.35 | 79.83 |
| hst_v4_unified | 7641.22 | 67.01 | 62.92 | 75.24 |
| hst_v5_2_unified | 7446.80 | 68.75 | 67.77 | 71.01 |
| hst_v6_giga | 7256.66 | 70.56 | 67.59 | 79.88 |
| hst.v6.1 | 6054.42 | 84.57 | 83.44 | 85.76 |
| hst.v7.1_ultimate | 3551.75 | 144.15 | 143.01 | 145.23 |
| HST-chaos_logic_ai | 2825.86 | 181.18 | 180.64 | 181.66 |
| hst.v7.0_agile | 2492.15 | 205.45 | 199.18 | 213.80 |

## Performance Insights

- **Best Average TPS**: hst_v4_unified (7902.06 tokens/sec)
- **Most Efficient**: hst_v4_unified (1956.68 TPS per M params)
- **Largest Model**: hst.v7.1_ultimate (57.29M parameters)

## Model Details

### HST-chaos_logic_ai.py

- **Class**: HSTv7Agile
- **Parameters**: 21.21M
- **Layers**: 8
- **Config**: d_model=128, n_layers=8, n_heads=4
- **Performance**:
  - seq_len=128: 2527.45 TPS, 50.64ms
  - seq_len=256: 3094.85 TPS, 82.72ms
  - seq_len=512: 2825.86 TPS, 181.18ms

### hst.v6.1.py

- **Class**: HSTv6Giga
- **Parameters**: 16.89M
- **Layers**: 8
- **Config**: d_model=128, n_layers=8, n_heads=4
- **Performance**:
  - seq_len=128: 4584.08 TPS, 27.92ms
  - seq_len=256: 6113.53 TPS, 41.87ms
  - seq_len=512: 6054.42 TPS, 84.57ms

### hst.v7.0_agile.py

- **Class**: HSTv7Agile
- **Parameters**: 21.21M
- **Layers**: 8
- **Config**: d_model=128, n_layers=8, n_heads=4
- **Performance**:
  - seq_len=128: 2647.90 TPS, 48.34ms
  - seq_len=256: 2926.12 TPS, 87.49ms
  - seq_len=512: 2492.15 TPS, 205.45ms

### hst.v7.1_ultimate.py

- **Class**: HSTv7_1Ultimate
- **Parameters**: 57.29M
- **Config**: d_model=128, n_layers=6, n_heads=4
- **Performance**:
  - seq_len=128: 2174.03 TPS, 58.88ms
  - seq_len=256: 3133.14 TPS, 81.71ms
  - seq_len=512: 3551.75 TPS, 144.15ms

### hst_v3_ultra.py

- **Class**: HSTv3Ultra
- **Parameters**: 4.04M
- **Config**: d_model=128, n_layers=6, n_heads=4
- **Performance**:
  - seq_len=128: 6263.34 TPS, 20.44ms
  - seq_len=256: 8871.94 TPS, 28.86ms
  - seq_len=512: 7719.76 TPS, 66.32ms

### hst_v4_unified.py

- **Class**: HSTv4Unified
- **Parameters**: 4.04M
- **Config**: d_model=128, n_layers=6, n_heads=4
- **Performance**:
  - seq_len=128: 7119.44 TPS, 17.98ms
  - seq_len=256: 8945.53 TPS, 28.62ms
  - seq_len=512: 7641.22 TPS, 67.01ms

### hst_v5_2_unified.py

- **Class**: HSTv5_2Unified
- **Parameters**: 7.12M
- **Config**: d_model=128, n_layers=6, n_heads=4
- **Performance**:
  - seq_len=128: 5728.28 TPS, 22.35ms
  - seq_len=256: 7690.17 TPS, 33.29ms
  - seq_len=512: 7446.80 TPS, 68.75ms

### hst_v6_giga.py

- **Class**: HSTv6Giga
- **Parameters**: 7.12M
- **Config**: d_model=128, n_layers=6, n_heads=4
- **Performance**:
  - seq_len=128: 5706.49 TPS, 22.43ms
  - seq_len=256: 7526.98 TPS, 34.01ms
  - seq_len=512: 7256.66 TPS, 70.56ms

