# Raw Telemetry Log Quality Audit

Current as of August 11, 2026. This is an audit, not a deletion list. Do not delete raw logs unless a backup exists and the exclusion rationale is recorded.

## Summary By Category

| Category | Count |
|---|---:|
| accepted/formal-or-candidate | 26 |
| debug/bench | 5 |
| debug/calibration | 13 |
| debug/gps-or-bench | 0 |
| debug/interrupted | 1 |
| development/new-route-test | 0 |
| exploratory/legacy | 0 |
| keep/unknown | 6 |
| legacy/square-test | 3 |

## Logs That Are Low-Value For Formal Results

| File | Category | Rows | Duration s | GPS valid | Why low-value / how to treat |
|---|---|---:|---:|---:|---|
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-5_20260720_114957.csv` | keep/unknown | 217 | 231.2 | 154 | Low successful-cycle fraction (154/217). Packet/request issues: request_fail=64, seq_gap=0. |
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-5_20260721_112657.csv` | debug/interrupted | 2 | 0.6 | 0 | Very short capture, likely connection/start-stop test. No GPS-valid rows. |
| `ugv_t147_bench_20260715_191016.csv` | keep/unknown | 32 | 30.9 | 32 | No obvious issue; inspect manually before excluding. |
| `ugv_t147_bench_20260715_191250.csv` | keep/unknown | 235 | 234.6 | 235 | No obvious issue; inspect manually before excluding. |
| `ugv_t147_bench_20260716_150127.csv` | keep/unknown | 109 | 73.1 | 0 | No GPS-valid rows. |
| `ugv_t147_bench_20260805_114401.csv` | debug/bench | 426 | 290.3 | 0 | August 5 bench/AprilTag prep or diagnostic run; not accepted benign matrix naming. No GPS-valid rows. Mostly stationary or no meaningful command motion. |
| `ugv_t147_bench_20260805_115355.csv` | debug/bench | 24 | 14.2 | 0 | August 5 bench/AprilTag prep or diagnostic run; not accepted benign matrix naming. No GPS-valid rows. |
| `ugv_t147_bench_20260805_115442.csv` | debug/bench | 186 | 113.0 | 0 | August 5 bench/AprilTag prep or diagnostic run; not accepted benign matrix naming. No GPS-valid rows. Mostly stationary or no meaningful command motion. |
| `ugv_t147_bench_20260805_120833.csv` | debug/bench | 5 | 2.3 | 0 | Very short capture, likely connection/start-stop test. August 5 bench/AprilTag prep or diagnostic run; not accepted benign matrix naming. No GPS-valid rows. |
| `ugv_t147_bench_20260805_172558.csv` | debug/bench | 20 | 11.6 | 0 | August 5 bench/AprilTag prep or diagnostic run; not accepted benign matrix naming. No GPS-valid rows. |
| `ugv_t147_interactive_20260805_174551.csv` | keep/unknown | 173 | 223.9 | 0 | No GPS-valid rows. Packet/request issues: request_fail=1, seq_gap=0. |
| `ugv_t147_interactive_20260805_192736.csv` | keep/unknown | 185 | 220.6 | 0 | No GPS-valid rows. |
| `ugv_t147_interactive_20260810_184345.csv` | debug/calibration | 87 | 95.8 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_184844.csv` | debug/calibration | 44 | 53.9 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_185111.csv` | debug/calibration | 31 | 34.3 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_185359.csv` | debug/calibration | 27 | 29.5 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_185606.csv` | debug/calibration | 20 | 18.8 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_185729.csv` | debug/calibration | 27 | 32.7 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_185813.csv` | debug/calibration | 45 | 50.0 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_190047.csv` | debug/calibration | 35 | 37.9 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_190229.csv` | debug/calibration | 36 | 35.7 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_190322.csv` | debug/calibration | 60 | 63.5 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_191810.csv` | debug/calibration | 1 | 0.0 | 0 | Very short capture, likely connection/start-stop test. Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_191818.csv` | debug/calibration | 80 | 84.2 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_interactive_20260810_193945.csv` | debug/calibration | 43 | 44.0 | 0 | Interactive turn-calibration/debug run from August 10, not a formal dataset trial. No GPS-valid rows. |
| `ugv_t147_square_0_5m_x1_20260715_185022.csv` | legacy/square-test | 9 | 8.5 | 9 | Very short capture, likely connection/start-stop test. Legacy square script test before final naming/protocol. |
| `ugv_t147_square_0_5m_x1_20260716_145829.csv` | legacy/square-test | 81 | 53.6 | 0 | Legacy square script test before final naming/protocol. No GPS-valid rows. Packet/request issues: request_fail=1, seq_gap=1. |
| `ugv_t147_square_0_5m_x1_20260716_150513.csv` | legacy/square-test | 29 | 18.4 | 0 | Legacy square script test before final naming/protocol. No GPS-valid rows. |

## Accepted Or Candidate Formal Logs

| File | Rows | Duration s | GPS valid | Note |
|---|---:|---:|---:|---|
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_20260720_105445.csv` | 157 | 152.4 | 157 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_20260720_111623.csv` | 182 | 164.5 | 182 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-2_20260720_110006.csv` | 99 | 97.8 | 99 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-2_20260720_111921.csv` | 184 | 265.4 | 184 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-3_20260720_110338.csv` | 158 | 164.6 | 155 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-3_20260720_112359.csv` | 74 | 77.1 | 74 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-3_20260720_112722.csv` | 169 | 168.8 | 169 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-4_20260720_110854.csv` | 175 | 166.1 | 175 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-4_20260720_113158.csv` | 165 | 165.7 | 165 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-5_20260720_113452.csv` | 165 | 165.8 | 165 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_20260721_113240.csv` | 185 | 155.9 | 185 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-2_20260715_185044.csv` | 153 | 152.6 | 153 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-3_20260715_185341.csv` | 158 | 157.1 | 158 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-4_20260715_190007.csv` | 164 | 163.1 | 164 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-low_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-5_20260715_190320.csv` | 155 | 153.9 | 155 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_20260720_113804.csv` | 152 | 151.7 | 152 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-2_20260720_114143.csv` | 154 | 152.9 | 154 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-3_20260720_114429.csv` | 150 | 149.0 | 150 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-4_20260720_114711.csv` | 153 | 153.7 | 153 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-rough_permeable_concrete_latency-wifi_baseline_route-square0p5x3_attack-none_trial-5_20260720_115507.csv` | 156 | 155.6 | 156 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_20260721_113640.csv` | 160 | 143.1 | 160 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-1_20260721_114324.csv` | 156 | 141.6 | 156 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-2_20260721_114955.csv` | 150 | 140.5 | 150 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-3_20260721_115218.csv` | 153 | 143.6 | 153 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-4_20260721_115444.csv` | 152 | 141.8 | 152 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
| `speed-medium_surface-smooth_kitchen_floor_latency-wifi_baseline_route-square0p5x3_attack-none_trial-5_20260721_115709.csv` | 153 | 143.0 | 153 | Formal benign matrix-style name; keep unless duplicate/error trial excluded by manifest. |
