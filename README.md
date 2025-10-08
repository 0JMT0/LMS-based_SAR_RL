# LMS-based SAR ADC Calibration with Q-learning

LMS-based background calibration for SAR ADCs with Q-learning–driven step-size selection. Implements the provided pseudocode and technical notes, including ENOB-discretized state space, epsilon-greedy action selection over 9 LMS step sizes, and a reward shaped by ENOB improvement.

## Highlights
- RL + LMS core
  - 411 ENOB states: `8.00..12.10` with `0.01` step.
  - 9 actions (LMS steps): `A = [2^-7 .. 2^-15]`.
  - LMS update (Eq.2): `w_{i,n+1} = w_{i,n} + step * E_PN[b_out,i]`.
  - Q-learning (Eq.3) Bellman update, epsilon-greedy exploration.
- Training modes
  - Standard: `--episodes`, `--max-steps` with fixed epsilon decay.
  - Paper-like grouped demo: randomly generate 50 groups of 5% mismatch; each group trains 10 episodes with fixed true weight (total 500 episodes) to emulate repeated calibration across dies.
- I/O and persistence
  - CSV per-episode logs export.
  - `.npy` save/load of Q-tables per comparator noise level (`sigma_cmp_lsb`).

## File Layout
- `LMS-based_SAR_RL/lms_sar_rl.py` — main implementation, CLI, and demos.
- `LMS-based_SAR_RL/README.md` — this guide.
- Optional outputs (created by you):
  - `LMS-based_SAR_RL/paper_logs.csv`, `LMS-based_SAR_RL/demo_logs.csv` — CSV logs.
  - `LMS-based_SAR_RL/Q_sigma=<noise>.npy` — saved Q-table per noise level.

## Quick Start
- Standard quick run:
  - `python LMS-based_SAR_RL/lms_sar_rl.py`
- Paper-like grouped demo (50×10=500 episodes):
  - `python LMS-based_SAR_RL/lms_sar_rl.py --paper-demo --groups 50 --per-group-episodes 10 --mismatch-sigma 0.05 --max-steps 250 --alpha 0.1 --gamma 0.8 --epsilon 0.5 --noise-levels 0.25 --csv LMS-based_SAR_RL/paper_logs.csv --save-qdir LMS-based_SAR_RL`

中文（簡）：啟用論文式示範，隨機生成 50 組 5% 失配，每組 10 次訓練（共 500 episodes），並輸出 CSV 與 Q-table。

## CLI Options
- `--episodes` int: number of training episodes (standard mode; ignored in paper-demo).
- `--max-steps` int: max steps per episode.
- `--alpha` float: Q-learning learning rate.
- `--gamma` float: discount factor.
- `--epsilon` float: initial epsilon for epsilon-greedy; decays each episode.
- `--seed` int: random seed base; different noise levels offset by index.
- `--noise-levels` str: comma-separated comparator noise sigmas in LSB, e.g., `0.1,0.25,0.5`.
- `--csv` path: export per-episode metrics to CSV. If multiple noise levels, `_sigma=<val>` is appended.
- `--save-qdir` dir: save Q-table as `Q_sigma=<noise>.npy`.
- `--load-qdir` dir: load Q-table if available; skips training for that noise level.
- `--paper-demo`: enable grouped demo; uses `--groups * --per-group-episodes` episodes.
- `--groups` int: number of groups (default 50).
- `--per-group-episodes` int: episodes per group (default 10).
- `--mismatch-sigma` float: per-group true-weight Gaussian sigma (e.g., `0.05` for 5%).
 - `--dump-q-csv` path: dump the full Q-table to CSV (ENOB as first column, actions as columns). With multiple noise levels, appends `_sigma=<val>` or writes separate files if a directory is given.
 - `--print-q` spec: pretty-print selected Q rows to console. `spec` can be a comma list like `8.5,10.5,11.8`, a range like `8.0:12.1:0.5`, or `all` (prints all 411 rows).

## CSV Export
When `--csv` is given, a CSV with the following columns is written:
- `episode`: 1-based episode index.
- `episode_mean_enob`: mean ENOB over steps within the episode.
- `episode_last_enob`: ENOB at the final step of the episode.
- `episode_steps`: number of steps actually taken (may be < max if early stop triggers).
- `episode_total_reward`: sum of rewards over the episode.
- `epsilon_hist`: epsilon value used at the start of the episode.

If multiple noise levels are trained in the same run and `--csv` ends with `.csv`, the tool saves separate files with suffix `_sigma=<value>.csv`. If `--csv` points to a directory, files are written as `logs_sigma=<value>.csv` inside that directory.

## Q-table Save/Load
- Save: pass `--save-qdir <dir>` to save each trained Q-table as `Q_sigma=<noise>.npy`.
- Load: pass `--load-qdir <dir>` to load existing tables. If a table is found for a noise level, training is skipped and the loaded table is used for the demo/summary in that run.

### Loading Q‑tables correctly (know‑how)
- `--load-qdir` expects a directory, not a file path. The loader looks for files named `Q_sigma=<noise>.npy` inside that directory, where `<noise>` matches each value in `--noise-levels`.
- Correct usage examples:
  - From inside the folder that contains the file: `python lms_sar_rl.py --load-qdir . --noise-levels 0.25`
  - From the repo root: `python lms_sar_rl.py --load-qdir . --noise-levels 0.25`
- Incorrect usage to avoid: `--load-qdir Q_sigma=0.25.npy` (that is a file path; the code expects a directory).
- Multi‑noise behavior: for each value in `--noise-levels`, the loader tries to find `Q_sigma=<that_value>.npy`. Missing ones will be trained as usual; found ones will be used without retraining.
- Verify that loading (not training) occurred:
  - Console shows: `Loaded Q-table from <dir> for sigma=<value>`
  - Epsilon does not decay (final ≈ initial), and action usage/steps in the summary remain zeros if no training ran.
  - Wall time is small (only I/O + summary).
  - If you accidentally passed a file to `--load-qdir`, training will run instead; you’ll notice epsilon decays and action usage is non‑zero.

## Viewing the Q‑table (as a table)
You can render the learned Q as a matrix similar to the paper’s figure.

- CSV dump (open in Excel/Sheets):
  - `python lms_sar_rl.py --load-qdir . --noise-levels 0.25 --dump-q-csv Q_sigma=0.25.csv`
- Print selected ENOB rows in console:
  - A few rows: `--print-q 8.5,10.5,11.8`
  - A range: `--print-q 8.0:12.1:0.5`
  - All rows: `--print-q all` (411 rows; long)

Orientation
- Rows: ENOB states (index i → ENOB = `8.00 + 0.01*i`).
- Columns: actions/step sizes labeled as `mu_w1..mu_w9` and shown numerically under the header (i.e., `[2^-7 .. 2^-15]`).
- Policy selection at a state: `argmax` across that row.

## Training Modes
- Standard training
  - Example: `python LMS-based_SAR_RL/lms_sar_rl.py --episodes 150 --max-steps 250 --alpha 0.1 --gamma 0.8 --epsilon 0.5`
- Paper-like grouped training (recommended for reproducible benchmarking)
  - Example: `python LMS-based_SAR_RL/lms_sar_rl.py --paper-demo --groups 50 --per-group-episodes 10 --mismatch-sigma 0.05 --noise-levels 0.25`
  - Behavior: For each group, a fixed true weight (mismatch) is sampled and held constant across its episodes; the digital weight estimate is reinitialized per episode. This mimics running calibration across multiple dies with consistent mismatch per die.

## Output Summary
Each run prints a summary:
- Q-table shape, episodes/steps, alpha/gamma, epsilon (start/final), wall time.
- ENOB metrics: mean over last 10 episodes and its std.
- Steps per episode (mean/min/max) and total reward stats.
- Action usage counts/percentages across the run.
- Recommended step size at representative ENOB states (8.5, 10.5, 11.8).
- One-step calibration demo: true weight, initial estimate `win0`, measured correlation `E_PN`, chosen step, `win1`, and ENOB before/after.

## Internals & Extensibility
- Environment model (`SARADCEnv`)
  - `measure_correlation(win)`: simulates PN-based correlation, `E_PN ≈ corr_gain * (true_weight - win) + noise`.
  - `compute_enob(win)`: derives ENOB from an SNR model: `ENOB = (SNR_dB - 1.76)/6.02`, with penalties for comparator noise and mismatch.
- Reward
  - Built from ENOB delta with slight penalty for stagnation and a small bonus in high-ENOB region (≥ 11.5).
- Application (lookup table)
  - In deployment, estimate ENOB/state and select the action as `argmax(Q[state, :])` (no exploration) and perform one LMS update.
- Swap-in real measurements
  - Replace `measure_correlation` with your PN correlation measurement.
  - Replace/provide your ENOB estimator during application; map ENOB to state via the provided utility.

## Reproducibility
- Control randomness via `--seed`. For multiple noise levels, the seed is offset by index for each level.

## Notes
- The environment and reward shaping are simple proxies for demonstration. Calibrate gains/noise models to your silicon or detailed simulator for best results.
- In paper-demo mode, `--episodes` is ignored; total episodes are `groups * per_group_episodes`.

---

If you need additional artifacts (plots of ENOB over episodes, saving action histograms, or multi-run aggregations), those can be added on request.
