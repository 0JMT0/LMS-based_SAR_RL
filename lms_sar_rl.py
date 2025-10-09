import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional


# ------------------------------------------------------------
# LMS-based background calibration with Q-learning optimization
# Implements the pseudocode and technical notes provided.
# ------------------------------------------------------------


@dataclass
class RLConfig:
    alpha: float = 0.1   # Q-learning learning rate
    gamma: float = 0.8   # discount factor
    epsilon: float = 0.5 # epsilon-greedy exploration
    num_episodes: int = 200
    max_steps: int = 300


def build_action_set() -> np.ndarray:
    """9 LMS step sizes A = [2^-7, ..., 2^-15]."""
    return np.array([2.0 ** (-k) for k in range(7, 16)], dtype=float)


def build_enob_states() -> np.ndarray:
    """ENOB states discretized from 8.00 to 12.10 (inclusive) with 0.01 step => 411 states."""
    return np.arange(8.00, 12.11, 0.01, dtype=float)


def enob_to_state_index(enob: float, enob_states: np.ndarray) -> int:
    """Map a continuous ENOB value to the nearest discrete state index."""
    # Clip to state range then map to nearest 0.01 grid.
    enob_c = float(np.clip(enob, enob_states[0], enob_states[-1]))
    idx = int(round((enob_c - enob_states[0]) / 0.01))
    idx = max(0, min(idx, len(enob_states) - 1))
    return idx


def build_reward_matrix(enob_states: np.ndarray) -> np.ndarray:
    """Construct a reward matrix R[s, s'] based on ENOB improvement.

    Reward design highlights:
    - Positive for ENOB improvement (proportional to delta ENOB).
    - Slight negative for degradation or stagnation.
    - Bonus for reaching high-ENOB region (>= 11.5).
    """
    n = len(enob_states)
    R = np.zeros((n, n), dtype=float)
    for s in range(n):
        for sp in range(n):
            delta = enob_states[sp] - enob_states[s]
            # Base reward scaled by ENOB delta
            r = 10.0 * delta
            # Penalize stagnation lightly
            if sp == s:
                r -= 0.2
            # Bonus for high ENOB region
            if enob_states[sp] >= 11.5:
                r += 0.5
            R[s, sp] = r
    return R


def lms_calibration(win: float, step: float, E_PN_bout_i: float) -> float:
    """LMS calibration update (Eq.2): win+1 = win + step * E_PN[b_out, i]."""
    return win + step * E_PN_bout_i


def q_update(Q: np.ndarray, st: int, at: int, rt: float, stp1: int, alpha: float, gamma: float) -> None:
    """Q-learning Bellman update (Eq.3)."""
    Q[st, at] = (1.0 - alpha) * Q[st, at] + alpha * (rt + gamma * np.max(Q[stp1, :]))


def select_action(Q: np.ndarray, st: int, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy selection over actions (step sizes)."""
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    return int(np.argmax(Q[st, :]))


class SARADCEnv:
    """Simple SAR ADC calibration environment model for a single bit weight.

    This environment abstracts measurement of the correlation E_PN with PN injection
    and maps the calibration error to an ENOB estimate using a simple SNR model.

    Notes:
    - True analog bit weight is (1 + mismatch). Digital weight estimate `win` is adapted.
    - E_PN measurement is proportional to (true - estimate) with injected noise.
    - ENOB is derived from SNR_dB with ideal quantization SNR and extra penalties for
      comparator noise and mismatch error.
    """

    def __init__(
        self,
        n_bits: int = 12,
        sigma_cmp_lsb: float = 0.25,
        corr_gain: float = 0.8,
        corr_noise_lsb: float = 0.02,
        rng: Optional[np.random.Generator] = None,
        cal_bit: Optional[int] = None,
    ) -> None:
        self.n_bits = n_bits
        self.sigma_cmp_lsb = float(sigma_cmp_lsb)
        self.corr_gain = float(corr_gain)
        self.corr_noise_lsb = float(corr_noise_lsb)
        self.rng = rng if rng is not None else np.random.default_rng()
        # Internal state
        self.true_weight = 1.0
        self.win = 1.0
        # Which bit is being calibrated (index 0 = LSB). Default: middle bit.
        self.cal_bit = int(cal_bit) if cal_bit is not None else max(0, min(n_bits - 1, n_bits // 2))

    def reset(self, mismatch_sigma: float = 0.01, win_init_sigma: float = 0.02, fixed_true_weight: Optional[float] = None) -> float:
        """Reset environment with random true mismatch and initial digital weight.

        Returns the initial win.
        """
        # True analog bit weight: fixed or 1 + gaussian mismatch
        if fixed_true_weight is None:
            self.true_weight = 1.0 + float(self.rng.normal(0.0, mismatch_sigma))
        else:
            self.true_weight = float(fixed_true_weight)
        # Initialize digital estimate around nominal with some offset
        self.win = 1.0 + float(self.rng.normal(0.0, win_init_sigma))
        return self.win

    # --- Measurement & ENOB model ---
    def measure_correlation(self, win: float) -> float:
        """Simulate E_PN[b_out, i] measurement with PN injection.

        E_PN ≈ corr_gain * (true_weight - win) + noise.
        Noise represents finite averaging and comparator noise folded into the estimator.
        Units are arbitrary but consistent with LMS update step.
        """
        noise = float(self.rng.normal(0.0, self.corr_noise_lsb))
        return self.corr_gain * (self.true_weight - win) + noise

    def compute_enob(self, win: float) -> float:
        """Map current weight estimate to an ENOB via an SNR model.

        SNR_ideal_dB = 6.02*N + 1.76 (quantization only)
        Additional degradation:
          - comparator noise: relative to quantization RMS noise
          - mismatch error: modeled as ENOB penalty proportional to |error|
        """
        # Ideal quantization SNR
        snr_ideal_db = 6.02 * self.n_bits + 1.76

        # Comparator noise contribution vs quantization RMS noise
        q_rms_lsb = 1.0 / np.sqrt(12.0)
        noise_ratio = (self.sigma_cmp_lsb / q_rms_lsb) if q_rms_lsb > 0 else 0.0
        snr_cmp_loss_db = 10.0 * np.log10(1.0 + noise_ratio * noise_ratio)

        # Mismatch induced ENOB loss (heuristic): proportional to |error| in %
        err = abs(win - self.true_weight)
        # If 1% error ~ 1e-2 => ~0.6 ENOB loss (tunable)
        enob_loss_from_mismatch = 60.0 * err  # 0.6 ENOB per 1% absolute error

        # Combine: translate ENOB mismatch loss into SNR dB loss equivalently
        # 1 ENOB ≈ 6.02 dB SNR
        snr_mismatch_loss_db = 6.02 * enob_loss_from_mismatch

        snr_db = snr_ideal_db - snr_cmp_loss_db - snr_mismatch_loss_db
        enob = (snr_db - 1.76) / 6.02
        return float(enob)

    def simulate_sar_code(self, x_norm: Optional[float] = None, use_true_weights: bool = True) -> Tuple[int, float]:
        """Simulate an N-bit SAR conversion and return (code_int, x_norm).

        - x_norm in [0,1]; if None uses RNG uniform.
        - use_true_weights: if True, the calibrated bit weight is self.true_weight; otherwise ideal 1.0.
        """
        N = self.n_bits
        rng = self.rng
        x = float(rng.random()) if x_norm is None else float(x_norm)
        # Input in LSB units (unipolar)
        full_scale = (1 << N) - 1
        x_lsb = x * full_scale

        # Build weight factors per bit (LSB index 0)
        factors = np.ones(N, dtype=float)
        factors[self.cal_bit] = (self.true_weight if use_true_weights else 1.0)

        code = 0
        for j in range(N - 1, -1, -1):
            trial = code | (1 << j)
            # DAC in LSB units using true weights for comparator decisions
            dac = 0.0
            m = trial
            k = 0
            while m:
                if m & 1:
                    dac += (1 << k) * factors[k]
                m >>= 1
                k += 1
            # Comparator (with noise)
            resid = x_lsb - dac + rng.normal(0.0, self.sigma_cmp_lsb)
            if resid >= 0.0:
                code = trial
        return code, x

    # Explicit PN + re-quantizer correlation measurement for detailed tracing
    def measure_correlation_explicit(self, win: float, samples: int = 256, pn_amp: float = 0.01) -> Tuple[float, float, float, np.ndarray]:
        """Return (E_PN, dout_mean, rq_mean, dout_bits) using explicit PN and 1-bit re-quantizer.

        PN sequence pn[n] in {+1, -1}; residue r[n] = (true-win) + pn_amp*pn[n] + noise;
        re-quantizer rq[n] = sign(r[n]) in {+1, -1}; digital code Dout[n] = (rq[n] > 0) in {0,1}.
        E_PN ≈ mean(pn[n] * rq[n]); dout_mean = mean(Dout); rq_mean = mean(rq).
        """
        rng = self.rng
        pn = rng.choice([-1.0, 1.0], size=int(samples))
        noise = rng.normal(0.0, self.sigma_cmp_lsb, size=int(samples))
        resid = (self.true_weight - win) + pn_amp * pn + noise
        rq = np.sign(resid)
        rq[rq == 0.0] = 1.0
        dout = (rq > 0).astype(float)
        E_PN = float(np.mean(pn * rq))
        dout_mean = float(np.mean(dout))
        rq_mean = float(np.mean(rq))
        return E_PN, dout_mean, rq_mean, dout


def get_reward(st: int, stp1: int, R: np.ndarray) -> float:
    return float(R[st, stp1])


class _TraceCSV:
    def __init__(self, path: str, include_bits: bool = False, include_code: bool = False) -> None:
        import csv, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self.include_bits = include_bits
        self.include_code = include_code
        self._w.writerow([
            "mode", "sigma_cmp_lsb", "group", "episode", "ep_in_group", "step",
            "state_before", "enob_before", "action", "step_value", "E_PN",
            "dout_mean", "rq_mean", "dout_bits", "sar_code", "sar_x_norm",
            "win_before", "win_after", "err_before", "err_after", "state_after",
            "enob_after", "reward", "epsilon",
        ])

    def write(self, row: Tuple) -> None:
        self._w.writerow(list(row))

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def train_q_table(
    cfg: RLConfig,
    env: SARADCEnv,
    enob_states: np.ndarray,
    actions: np.ndarray,
    R: np.ndarray,
    seed: Optional[int] = None,
    *,
    pn_explicit: bool = False,
    corr_samples: int = 256,
    pn_amp: float = 0.01,
    trace: Optional[_TraceCSV] = None,
    trace_code: bool = False,
    sar_fixed: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Train Q-table for LMS step-size selection via Q-learning.

    Returns:
        Q: learned Q-table with shape [len(enob_states), len(actions)]
        logs: dict of arrays (episode_mean_enob, episode_last_enob)
    """
    rng = np.random.default_rng(seed)
    S = len(enob_states)
    A = len(actions)
    Q = np.zeros((S, A), dtype=float)

    episode_mean_enob = np.zeros(cfg.num_episodes, dtype=float)
    episode_last_enob = np.zeros(cfg.num_episodes, dtype=float)
    episode_steps = np.zeros(cfg.num_episodes, dtype=int)
    episode_total_reward = np.zeros(cfg.num_episodes, dtype=float)
    epsilon_hist = np.zeros(cfg.num_episodes, dtype=float)
    action_counts = np.zeros(A, dtype=int)

    for ep in range(cfg.num_episodes):
        # Reset environment
        win = env.reset()
        enob = env.compute_enob(win)
        st = enob_to_state_index(enob, enob_states)

        enob_sum = 0.0
        total_reward = 0.0
        epsilon_hist[ep] = cfg.epsilon
        for t in range(cfg.max_steps):
            # 1) select action
            at = select_action(Q, st, cfg.epsilon, rng)
            step = actions[at]
            action_counts[at] += 1

            # 2) LMS calibration using measured correlation
            if pn_explicit:
                E_PN, dout_mean, rq_mean, dout_vec = env.measure_correlation_explicit(win, samples=corr_samples, pn_amp=pn_amp)
            else:
                E_PN = env.measure_correlation(win)
                dout_mean, rq_mean = np.nan, np.nan
                dout_vec = None
            win_prev = win
            enob_prev = enob
            st_prev = st
            win_new = lms_calibration(win_prev, step, E_PN)

            # 3) compute next ENOB and reward
            enob_new = env.compute_enob(win_new)
            stp1 = enob_to_state_index(enob_new, enob_states)
            rt = get_reward(st, stp1, R)
            total_reward += rt

            # 4) Q update
            q_update(Q, st, at, rt, stp1, cfg.alpha, cfg.gamma)

            # 5) state, weight update
            win = win_new
            enob = enob_new
            st = stp1

            enob_sum += enob

            # Trace row: standard mode
            if trace is not None:
                if getattr(trace, "include_bits", False) and dout_vec is not None:
                    bits_list = ["1" if v > 0.5 else "0" for v in dout_vec.tolist()]
                    # Group into bytes for readability: '01011010 11100011 ...'
                    if len(bits_list) > 0:
                        groups = ["".join(bits_list[i:i+8]) for i in range(0, len(bits_list), 8)]
                        dout_bits_str = " ".join(groups)
                    else:
                        dout_bits_str = ""
                else:
                    dout_bits_str = ""
                if getattr(trace, "include_code", False) and trace_code:
                    sar_x = sar_fixed if sar_fixed is not None else env.rng.random()
                    sar_code, x_norm = env.simulate_sar_code(x_norm=sar_x, use_true_weights=True)
                else:
                    sar_code, x_norm = "", ""
                err_before = abs(win_prev - env.true_weight)
                err_after = abs(win_new - env.true_weight)
                trace.write((
                    "standard",
                    env.sigma_cmp_lsb,
                    "",
                    ep + 1,
                    "",
                    t + 1,
                    st_prev,
                    enob_prev,
                    at,
                    float(step),
                    float(E_PN),
                    float(dout_mean),
                    float(rq_mean),
                    dout_bits_str,
                    sar_code,
                    x_norm,
                    float(win_prev),
                    float(win_new),
                    float(err_before),
                    float(err_after),
                    stp1,
                    enob_new,
                    float(rt),
                    float(cfg.epsilon),
                ))

            # Optional early stopping if near top ENOB and stable
            if st >= (S - 2):
                break

        episode_mean_enob[ep] = enob_sum / (t + 1)
        episode_last_enob[ep] = enob
        episode_steps[ep] = t + 1
        episode_total_reward[ep] = total_reward

        # Mild epsilon decay to favor exploitation later
        cfg.epsilon = max(0.05, cfg.epsilon * 0.995)

    logs = {
        "episode_mean_enob": episode_mean_enob,
        "episode_last_enob": episode_last_enob,
        "episode_steps": episode_steps,
        "episode_total_reward": episode_total_reward,
        "epsilon_hist": epsilon_hist,
        "action_counts": action_counts,
    }
    return Q, logs


def train_q_table_grouped(
    cfg: RLConfig,
    env: SARADCEnv,
    enob_states: np.ndarray,
    actions: np.ndarray,
    R: np.ndarray,
    groups: int = 50,
    per_group_episodes: int = 10,
    mismatch_sigma: float = 0.05,
    seed: Optional[int] = None,
    show_progress: bool = True,
    *,
    pn_explicit: bool = False,
    corr_samples: int = 256,
    pn_amp: float = 0.01,
    trace: Optional[_TraceCSV] = None,
    trace_code: bool = False,
    sar_fixed: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Paper-like grouped training: 50 groups, each 10 episodes => 500 episodes.

    For each group, a fixed true_weight (mismatch ~ N(0, mismatch_sigma)) is sampled
    and held constant across the 10 episodes; only the initial digital estimate win
    is re-initialized at each episode.
    """
    rng = np.random.default_rng(seed)
    S = len(enob_states)
    A = len(actions)
    Q = np.zeros((S, A), dtype=float)

    total_eps = groups * per_group_episodes
    episode_mean_enob = np.zeros(total_eps, dtype=float)
    episode_last_enob = np.zeros(total_eps, dtype=float)
    episode_steps = np.zeros(total_eps, dtype=int)
    episode_total_reward = np.zeros(total_eps, dtype=float)
    epsilon_hist = np.zeros(total_eps, dtype=float)
    action_counts = np.zeros(A, dtype=int)
    group_true_weights = np.zeros(groups, dtype=float)

    ep_idx = 0
    for g in range(groups):
        true_w = 1.0 + float(rng.normal(0.0, mismatch_sigma))
        group_true_weights[g] = true_w
        if show_progress:
            print(f"[Group {g+1}/{groups}] true_weight={true_w:.6f}")

        for k in range(per_group_episodes):
            epsilon_hist[ep_idx] = cfg.epsilon
            # Reset win but keep true mismatch fixed
            win = env.reset(win_init_sigma=0.02, fixed_true_weight=true_w)
            enob = env.compute_enob(win)
            st = enob_to_state_index(enob, enob_states)

            enob_sum = 0.0
            total_reward = 0.0
            for t in range(cfg.max_steps):
                at = select_action(Q, st, cfg.epsilon, rng)
                step = actions[at]
                action_counts[at] += 1

                if pn_explicit:
                    E_PN, dout_mean, rq_mean, dout_vec = env.measure_correlation_explicit(win, samples=corr_samples, pn_amp=pn_amp)
                else:
                    E_PN = env.measure_correlation(win)
                    dout_mean, rq_mean = np.nan, np.nan
                    dout_vec = None
                win_prev = win
                enob_prev = enob
                st_prev = st
                win_new = lms_calibration(win_prev, step, E_PN)

                enob_new = env.compute_enob(win_new)
                stp1 = enob_to_state_index(enob_new, enob_states)
                rt = get_reward(st, stp1, R)
                total_reward += rt

                q_update(Q, st, at, rt, stp1, cfg.alpha, cfg.gamma)

                win = win_new
                enob = enob_new
                st = stp1
                enob_sum += enob

                # Trace row: grouped mode
                if trace is not None:
                    if getattr(trace, "include_bits", False) and dout_vec is not None:
                        bits_list = ["1" if v > 0.5 else "0" for v in dout_vec.tolist()]
                        if len(bits_list) > 0:
                            groups = ["".join(bits_list[i:i+8]) for i in range(0, len(bits_list), 8)]
                            dout_bits_str = " ".join(groups)
                        else:
                            dout_bits_str = ""
                    else:
                        dout_bits_str = ""
                    if getattr(trace, "include_code", False) and trace_code:
                        sar_x = sar_fixed if sar_fixed is not None else env.rng.random()
                        sar_code, x_norm = env.simulate_sar_code(x_norm=sar_x, use_true_weights=True)
                    else:
                        sar_code, x_norm = "", ""
                    err_before = abs(win_prev - env.true_weight)
                    err_after = abs(win_new - env.true_weight)
                    trace.write((
                        "grouped",
                        env.sigma_cmp_lsb,
                        g + 1,
                        ep_idx + 1,
                        k + 1,
                        t + 1,
                        st_prev,
                        enob_prev,
                        at,
                        float(step),
                        float(E_PN),
                        float(dout_mean),
                        float(rq_mean),
                        dout_bits_str,
                        sar_code,
                        x_norm,
                        float(win_prev),
                        float(win_new),
                        float(err_before),
                        float(err_after),
                        stp1,
                        enob_new,
                        float(rt),
                        float(cfg.epsilon),
                    ))

                if st >= (S - 2):
                    break

            episode_mean_enob[ep_idx] = enob_sum / (t + 1)
            episode_last_enob[ep_idx] = enob
            episode_steps[ep_idx] = t + 1
            episode_total_reward[ep_idx] = total_reward

            if show_progress:
                print(
                    f"  ep {k+1:02d}/{per_group_episodes}: last_ENOB={enob:.3f}, steps={t+1}, totalR={total_reward:.2f}, eps={cfg.epsilon:.3f}"
                )

            # epsilon decay per episode
            cfg.epsilon = max(0.05, cfg.epsilon * 0.995)
            ep_idx += 1

    logs = {
        "episode_mean_enob": episode_mean_enob,
        "episode_last_enob": episode_last_enob,
        "episode_steps": episode_steps,
        "episode_total_reward": episode_total_reward,
        "epsilon_hist": epsilon_hist,
        "action_counts": action_counts,
        "group_true_weights": group_true_weights,
    }
    return Q, logs


def export_logs_csv(path: str, logs: Dict[str, np.ndarray]) -> None:
    import csv
    keys = [
        "episode_mean_enob",
        "episode_last_enob",
        "episode_steps",
        "episode_total_reward",
        "epsilon_hist",
    ]
    n = len(logs[keys[0]]) if keys[0] in logs else 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + keys)
        for i in range(n):
            row = [i + 1]
            for k in keys:
                v = logs.get(k)
                row.append(None if v is None else (v[i] if i < len(v) else None))
            w.writerow(row)


def save_q_table(Q: np.ndarray, sigma: float, out_dir: str) -> str:
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"Q_sigma={sigma:g}.npy"
    path = os.path.join(out_dir, fname)
    np.save(path, Q)
    return path


def load_q_table(sigma: float, in_dir: str) -> Optional[np.ndarray]:
    import os
    fname = f"Q_sigma={sigma:g}.npy"
    path = os.path.join(in_dir, fname)
    if os.path.isfile(path):
        return np.load(path)
    return None


def dump_q_table_to_csv(Q: np.ndarray, actions: np.ndarray, enob_states: np.ndarray, out_path: str) -> str:
    """Save Q as CSV with ENOB as first column and columns for each action (step)."""
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    header = "ENOB," + ",".join(f"{a:.8f}" for a in actions)
    data = np.column_stack([enob_states, Q])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    return out_path


def parse_enob_targets(spec: str, enob_states: np.ndarray) -> np.ndarray:
    """Parse ENOB targets from string. Supports:
    - comma separated list: "8.5,10.5,11.8"
    - range: "8.0:12.1:0.5" (step optional; default 0.5)
    - "all": returns full ENOB grid (be careful; 411 rows)
    """
    s = spec.strip().lower()
    if s == "all":
        return enob_states.copy()
    if ":" in s and "," not in s:
        parts = [p for p in s.split(":") if p]
        if len(parts) < 2:
            raise ValueError("Range must be 'start:end[:step]'")
        start = float(parts[0])
        end = float(parts[1])
        step = float(parts[2]) if len(parts) >= 3 else 0.5
        return np.arange(start, end + 1e-9, step, dtype=float)
    # comma-separated list
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return np.array(vals, dtype=float)


def print_q_rows(Q: np.ndarray, actions: np.ndarray, enob_states: np.ndarray, targets: np.ndarray, max_rows: int = 60) -> None:
    """Pretty-print selected rows of Q as an ASCII table."""
    # Cap excessive rows to keep console readable
    if len(targets) > max_rows:
        print(f"[print-q] Requested {len(targets)} rows; showing first {max_rows}.")
        targets = targets[:max_rows]

    # Header
    col_labels = [f"mu_w{i+1}" for i in range(len(actions))]
    header1 = f"{'ENOB':>7} | " + " ".join(f"{lbl:>12}" for lbl in col_labels)
    header2 = f"{'':>7} | " + " ".join(f"{a:>12.8f}" for a in actions)
    print(header1)
    print(header2)
    print("-" * len(header1))

    # Rows
    for en in targets:
        idx = enob_to_state_index(float(en), enob_states)
        row = Q[idx, :]
        values = " ".join(f"{v:>12.6f}" for v in row)
        print(f"{en:7.2f} | {values}")


def calibrate_once(
    win: float,
    Q: np.ndarray,
    actions: np.ndarray,
    enob_states: np.ndarray,
    measure_corr_fn: Callable[[float], float],
) -> float:
    """Application phase: lookup best step from Q for current ENOB state and update once.

    Args:
        win: current digital weight estimate
        Q: learned Q-table
        actions: action set (step sizes)
        enob_states: discretized ENOB states
        measure_corr_fn: function(win)->E_PN correlation measurement

    Returns:
        win_new after one LMS update using selected step size.
    """
    # Estimate ENOB from an external function is not available here; typically you'd
    # compute ENOB via system telemetry. For demo we approximate via a proxy mapping:
    # Users should replace `proxy_enob_estimate` with their actual estimator if available.
    def proxy_enob_estimate(w: float) -> float:
        # Proxy assumes higher |E_PN| => larger error => lower ENOB.
        # This is a rough online proxy if only correlation is observable.
        # Map |E_PN| in [0, ~0.1] to ENOB in [~8.0, ~12.0].
        e = abs(measure_corr_fn(w))
        e = float(np.clip(e, 0.0, 0.1))
        return 12.0 - 40.0 * e

    st = enob_to_state_index(proxy_enob_estimate(win), enob_states)
    at = int(np.argmax(Q[st, :]))
    step = float(actions[at])
    E_PN = float(measure_corr_fn(win))
    win_new = lms_calibration(win, step, E_PN)
    return win_new


def train_multiple_q_tables(
    cfg: RLConfig,
    noise_levels_lsb: Tuple[float, ...] = (0.1, 0.25, 0.5),
    seed: Optional[int] = 42,
) -> Dict[float, np.ndarray]:
    """Train multiple Q tables for different comparator noise levels.

    Returns dict: sigma_cmp_lsb -> Q-table.
    """
    actions = build_action_set()
    enob_states = build_enob_states()
    R = build_reward_matrix(enob_states)
    tables: Dict[float, np.ndarray] = {}

    for i, sigma in enumerate(noise_levels_lsb):
        env = SARADCEnv(sigma_cmp_lsb=float(sigma), rng=np.random.default_rng(seed + i))
        # Copy cfg to avoid mutating original epsilon decay, etc.
        local_cfg = RLConfig(
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            epsilon=cfg.epsilon,
            num_episodes=cfg.num_episodes,
            max_steps=cfg.max_steps,
        )
        Q, _ = train_q_table(local_cfg, env, enob_states, actions, R, seed=(seed + i))
        tables[float(sigma)] = Q
    return tables


if __name__ == "__main__":
    # CLI with options for episodes/steps/alpha/gamma/epsilon, CSV export,
    # Q-table save/load, and paper-like grouped demo.
    import time
    import argparse
    import os

    parser = argparse.ArgumentParser(description="LMS-based SAR ADC calibration with Q-learning")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=250, help="Max steps per episode")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning learning rate")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Initial epsilon for epsilon-greedy")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--noise-levels", type=str, default="0.25", help="Comma-separated comparator noise sigma (LSB), e.g. 0.1,0.25,0.5")
    parser.add_argument("--csv", type=str, default=None, help="Path to export per-episode CSV logs")
    parser.add_argument("--save-qdir", type=str, default=None, help="Directory to save Q-tables (.npy) per noise level")
    parser.add_argument("--load-qdir", type=str, default=None, help="Directory to load Q-tables (.npy) per noise level")
    parser.add_argument("--paper-demo", action="store_true", help="Run 50-group x 10-episodes paper-like demo (5% mismatch)")
    parser.add_argument("--groups", type=int, default=50, help="Number of groups for paper-like demo")
    parser.add_argument("--per-group-episodes", type=int, default=10, help="Episodes per group for paper-like demo")
    parser.add_argument("--mismatch-sigma", type=float, default=0.05, help="Mismatch sigma for true weight in paper-like demo")
    parser.add_argument("--dump-q-csv", type=str, default=None, help="Dump Q-table to CSV. If multiple noise levels, suffix _sigma=<val> is added or files are placed inside the given directory.")
    parser.add_argument("--print-q", type=str, default=None, help='Print Q rows. Accepts comma list (e.g., "8.5,10.5"), a range "8.0:12.1:0.5", or "all" (411 rows).')
    parser.add_argument("--trace-csv", type=str, default=None, help="Record per-step iteration trace (incl. Dout and next weight) to CSV.")
    parser.add_argument("--trace-dout", action="store_true", help="Include digital code Dout bits per step in the trace CSV (requires --pn-explicit).")
    parser.add_argument("--pn-explicit", action="store_true", help="Use explicit PN + 1-bit re-quantizer to compute correlation (enables Dout logging).")
    parser.add_argument("--corr-samples", type=int, default=256, help="Samples used to estimate correlation when --pn-explicit is on.")
    parser.add_argument("--pn-amp", type=float, default=0.01, help="PN injection amplitude used in explicit correlation mode.")
    parser.add_argument("--trace-sar-code", action="store_true", help="Simulate and record a full N-bit SAR code per step in the trace CSV.")
    parser.add_argument("--sar-fixed", type=float, default=None, help="Normalized input in [0,1] for SAR code simulation; if omitted, uses random per step.")
    args = parser.parse_args()

    actions = build_action_set()
    enob_states = build_enob_states()
    R = build_reward_matrix(enob_states)

    # Parse noise levels
    noise_levels = [float(s) for s in args.noise_levels.split(",") if s.strip()]

    def print_summary(Q: np.ndarray, logs: Dict[str, np.ndarray], cfg: RLConfig, sigma: float, t0: float, t1: float) -> None:
        print("=== Training Summary ===")
        print(f"Noise sigma (LSB): {sigma:g}")
        print(f"Q-table shape: {Q.shape}")
        print(f"Episodes: {cfg.num_episodes}, Max steps/ep: {cfg.max_steps}")
        print(f"Alpha: {cfg.alpha}, Gamma: {cfg.gamma}")
        print(f"Final epsilon: {cfg.epsilon:.3f} (start {logs['epsilon_hist'][0]:.3f})")
        print(f"Action set (steps): {np.array2string(actions, precision=7)}")
        print(f"ENOB state range: {enob_states[0]:.2f}..{enob_states[-1]:.2f} (n={len(enob_states)})")
        print(f"Wall time: {(t1 - t0):.2f}s")

        last10 = logs["episode_last_enob"][-10:]
        last10_mean = float(np.mean(last10))
        last10_std = float(np.std(last10, ddof=0))
        print(f"Mean ENOB (last 10 eps): {last10_mean:.3f} +/- {last10_std:.3f}")

        steps_mean = float(np.mean(logs["episode_steps"]))
        steps_min = int(np.min(logs["episode_steps"]))
        steps_max = int(np.max(logs["episode_steps"]))
        print(f"Steps/episode: mean {steps_mean:.1f}, min {steps_min}, max {steps_max}")

        rew_mean = float(np.mean(logs["episode_total_reward"]))
        rew_last10 = float(np.mean(logs["episode_total_reward"][-10:]))
        print(f"Total reward/episode: mean {rew_mean:.2f}, last10 {rew_last10:.2f}")

        counts = logs["action_counts"].astype(int)
        total_actions = int(np.sum(counts))
        with np.errstate(divide='ignore', invalid='ignore'):
            perc = 100.0 * counts / total_actions if total_actions > 0 else np.zeros_like(counts)
        print("Action usage (counts, %):")
        for i, (c, p, a) in enumerate(zip(counts, perc, actions)):
            print(f"  a[{i}] step={a:.8f}: {c} ({p:.1f}%)")

        # Recommended step sizes for representative ENOB states
        def best_step_for_enob(target_enob: float) -> float:
            idx = enob_to_state_index(target_enob, enob_states)
            at = int(np.argmax(Q[idx, :]))
            return float(actions[at])

        for e in (8.5, 10.5, 11.8):
            print(f"Best step at ENOB {e:.2f}: {best_step_for_enob(e):.8f}")

    # Iterate over noise levels
    for i, sigma in enumerate(noise_levels):
        rng = np.random.default_rng(args.seed + i)
        env = SARADCEnv(n_bits=12, sigma_cmp_lsb=float(sigma), corr_gain=0.8, corr_noise_lsb=0.02, rng=rng)
        cfg = RLConfig(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, num_episodes=args.episodes, max_steps=args.max_steps)

        Q = None
        # Attempt load
        if args.load_qdir:
            Q = load_q_table(sigma=float(sigma), in_dir=args.load_qdir)
            if Q is not None:
                print(f"Loaded Q-table from {args.load_qdir} for sigma={sigma:g}")

        t0 = time.time()
        logs: Dict[str, np.ndarray]
        # Optional per-step trace
        trace_logger = None
        if args.trace_csv:
            trace_path = args.trace_csv
            import os
            if len(noise_levels) > 1 and trace_path.lower().endswith(".csv"):
                root, ext = os.path.splitext(trace_path)
                trace_path = f"{root}_sigma={sigma:g}{ext}"
            elif os.path.isdir(args.trace_csv):
                trace_path = os.path.join(args.trace_csv, f"trace_sigma={sigma:g}.csv")
            # include bits if explicitly requested or when PN explicit mode is used
            trace_logger = _TraceCSV(
                trace_path,
                include_bits=(args.trace_dout or args.pn_explicit),
                include_code=args.trace_sar_code,
            )

        if Q is None:
            if args.paper_demo:
                # paper-like grouped run: groups x per_group_episodes overrides episodes count
                cfg.num_episodes = args.groups * args.per_group_episodes
                Q, logs = train_q_table_grouped(
                    cfg, env, enob_states, actions, R,
                    groups=args.groups,
                    per_group_episodes=args.per_group_episodes,
                    mismatch_sigma=args.mismatch_sigma,
                    seed=args.seed + i,
                    show_progress=True,
                    pn_explicit=args.pn_explicit,
                    corr_samples=args.corr_samples,
                    pn_amp=args.pn_amp,
                    trace=trace_logger,
                    trace_code=args.trace_sar_code,
                    sar_fixed=args.sar_fixed,
                )
            else:
                Q, logs = train_q_table(
                    cfg, env, enob_states, actions, R,
                    seed=args.seed + i,
                    pn_explicit=args.pn_explicit,
                    corr_samples=args.corr_samples,
                    pn_amp=args.pn_amp,
                    trace=trace_logger,
                    trace_code=args.trace_sar_code,
                    sar_fixed=args.sar_fixed,
                )
        else:
            # If loaded, we can still generate placeholder logs for the demo by running 1 eval episode
            # but here we just construct empty logs with zeros for consistency.
            logs = {
                "episode_mean_enob": np.zeros(cfg.num_episodes, dtype=float),
                "episode_last_enob": np.zeros(cfg.num_episodes, dtype=float),
                "episode_steps": np.zeros(cfg.num_episodes, dtype=int),
                "episode_total_reward": np.zeros(cfg.num_episodes, dtype=float),
                "epsilon_hist": np.zeros(cfg.num_episodes, dtype=float),
                "action_counts": np.zeros(len(actions), dtype=int),
            }
        t1 = time.time()
        if trace_logger is not None:
            trace_logger.close()
            print(f"Saved iteration trace to {trace_path}")

        print_summary(Q, logs, cfg, sigma=float(sigma), t0=t0, t1=t1)

        # Dump Q-table to CSV if requested
        if args.dump_q_csv:
            out_path = args.dump_q_csv
            import os
            if len(noise_levels) > 1 and out_path.lower().endswith(".csv"):
                root, ext = os.path.splitext(out_path)
                out_path = f"{root}_sigma={sigma:g}{ext}"
            elif os.path.isdir(args.dump_q_csv):
                out_path = os.path.join(args.dump_q_csv, f"Q_sigma={sigma:g}.csv")
            saved_csv = dump_q_table_to_csv(Q, actions, enob_states, out_path)
            print(f"Saved Q CSV to {saved_csv}")

        # Print selected Q rows as ASCII table if requested
        if args.print_q:
            try:
                targets = parse_enob_targets(args.print_q, enob_states)
                print_q_rows(Q, actions, enob_states, targets)
            except Exception as e:
                print(f"[print-q] Failed to parse/print rows: {e}")

        # Export CSV logs if requested
        if args.csv:
            out_path = args.csv
            if len(noise_levels) > 1 and out_path.lower().endswith(".csv"):
                root, ext = os.path.splitext(out_path)
                out_path = f"{root}_sigma={sigma:g}{ext}"
            elif os.path.isdir(args.csv):
                out_path = os.path.join(args.csv, f"logs_sigma={sigma:g}.csv")
            export_logs_csv(out_path, logs)
            print(f"Saved CSV logs to {out_path}")

        # Save Q-table if requested
        if args.save_qdir:
            saved = save_q_table(Q, sigma=float(sigma), out_dir=args.save_qdir)
            print(f"Saved Q-table to {saved}")

        # One-step calibration demo on this noise level
        win0 = env.reset()
        true_w = env.true_weight
        E0 = env.measure_correlation(win0)
        e_clip = float(np.clip(abs(E0), 0.0, 0.1))
        enob_proxy = 12.0 - 40.0 * e_clip
        st0 = enob_to_state_index(enob_proxy, enob_states)
        at0 = int(np.argmax(Q[st0, :]))
        step0 = float(actions[at0])
        win1 = lms_calibration(win0, step0, E0)

        enob0 = env.compute_enob(win0)
        enob1 = env.compute_enob(win1)
        err0 = abs(win0 - true_w)
        err1 = abs(win1 - true_w)

        print("=== One-Step Calibration Demo ===")
        print(f"True weight: {true_w:.6f}")
        print(f"win0: {win0:.6f}, |error|: {err0:.6f}")
        print(f"Measured E_PN: {E0:.6f}, chosen step: {step0:.8f}")
        print(f"win1: {win1:.6f}, |error|: {err1:.6f}")
        print(f"ENOB before: {enob0:.3f}, after one step: {enob1:.3f}")
