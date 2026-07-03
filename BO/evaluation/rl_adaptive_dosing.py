#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Adaptive Dosing Agent for GENEVO2 Biosensor.

WHAT THIS DOES
==============
Replaces the fixed "release dose when R > threshold" rule in pkpd_closed_loop.py
with a learned Q-policy that adapts dose magnitude to each patient's trajectory.

The fixed rule's problem: it always releases the same dose for any R > 1.08,
ignoring whether R = 1.1 (barely elevated, small dose optimal) or R = 2.5
(severely elevated CKD, larger dose needed). The RL agent learns the right
dose per state.

STATE SPACE (discretized Q-table)
----------------------------------
s = (R_bin, cycle_bin, dose_bin)

  R_bin       : 8 bins over [0.8, 3.0] — measured composite ratio
  cycle_bin   : 12 bins (one per month)
  dose_bin    : 5 bins for cumulative dose so far [0, 0.5, 1.0, 1.5, 2.0+]

Total states: 8 x 12 x 5 = 480 (small enough for tabular Q-learning)

ACTION SPACE
------------
a in {0: skip, 1: half, 2: three-quarter, 3: full, 4: one-and-quarter}
Dose fractions: {0.0, 0.5, 0.75, 1.0, 1.25} × standard_dose

REWARD
------
At each cycle:
  r_bmd     = bmd_gain_this_cycle / BMD_MAX_PER_DOSE   # normalized
  r_overdose = -3.0 * (dose/D_SAFE - 1)^2 if dose > D_SAFE else 0
  r_missed   = -0.5 if true_R > threshold but dose = 0
  reward = r_bmd + r_overdose + r_missed

Terminal bonus: +2.0 if patient normalizes (final_R < threshold)
Terminal penalty: -1.0 if patient never normalizes

LIMITATIONS (honest)
---------------------
- Trained on synthetic data from pkpd_closed_loop.py (not real patients)
- Crosstalk between patients is modeled only through noise, not inter-patient
  ODE variation
- Q-table is small and may not generalize to edge cases

USAGE
-----
  # Train:
  python BO/evaluation/rl_adaptive_dosing.py --train --n-episodes 5000 \
      --out BO/bo_results/rl_policy.pkl

  # Evaluate vs fixed dosing:
  python BO/evaluation/rl_adaptive_dosing.py --eval \
      --policy BO/bo_results/rl_policy.pkl \
      --n-patients 200 --all-scenarios
"""

import numpy as np
import json
import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PK/PD constants (mirrors pkpd_closed_loop.py)
# ---------------------------------------------------------------------------
T_HALF_DAYS       = 6.9
K_EL              = np.log(2) / T_HALF_DAYS
F_SUPP_SOST       = 0.27
F_SUPP_CTX        = 0.55
F_SUPP_P1NP       = 0.20
BMD_MAX_PER_DOSE  = 0.06 / 4.0
BMD_T_RECOVERY    = 30.0
BMD_BASELINE_GCMS = 0.775
DRUG_THRESHOLD_FRAC = 1.08
D_HALF            = 0.15
D_SAFE            = 0.50
ALPHA_OD          = 3.0
K_RELEASE         = 1.0

SENSOR_HALF_LIFE_DAYS = 180.0
_K_DEG_SENSOR = np.log(2) / SENSOR_HALF_LIFE_DAYS

_NOMINAL_CONCS = {
    "healthy":  {"scl": 0.375, "ctx": 0.200, "p1np": 0.350},
    "pmo_mild": {"scl": 0.5625, "ctx": 0.300, "p1np": 0.385},
    "pmo":      {"scl": 0.875,  "ctx": 0.500, "p1np": 0.525},
    "ckd_mbd":  {"scl": 1.125,  "ctx": 0.500, "p1np": 0.625},
}

# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------
_R_BINS     = np.array([0.8, 1.0, 1.1, 1.2, 1.5, 1.8, 2.2, 3.0, np.inf])
_CYCLE_BINS = 12
_DOSE_BINS  = np.array([0.0, 0.3, 0.7, 1.2, 1.8, np.inf])
# Rate-of-change bins: falling / stable / rising / rapidly-rising
_DR_BINS    = np.array([-np.inf, -0.03, 0.03, 0.10, np.inf])

ACTIONS = [0.0, 0.5, 0.75, 1.0, 1.25]   # dose multipliers (× standard_dose)
N_ACTIONS = len(ACTIONS)


def _r_bin(R: float) -> int:
    return int(np.searchsorted(_R_BINS, R, side='right') - 1)


def _dose_bin(cumulative_dose: float) -> int:
    return int(np.searchsorted(_DOSE_BINS, cumulative_dose, side='right') - 1)


def _dr_bin(dR: float) -> int:
    return int(np.searchsorted(_DR_BINS, dR, side='right') - 1)


def _state(R: float, cycle: int, cumulative_dose: float,
           dR: float = 0.0) -> Tuple[int, int, int, int]:
    return (
        np.clip(_r_bin(R),         0, len(_R_BINS)    - 2),
        np.clip(cycle,             0, _CYCLE_BINS      - 1),
        np.clip(_dose_bin(cumulative_dose), 0, len(_DOSE_BINS) - 2),
        np.clip(_dr_bin(dR),       0, len(_DR_BINS)   - 2),
    )


# ---------------------------------------------------------------------------
# Q-table agent
# ---------------------------------------------------------------------------

class QDosageAgent:
    """
    Tabular Q-learning agent for adaptive dosing.

    Q[r_bin, cycle_bin, dose_bin, action] -> expected cumulative reward
    """

    def __init__(self, alpha: float = 0.05, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.999):
        n_r    = len(_R_BINS) - 1
        n_cyc  = _CYCLE_BINS
        n_dose = len(_DOSE_BINS) - 1
        n_dr   = len(_DR_BINS) - 1
        self.Q = np.zeros((n_r, n_cyc, n_dose, n_dr, N_ACTIONS), dtype=np.float32)
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._episode = 0

    def select_action(self, state: Tuple[int, int, int, int],
                      greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        r_b, cyc_b, dose_b, dr_b = state
        return int(np.argmax(self.Q[r_b, cyc_b, dose_b, dr_b]))

    def update(self, state, action, reward, next_state, done):
        r_b,  cyc_b,  dose_b,  dr_b  = state
        r_nb, cyc_nb, dose_nb, dr_nb = next_state
        q_cur = self.Q[r_b,  cyc_b,  dose_b,  dr_b,  action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[r_nb, cyc_nb, dose_nb, dr_nb])
        self.Q[r_b, cyc_b, dose_b, dr_b, action] += self.alpha * (td_target - q_cur)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._episode += 1

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'Q': self.Q, 'alpha': self.alpha, 'gamma': self.gamma,
                'epsilon': self.epsilon, 'episode': self._episode,
                'Q_shape': list(self.Q.shape),
            }, f)
        logger.info(f"Q-policy saved to {path}")

    @classmethod
    def load(cls, path: str) -> "QDosageAgent":
        with open(path, 'rb') as f:
            d = pickle.load(f)
        agent = cls()
        expected_shape = list(agent.Q.shape)
        loaded_shape   = d.get('Q_shape', list(d['Q'].shape))
        if loaded_shape != expected_shape:
            logger.warning(
                f"Q-table shape mismatch: file={loaded_shape}, current={expected_shape}. "
                "Starting fresh — retrain the policy."
            )
            return agent  # return fresh agent rather than silently loading wrong shape
        agent.Q        = d['Q']
        agent.alpha    = d.get('alpha', 0.05)
        agent.gamma    = d.get('gamma', 0.95)
        agent.epsilon  = d.get('epsilon', 0.0)
        agent._episode = d.get('episode', 0)
        return agent


# ---------------------------------------------------------------------------
# Environment: single-patient episode
# ---------------------------------------------------------------------------

class PatientEnv:
    """
    Single-patient closed-loop RL environment.

    Mirrors PKPDClosedLoop but exposes a step() interface and supports
    variable dose actions from the agent.
    """

    def __init__(self, config: Dict, scenario: str, n_cycles: int = 12,
                 cycle_days: int = 28, noise_sigma: float = 0.15,
                 with_degradation: bool = True,
                 rng: Optional[np.random.RandomState] = None):
        self.config     = config
        self.scenario   = scenario
        self.n_cycles   = n_cycles
        self.cycle_days = cycle_days
        self.noise_sigma = noise_sigma
        self.with_degradation = with_degradation
        self.rng = rng or np.random.RandomState()

        self.kd      = float(config.get("kd_nm", 1.0))
        self.kd_ctx  = float(config.get("kd_ctx_nm", 1.0))
        self.kd_p1np = float(config.get("kd_p1np_nm", 1.0))
        self.w_ctx   = float(config.get("w_ctx", 0.1))
        self.w_p1np  = float(config.get("w_p1np", 0.1))
        self.w_scl   = max(0.0, 1.0 - self.w_ctx - self.w_p1np)

        concs = _NOMINAL_CONCS[scenario]
        self.scl0  = concs["scl"]
        self.ctx0  = concs["ctx"]
        self.p1np0 = concs["p1np"]

        # Patient-specific noise realization (drawn once)
        n = n_cycles
        if noise_sigma > 0:
            self._noise_scl  = self.rng.lognormal(0.0, noise_sigma, n)
            self._noise_ctx  = self.rng.lognormal(0.0, noise_sigma, n)
            self._noise_p1np = self.rng.lognormal(0.0, noise_sigma, n)
        else:
            self._noise_scl  = np.ones(n)
            self._noise_ctx  = np.ones(n)
            self._noise_p1np = np.ones(n)

        # Compute personalized threshold from baseline R
        R_base = self._composite_R(self.scl0, self.ctx0, self.p1np0)
        self.threshold = max(DRUG_THRESHOLD_FRAC, R_base * 0.90)

        self.prev_R_corrected: Optional[float] = None
        self.reset()

    def _occupancy(self, conc: float, kd: float) -> float:
        return float(conc / (kd + conc + 1e-12))

    def _composite_R(self, scl: float, ctx: float, p1np: float) -> float:
        h = _NOMINAL_CONCS["healthy"]
        th_scl_h  = self._occupancy(h["scl"],  self.kd)
        th_ctx_h  = self._occupancy(h["ctx"],  self.kd_ctx)
        th_p1np_h = self._occupancy(h["p1np"], self.kd_p1np)
        eps = 1e-12
        R = (
            self.w_scl  * self._occupancy(scl,  self.kd)   / max(th_scl_h,  eps) +
            self.w_ctx  * self._occupancy(ctx,  self.kd_ctx)  / max(th_ctx_h,  eps) +
            self.w_p1np * self._occupancy(p1np, self.kd_p1np) / max(th_p1np_h, eps)
        )
        return float(R)

    def reset(self) -> Tuple[int, int, int, int]:
        self.cycle            = 0
        self.C_drug           = 0.0
        self.bmd_total        = 0.0
        self.cum_dose         = 0.0
        self.prev_R_corrected = None
        R0 = self._composite_R(self.scl0, self.ctx0, self.p1np0)
        # Degradation at t=0 is always 1.0; R_corrected = R0
        return _state(R0, 0, 0.0, 0.0)

    def step(self, action_idx: int) -> Tuple[Tuple[int,int,int,int], float, bool, Dict]:
        """
        Apply action (dose multiplier) and advance one cycle.

        Returns (next_state, reward, done, info).
        Uses degradation-corrected R (R_measured / deg_factor) so the agent
        operates on disease-severity signal, not the attenuated sensor output.
        """
        cycle  = self.cycle
        t_day  = cycle * self.cycle_days

        # Sensor degradation at this cycle
        if self.with_degradation:
            deg = float(np.exp(-_K_DEG_SENSOR * t_day))
        else:
            deg = 1.0
        safe_deg = max(deg, 0.05)  # floor at 5% to prevent runaway noise amplification

        # Effective biomarker levels under current drug load + per-patient noise
        supp_sost  = F_SUPP_SOST  * self.C_drug
        supp_ctx   = F_SUPP_CTX   * self.C_drug
        supp_p1np  = F_SUPP_P1NP  * self.C_drug

        scl_eff  = self.scl0  * (1.0 - supp_sost)  * self._noise_scl[cycle]
        ctx_eff  = self.ctx0  * (1.0 - supp_ctx)   * self._noise_ctx[cycle]
        p1np_eff = self.p1np0 * (1.0 - supp_p1np)  * self._noise_p1np[cycle]

        R_true     = self._composite_R(scl_eff, ctx_eff, p1np_eff)
        R_measured = R_true * deg
        # Degradation-corrected signal: device compensates for its known decay curve
        R_corrected = R_measured / safe_deg

        # Rate of change (disease progression indicator)
        dR = R_corrected - self.prev_R_corrected if self.prev_R_corrected is not None else 0.0
        self.prev_R_corrected = R_corrected

        # Standard dose scales from R_corrected (true disease severity)
        std_dose  = K_RELEASE * max(0.0, (R_corrected - self.threshold) / self.threshold)
        dose_mult = ACTIONS[action_idx]
        dose      = std_dose * dose_mult

        # BMD gain with overdose penalty
        if dose > 0:
            bmd_gross = BMD_MAX_PER_DOSE * (dose / (dose + D_HALF))
            if dose > D_SAFE:
                od      = (dose / D_SAFE - 1.0) ** 2 * ALPHA_OD
                bmd_net = max(0.0, bmd_gross - od * BMD_MAX_PER_DOSE * 0.2)
            else:
                od      = 0.0
                bmd_net = bmd_gross
        else:
            bmd_net = 0.0
            od      = 0.0

        self.bmd_total += bmd_net
        self.cum_dose  += dose
        self.C_drug    += dose

        # ---- Clinical-aligned reward ----------------------------------------
        # Primary: BMD gain (normalized)
        r_bmd = bmd_net / BMD_MAX_PER_DOSE if BMD_MAX_PER_DOSE > 0 else 0.0

        # Safety: overdose penalty (patient safety critical)
        r_overdose = -3.0 * od if dose > D_SAFE else 0.0

        # Efficacy: penalize skipping when disease clearly active
        if R_corrected > self.threshold * 1.15 and dose_mult == 0.0:
            r_missed = -1.5   # severely elevated and skipped → strong penalty
        elif R_corrected > self.threshold and dose_mult == 0.0:
            r_missed = -0.75  # moderately elevated and skipped
        else:
            r_missed = 0.0

        # Efficiency: reward high BMD-per-dose (avoid wasteful overdosing)
        if dose > 1e-6:
            r_eff = float(np.clip(0.2 * bmd_net / (dose * BMD_MAX_PER_DOSE + 1e-9), 0.0, 0.5))
        else:
            r_eff = 0.0

        reward = float(r_bmd + r_overdose + r_missed + r_eff)

        # Drug elimination
        self.C_drug *= float(np.exp(-K_EL * self.cycle_days))
        self.C_drug  = min(self.C_drug, 3.0)

        self.cycle += 1
        done = (self.cycle >= self.n_cycles)

        if done:
            # Terminal: bonus/penalty proportional to residual disease burden
            if R_corrected < self.threshold:
                reward += 3.0   # patient normalized
            else:
                excess  = max(0.0, R_corrected - self.threshold)
                reward -= 2.0 + min(excess, 2.0)  # proportional penalty

        # Next-state: estimate R_corrected at next cycle from current R_true
        if not done:
            next_t         = self.cycle * self.cycle_days
            next_deg       = float(np.exp(-_K_DEG_SENSOR * next_t)) if self.with_degradation else 1.0
            next_safe_deg  = max(next_deg, 0.05)
            next_R_meas    = R_true * next_deg            # proxy (drug effect not yet computed)
            next_R_corr    = next_R_meas / next_safe_deg
            next_dR        = next_R_corr - R_corrected   # estimated next rate
        else:
            next_R_corr = R_corrected
            next_dR     = dR

        next_state = _state(next_R_corr,
                            self.cycle if not done else self.n_cycles - 1,
                            self.cum_dose,
                            next_dR)

        info = {
            "R_true":      R_true,
            "R_measured":  R_measured,
            "R_corrected": R_corrected,
            "dR":          dR,
            "dose":        dose,
            "bmd_this_cycle": bmd_net,
            "degradation": deg,
        }
        return next_state, reward, done, info


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    config: Dict,
    n_episodes: int = 5000,
    scenarios: Optional[List[str]] = None,
    n_cycles: int = 12,
    cycle_days: int = 28,
    noise_sigma: float = 0.15,
    with_degradation: bool = True,
    seed: int = 42,
    log_interval: int = 500,
) -> QDosageAgent:
    """
    Train a Q-learning agent on n_episodes of virtual patient simulations.

    Cycles through scenarios so the agent learns a scenario-agnostic policy.
    """
    if scenarios is None:
        scenarios = ["pmo_mild", "pmo", "ckd_mbd"]

    rng = np.random.RandomState(seed)
    agent = QDosageAgent()

    episode_rewards = []
    episode_bmd     = []

    for ep in range(n_episodes):
        scenario = scenarios[ep % len(scenarios)]
        patient_rng = np.random.RandomState(rng.randint(0, 2**31))

        env = PatientEnv(
            config=config,
            scenario=scenario,
            n_cycles=n_cycles,
            cycle_days=cycle_days,
            noise_sigma=noise_sigma,
            with_degradation=with_degradation,
            rng=patient_rng,
        )

        state  = env.reset()
        total_r = 0.0

        while True:
            action  = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state    = next_state
            total_r += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_r)
        episode_bmd.append(env.bmd_total / BMD_BASELINE_GCMS * 100.0)

        if (ep + 1) % log_interval == 0:
            r_mean = np.mean(episode_rewards[-log_interval:])
            b_mean = np.mean(episode_bmd[-log_interval:])
            logger.info(
                f"Episode {ep+1}/{n_episodes} | "
                f"eps={agent.epsilon:.3f} | "
                f"reward={r_mean:.3f} | "
                f"BMD={b_mean:+.1f}%"
            )

    return agent


# ---------------------------------------------------------------------------
# Evaluation: RL vs fixed dosing
# ---------------------------------------------------------------------------

def evaluate(
    agent: QDosageAgent,
    config: Dict,
    scenario: str = "pmo",
    n_patients: int = 200,
    n_cycles: int = 12,
    cycle_days: int = 28,
    noise_sigma: float = 0.15,
    with_degradation: bool = True,
    seed: int = 99,
) -> Dict:
    """
    Compare RL adaptive dosing vs fixed dosing on n_patients virtual patients.

    Fixed dosing = always apply full dose when R > threshold (action_idx=3).
    RL dosing = greedy policy from trained agent.
    """
    rng = np.random.RandomState(seed)
    rl_results   = []
    fix_results  = []

    for i in range(n_patients):
        patient_seed = rng.randint(0, 2**31)

        for mode in ("rl", "fixed"):
            patient_rng = np.random.RandomState(patient_seed)
            env = PatientEnv(
                config=config,
                scenario=scenario,
                n_cycles=n_cycles,
                cycle_days=cycle_days,
                noise_sigma=noise_sigma,
                with_degradation=with_degradation,
                rng=patient_rng,
            )
            state   = env.reset()
            total_bmd = 0.0
            total_dose = 0.0
            missed = 0

            while True:
                if mode == "rl":
                    action = agent.select_action(state, greedy=True)
                else:
                    # Fixed: use action_idx=3 (full dose) whenever R > threshold
                    action = 3
                next_state, _, done, info = env.step(action)
                total_dose += info["dose"]
                state = next_state
                if done:
                    break

            final_bmd = env.bmd_total / BMD_BASELINE_GCMS * 100.0
            final_R   = info["R_measured"]

            rec = {
                "bmd_pct": final_bmd,
                "total_dose": total_dose,
                "normalized": final_R < env.threshold,
            }
            if mode == "rl":
                rl_results.append(rec)
            else:
                fix_results.append(rec)

    def _stats(recs, key):
        vals = [r[key] for r in recs]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "scenario": scenario,
        "n_patients": n_patients,
        "with_degradation": with_degradation,
        "rl": {
            "bmd_pct":    _stats(rl_results, "bmd_pct"),
            "total_dose": _stats(rl_results, "total_dose"),
            "norm_rate":  float(np.mean([r["normalized"] for r in rl_results]) * 100),
        },
        "fixed": {
            "bmd_pct":    _stats(fix_results, "bmd_pct"),
            "total_dose": _stats(fix_results, "total_dose"),
            "norm_rate":  float(np.mean([r["normalized"] for r in fix_results]) * 100),
        },
        "bmd_lift_pp": float(
            np.mean([r["bmd_pct"] for r in rl_results]) -
            np.mean([r["bmd_pct"] for r in fix_results])
        ),
        "dose_reduction_pct": float(
            (1.0 - np.mean([r["total_dose"] for r in rl_results]) /
             max(np.mean([r["total_dose"] for r in fix_results]), 1e-9)) * 100.0
        ),
    }


# ---------------------------------------------------------------------------
# Default best config
# ---------------------------------------------------------------------------

def _default_config() -> Dict:
    return {
        "kd_nm":          5.81,
        "sensitivity":    4.57,
        "response_time_s": 600.0,
        "kd_ctx_nm":      0.278,
        "kd_p1np_nm":     0.953,
        "w_ctx":          0.155,
        "w_p1np":         0.459,
        "biosensor_type": "array",
        "noise_preset":   "realistic",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GENEVO2 RL Adaptive Dosing Agent")
    parser.add_argument("--train", action="store_true", help="Train Q-learning agent")
    parser.add_argument("--eval",  action="store_true", help="Evaluate RL vs fixed dosing")
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--policy", type=str, default="BO/bo_results/rl_policy.pkl")
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--n-patients", type=int, default=200)
    parser.add_argument("--n-cycles",   type=int, default=12)
    parser.add_argument("--noise",      type=float, default=0.15)
    parser.add_argument("--no-degradation", action="store_true")
    parser.add_argument("--all-scenarios",  action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = _default_config()
    if args.config_json:
        with open(args.config_json) as f:
            raw = json.load(f)
        if "biosensor_design" in raw:
            d = raw["biosensor_design"]
            for k in ("kd_nm", "sensitivity", "kd_ctx_nm", "kd_p1np_nm", "w_ctx", "w_p1np"):
                if k in d:
                    config[k] = d[k]
        else:
            config.update(raw)

    with_deg = not args.no_degradation

    if args.train:
        print("=" * 72)
        print(f"TRAINING RL DOSING AGENT  |  {args.n_episodes} episodes")
        print(f"Sensor degradation: {'ON' if with_deg else 'OFF'} (t1/2={SENSOR_HALF_LIFE_DAYS}d)")
        print("=" * 72)

        agent = train(
            config=config,
            n_episodes=args.n_episodes,
            n_cycles=args.n_cycles,
            noise_sigma=args.noise,
            with_degradation=with_deg,
            log_interval=max(1, args.n_episodes // 10),
        )

        policy_path = args.policy
        os.makedirs(os.path.dirname(policy_path) or ".", exist_ok=True)
        agent.save(policy_path)
        print(f"\nPolicy saved to: {policy_path}")

    if args.eval:
        if not os.path.exists(args.policy):
            print(f"ERROR: policy not found at {args.policy}")
            print("Run with --train first.")
            import sys; sys.exit(1)

        agent = QDosageAgent.load(args.policy)
        print("=" * 72)
        print(f"EVALUATING RL vs FIXED DOSING  |  n_patients={args.n_patients}")
        print(f"Sensor degradation: {'ON' if with_deg else 'OFF'}")
        print("=" * 72)

        scenarios = ["pmo_mild", "pmo", "ckd_mbd"] if args.all_scenarios else ["pmo"]
        all_results = {}

        for sc in scenarios:
            res = evaluate(
                agent=agent,
                config=config,
                scenario=sc,
                n_patients=args.n_patients,
                n_cycles=args.n_cycles,
                noise_sigma=args.noise,
                with_degradation=with_deg,
            )
            all_results[sc] = res

            print(f"\n[{sc.upper()}]")
            print(f"  RL   BMD: {res['rl']['bmd_pct']['mean']:+.1f}% +/- {res['rl']['bmd_pct']['std']:.1f}%  "
                  f"| Dose: {res['rl']['total_dose']['mean']:.3f}  "
                  f"| Norm: {res['rl']['norm_rate']:.0f}%")
            print(f"  Fix  BMD: {res['fixed']['bmd_pct']['mean']:+.1f}% +/- {res['fixed']['bmd_pct']['std']:.1f}%  "
                  f"| Dose: {res['fixed']['total_dose']['mean']:.3f}  "
                  f"| Norm: {res['fixed']['norm_rate']:.0f}%")
            lift = res['bmd_lift_pp']
            dose_red = res['dose_reduction_pct']
            flag = "[OK]" if lift >= 0 else "[WARN]"
            print(f"  {flag} RL lift: {lift:+.2f} pp BMD  |  Dose reduction: {dose_red:.1f}%")

        if args.out:
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {args.out}")

        print("\n" + "=" * 72)
