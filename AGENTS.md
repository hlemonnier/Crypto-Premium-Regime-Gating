# AGENT.md - Coding Guide (Fasanara Premiums)

This file is a coding reference spec for implementing the project quickly and reproducibly.
It is written to be usable by a human or an AI agent while coding.

## 0) Goal (definition of done)
Deliver a reproducible baseline (and optional Hawkes) that:
- builds a debiased premium p(t) from BTCUSDC vs BTCUSDT (futures/perp)
- robustly filters p(t) -> p_smooth(t), sigma_hat(t), z_t
- computes state variables (H_t, T_t, chi_t) and detects regimes (change-point baseline)
- outputs decisions: Trade / Widen / Risk-off
- produces: metrics.csv + 3 figures (and n(t) figure if Hawkes enabled)

## 1) Data contract (minimal)
Required columns:
- timestamp_utc (UTC datetime)
- symbol (string)
- price (float)  # choose ONE convention: mark recommended
Optional:
- venue (string), volume (float)

Rules:
- UTC everywhere, dedupe timestamps per symbol
- resample to ONE frequency (1s or 1m)
- forward-fill max 2 intervals; otherwise missing
- filter glitches (outlier returns / duplicated timestamps)
- convert raw -> Parquet early (data/processed)

## 2) Definitions (dictionary)
- p_naive(t) = log P(BTCUSDC) - log P(BTCUSDT)
- log(USDT/USDC)_hat(t): stablecoin proxy from cross-asset replication (median across assets)
- p(t) = p_naive(t) - log(USDT/USDC)_hat(t)
- p_smooth(t): robust filtered p(t)
- sigma_hat(t): robust local scale estimate
- z_t = (p(t) - p_smooth(t)) / sigma_hat(t)
- events: timestamps t_i where |z_t| > u
- m_t (mispricing): default m_t = p_smooth(t)
- H_t: Shannon entropy of discretized z on window W
- T_t: local temperature = sqrt(E[z^2] on W)
- chi_t: local susceptibility = Var(m on W)
- regime_t: transient vs stress (change-point baseline)

Optional Hawkes (if enabled):
- lambda(t) = mu + sum alpha * exp(-beta*(t-t_i))
- n(t) approx alpha/beta (branching ratio), must be < 1

## 3) Default parameters (starting point)
- resample: 1s (or 1m if heavy)
- W: 3600s (1h)
- u: 3.0 (events if |z| > 3)
- k: 2.0 (enter if |m| > k*T)
- depeg flag: delta_log=0.002 (~20 bps), L=5 minutes consecutive

If Hawkes:
- warmup: 48h
- rolling: 12h
- stability: enforce beta > alpha + eps with eps=0.01

## 4) Run order (pipeline)
1) ingest -> aligned table (UTC, resampled)
2) compute p_naive on target pair
3) estimate log(USDT/USDC)_hat using multiple underlyings; compute p(t)
4) compute on-chain divergence proxy and depeg_flag (delta_log, L)
5) robust filter -> p_smooth, sigma_hat, z_t; mark events |z|>u
6) stat-mech -> H, T, chi
7) regimes -> change-point baseline -> regime_t
8) decision rules -> Trade/Widen/Risk-off
9) backtest minimal -> metrics.csv + trade log
10) plots -> Figure 1/2/3 (and Hawkes figure if enabled)

## 5) Decision logic (baseline)
Priority: safety first.
- If depeg_flag active -> Risk-off (override everything)
- Else if regime == stress -> Risk-off or at least Widen
- Else (regime transient):
  - Trade only if |m_t| > k*T_t
  - Widen if T_t or chi_t exceed thresholds

Hawkes add-on:
- If n(t) > 0.85 -> Risk-off
- If n(t) > 0.70 -> Widen

## 6) Evaluation (what to report)
Compare:
- naive: mean-reversion using p_naive with fixed threshold
- gated: mean-reversion using p(t), regime gating, adaptive threshold k*T
- (optional) gated+hawkes

Report:
- Sharpe, max drawdown, turnover, PnL net after costs
- flip-rate (decision stability)
- ablation: debias only / +robust / +regime / (+hawkes)

## 7) Figures (must-have)
- Fig1: timeline p_naive vs p_smooth + regimes + events |z|>u + episode annotations
- Fig2: (no-Hawkes) T_t and chi_t / (Hawkes) n(t) with thresholds
- Fig3: phase space:
  - no-Hawkes: (T_t, |m_t|) with decision regions
  - Hawkes: (T_t, n(t)) with decision regions

## 8) Quick sanity checks
- outside stress, log(USDT/USDC)_hat should be near 0 and smooth
- sigma_hat should not collapse to ~0
- regime segments should not flip every minute
- depeg_flag should fire on USDC SVB window (10-11 Mar 2023) if data covers it

## 9) Commit name requirement
- Every time the agent generates or modifies something, it must also provide an associated commit name describing that exact change.
- Preferred format: `<type>: <short-description>` (example: `feat: add regime gating decision thresholds`).

---
Last updated: 2026-02-11
