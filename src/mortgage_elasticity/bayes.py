"""The model: one Bayesian state-space model of weekly lodgement counts (PyMC).

    L_kt ~ BetaBinomial(M_kt, pi_kt c, (1 - pi_kt) c)
    logit pi_kt = alpha_k + g_t + seg_trend_kt + J_kt - beta_k x_kt(h)

    log u_k = mu_u + a_inv (INV - 1/2) + a_io (IO - 1/2) + c_lvr[band_k],   beta_k = u_k / (1 - s0_k)
    g_t          bank-wide random walk (or free week effects, time_effect = "fe")
    seg_trend_kt random walk per segment, innovations zero-sum across segments each week
    J_kt         one permanent level shift per logged non-price change
    x_k(h)       = gap_raw_k1 + W(h) (gap_raw_k - gap_raw_k1), the gap phased in with half-life h

g_t replaces free week dummies on purpose: dummies absorb every bank-wide price move, a smooth
trend lets those moves inform beta. The cost is one assumption: no unlogged bank-wide shock lands in
the same weeks as a bank-wide price move.

The soft-min temperature must equal 1 / beta_k, but r* is precomputed data. `estimate` solves that
with an outer loop of short fits, then runs the full fit at the final T.
"""

import pickle
import warnings
from dataclasses import dataclass, replace

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from mortgage_elasticity.config import Config
from mortgage_elasticity.model import lag_kernel
from mortgage_elasticity.panel import Prepared
from mortgage_elasticity.runlog import log

KEY = ["mu_u", "a_inv", "a_io", "h", "c", "u"]  # parameters whose ESS decides reliability
GLOBAL = ["h", "mu_u", "a_inv", "a_io", "c_lvr", "sigma_g", "sigma_v", "c"]  # global.csv


@dataclass
class Fit:
    """Posterior draws (first axis = draw) and what is needed to forecast from them."""
    prep: Prepared
    cfg: Config
    draws: dict[str, np.ndarray]
    mask: np.ndarray  # segment-weeks that were in the likelihood
    diag: dict  # sampler diagnostics
    prior_log_u: np.ndarray | None = None  # prior draws of log u_k, for prior_share
    T_path: pd.DataFrame | None = None  # soft-min temperature per iteration
    T_converged: bool | None = None
    placebo_step: np.ndarray | None = None  # fake price step of a placebo run

    @property
    def n(self) -> int:
        return len(self.draws["h"])

    def x(self) -> np.ndarray:
        """Lagged gap per draw (draws x periods x segments): model.lag run for every draw's h."""
        v, phi = self.prep.gap_raw, (1 - 2.0 ** (-1 / self.draws["h"]))[:, None]
        out = np.empty((self.n, *v.shape), np.float32)
        p = np.broadcast_to(v[0], (self.n, v.shape[1])).copy()
        for t in range(len(v)):
            p += phi * (v[t] - p)
            out[:, t] = p
        return out

    def pi(self) -> np.ndarray:
        """Fitted share per draw: draws x periods x segments."""
        eta = self.draws["level"] - self.draws["beta"][:, None, :] * self.x()
        if self.placebo_step is not None:
            px = np.stack([lag_kernel(len(self.placebo_step), h) @ self.placebo_step
                           for h in self.draws["h"]])
            eta = eta - self.draws["placebo"][:, None, None] * px
        return 1 / (1 + np.exp(-eta))

    def q(self, name: str) -> np.ndarray:
        """Reporting quantiles (low, middle, high) of a parameter, over draws."""
        return np.quantile(self.draws[name], self.cfg.q, axis=0)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "Fit":
        with open(path, "rb") as f:
            return pickle.load(f)


def _sampler(cfg: Config) -> str:
    if cfg.nuts_sampler != "auto":
        return cfg.nuts_sampler
    for name in ("nutpie", "numpyro"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return "pymc"


def lag_kernel_pt(n: int, phi):
    """model.lag_kernel as a tensor function of phi: W[t, j] = phi (1 - phi)^(t - j), j <= t."""
    d = np.subtract.outer(np.arange(n), np.arange(n))
    return phi * pt.exp(np.maximum(d, 0) * pt.log1p(-phi)) * (d >= 0)


def build_model(prep: Prepared, cfg: Config, mask: np.ndarray | None = None,
                placebo_step: np.ndarray | None = None) -> pm.Model:
    """mask: segment-weeks in the likelihood (default: every week with market lodgements).
    placebo_step: periods x segments fake price step (bp) that gets its own coefficient."""
    n, K = prep.shape
    mask = prep.obs if mask is None else mask & prep.obs
    dims = prep.dims
    is_inv = (dims["purpose"] == "INV").to_numpy(float)
    is_io = (dims["repayment"] == "IO").to_numpy(float)
    band = dims["lvr_band"].map({b: i for i, b in enumerate(cfg.lvr_bands)}).to_numpy()
    other = prep.moves("other")
    jump_on = np.zeros((len(other), n, K))  # event x week x segment: 1 from the event week on
    for e, ev in enumerate(other.itertuples()):
        jump_on[e, ev.t:, [prep.segs.index(s) for s in ev.segments]] = 1.0

    with pm.Model() as model:
        # sensitivity: uplift per bp, pooled across purpose / repayment / LVR band
        mu_u = pm.Normal("mu_u", np.log(cfg.prior_u_median), cfg.prior_u_log_sd)
        a_inv = pm.Normal("a_inv", 0, cfg.prior_dim_sd)
        a_io = pm.Normal("a_io", 0, cfg.prior_dim_sd)
        sigma_lvr = pm.HalfNormal("sigma_lvr", cfg.prior_lvr_sd)
        lvr_raw = pm.ZeroSumNormal("lvr_raw", sigma=1.0, shape=len(cfg.lvr_bands))
        c_lvr = pm.Deterministic("c_lvr", sigma_lvr * lvr_raw)
        log_u = mu_u + a_inv * (is_inv - 0.5) + a_io * (is_io - 0.5) + c_lvr[band]
        if cfg.segment_residual:
            sigma_eps = pm.HalfNormal("sigma_eps", cfg.segment_residual_sd)
            log_u = log_u + sigma_eps * pm.Normal("eps_raw", 0, 1, shape=K)
        log_u = pm.Deterministic("log_u", log_u)
        u = pm.Deterministic("u", pt.exp(log_u))
        beta = pm.Deterministic("beta", u / (1 - prep.s0))

        # natural share: segment constant + bank-wide trend + segment trend (+ jumps, covariates)
        alpha = pm.Normal("alpha", np.log(prep.s0 / (1 - prep.s0)), cfg.prior_alpha_sd, shape=K)
        if cfg.time_effect == "rw":
            sigma_g = pm.HalfNormal("sigma_g", cfg.prior_sigma_g)
            g_raw = pm.Normal("g_raw", 0, 1, shape=n - 1)
            g = pt.concatenate([pt.zeros(1), sigma_g * pt.cumsum(g_raw)])
        else:  # sum-to-zero week effects: identified against alpha
            g = pm.ZeroSumNormal("g_fe", sigma=cfg.prior_fe_sd, shape=n)
        level = alpha + g[:, None]
        if K > 1:  # zero-sum across segments, so common drift stays in g
            sigma_v = pm.HalfNormal("sigma_v", cfg.prior_sigma_v)
            v_raw = pm.ZeroSumNormal("v_raw", sigma=1.0, shape=(n - 1, K))
            level = level + pt.concatenate([pt.zeros((1, K)), sigma_v * pt.cumsum(v_raw, axis=0)])
        if cfg.jumps and len(other):
            jump = pm.Normal("jump", 0, cfg.prior_jump_sd, shape=len(other))
            level = level + pt.tensordot(jump, jump_on, axes=1)
        if prep.Z is not None:
            gamma = pm.Normal("gamma", 0, cfg.prior_gamma_sd, shape=prep.Z.shape[-1])
            level = level + pt.dot(prep.Z, gamma)
        level = pm.Deterministic("level", level)

        # price: the gap phased in with half-life h, inside the graph because h is a parameter
        h = pm.Truncated("h", pm.LogNormal.dist(np.log(cfg.prior_h_median), cfg.prior_h_log_sd),
                         lower=cfg.h_bounds[0], upper=cfg.h_bounds[1])
        W = lag_kernel_pt(n, 1 - 2.0 ** (-1 / h))
        x = prep.gap_raw[0] + pt.dot(W, prep.gap_raw - prep.gap_raw[0])
        eta = level - beta * x
        if placebo_step is not None:
            placebo = pm.Normal("placebo", 0, cfg.placebo_coef_sd)
            eta = eta - placebo * pt.dot(W, placebo_step)

        c = pm.LogNormal("c", np.log(cfg.prior_c_median), cfg.prior_c_log_sd)
        pi = pm.math.invlogit(eta)[np.nonzero(mask)]
        pm.BetaBinomial("L", n=prep.M[mask], alpha=pi * c, beta=(1 - pi) * c, observed=prep.L[mask])
    return model


def _ds(group):
    """A group of the sampler output as an xarray Dataset (newer ArviZ returns a DataTree)."""
    return group.to_dataset() if hasattr(group, "to_dataset") else group


def _diagnose(idata, cfg: Config) -> dict:
    post = _ds(idata.posterior)
    free = [v for v in post.data_vars if v not in ("level", "log_u", "beta", "c_lvr")]
    rhat = az.rhat(post[free])
    ess = az.ess(post[[v for v in KEY if v in post]])
    stats = _ds(idata.sample_stats)
    diag = {
        "max_rhat": float(max(float(rhat[v].max()) for v in rhat.data_vars)),
        "min_ess": float(min(float(ess[v].min()) for v in ess.data_vars)),
        "divergences": int(stats["diverging"].sum()) if "diverging" in stats else 0,
        "u_rhat": rhat["u"].to_numpy(), "u_ess": ess["u"].to_numpy(),
        "chains": post.sizes["chain"], "draws": post.sizes["draw"],
    }
    diag["reliable"] = bool(diag["max_rhat"] <= cfg.max_rhat and diag["min_ess"] >= cfg.min_ess
                            and diag["divergences"] == 0)
    return diag


def fit(prep: Prepared, cfg: Config, mask: np.ndarray | None = None, chains: int | None = None,
        tune: int | None = None, draws: int | None = None, prior: bool = False,
        placebo_step: np.ndarray | None = None, label: str = "fit", check: bool = True) -> Fit:
    """One NUTS fit. prior=True also draws from the prior (for prior_share and the prior check).
    check=False reports the sampler diagnostics without the unreliable-run warning (short fits)."""
    model = build_model(prep, cfg, mask, placebo_step)
    with model:
        idata = pm.sample(
            draws=draws or cfg.draws, tune=tune or cfg.tune, chains=chains or cfg.chains,
            target_accept=cfg.target_accept, random_seed=cfg.seed, nuts_sampler=_sampler(cfg),
            progressbar=False, compute_convergence_checks=False)
        prior_log_u = None
        if prior:
            pr = pm.sample_prior_predictive(cfg.prior_draws, var_names=["log_u"],
                                            random_seed=cfg.seed)
            prior_log_u = _ds(pr.prior)["log_u"].to_numpy().reshape(-1, prep.shape[1])
    diag = _diagnose(idata, cfg)
    log(f"{label}: {diag['chains']} chains x {diag['draws']} draws, max R-hat "
        f"{diag['max_rhat']:.3f}, min bulk ESS (key parameters) {diag['min_ess']:.0f}, "
        f"{diag['divergences']} divergences")
    if check and not diag["reliable"]:
        warnings.warn(f"{label}: UNRELIABLE RUN (need R-hat <= {cfg.max_rhat}, ESS >= "
                      f"{cfg.min_ess:.0f}, no divergences)")
    post = _ds(idata.posterior)
    out = {v: post[v].to_numpy().reshape(-1, *post[v].shape[2:])
           for v in post.data_vars if not v.endswith("_raw")}
    out["level"] = out["level"].astype(np.float32)
    return Fit(prep=prep, cfg=cfg, draws=out, diag=diag, prior_log_u=prior_log_u,
               mask=prep.obs if mask is None else mask & prep.obs, placebo_step=placebo_step)


def prior_T(s0: np.ndarray, segs, cfg: Config) -> pd.Series:
    """Starting temperature: T = 1 / beta at the prior median (or the fixed override)."""
    if cfg.softmin_T_bps is not None:
        return pd.Series(float(cfg.softmin_T_bps), index=segs)
    return pd.Series(np.clip((1 - s0) / cfg.prior_u_median, *cfg.T_bounds), index=segs)


def estimate(prep: Prepared, cfg: Config) -> Fit:
    """Solve T_k = 1 / beta_k by damped iteration over short fits, then run the full fit."""
    path, converged = None, None
    if prep.comp is not None and cfg.softmin_T_bps is None:
        T, path, converged = prep.T, [prep.T.rename("start")], False
        for it in range(1, cfg.T_max_iter + 1):
            short = fit(prep, cfg, chains=cfg.T_chains, tune=cfg.T_tune, draws=cfg.T_draws,
                        label=f"T loop {it}", check=False)
            beta = pd.Series(np.median(short.draws["beta"], axis=0), index=prep.segs)
            T_new = np.sqrt(T / beta).clip(*cfg.T_bounds)  # damped: geometric mean of old and new
            change = float((T_new / T - 1).abs().max())
            log(f"T loop {it}: T = {T_new.min():.1f}-{T_new.max():.1f}bp "
                f"(median {T_new.median():.1f}), largest change {change:.1%}")
            T = T_new
            prep = prep.with_T(T, cfg)
            path.append(T.rename(f"iter_{it}"))
            if change < cfg.T_tol:
                converged = True
                break
        path = pd.concat(path, axis=1).rename_axis("segment").reset_index()
        if not converged:
            warnings.warn(f"SOFT-MIN TEMPERATURE DID NOT CONVERGE in {cfg.T_max_iter} iterations "
                          f"(last change {change:.1%}, tolerance {cfg.T_tol:.0%}): the benchmark "
                          "is not consistent with the estimated sensitivity")
    full = fit(prep, cfg, prior=True, label="full fit")
    return replace(full, T_path=path, T_converged=converged)
