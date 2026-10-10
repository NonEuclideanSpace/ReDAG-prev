"""Flow-GRPO style SDE sampling with per-token log-probabilities, and the region-level clipped policy loss.

Convention (diffusers flow matching): x_sigma = (1 - sigma) x0 + sigma * eps, the model predicts v = eps - x0, and
sigma runs from 1 to 0. The SDE with the same marginals (Flow-GRPO, eq. 9-10):
  x_{sigma+dt} = x + [v + (s^2 / (2 sigma)) (x + (1 - sigma) v)] dt + s sqrt(-dt) eps,   s = a sqrt(sigma / (1 - sigma))
with dt = sigma_next - sigma < 0. At sigma = 1 the denominator uses the next sigma, as in the reference code.
"""
import math

import torch

LOG2PI = math.log(2.0 * math.pi)


def sde_coeffs(sigma, sigma_next, sigma_max, noise_level):
    """Scalar coefficients for one step. Returns (dt, std_dev_t, step_std)."""
    sigma = float(sigma)
    sigma_next = float(sigma_next)
    dt = sigma_next - sigma
    s_eff = sigma_max if sigma >= 1.0 else sigma
    std_dev_t = math.sqrt(sigma / (1.0 - s_eff)) * float(noise_level)
    step_std = std_dev_t * math.sqrt(max(0.0, -dt))
    return dt, std_dev_t, step_std


def sde_mean(x, v, sigma, sigma_next, sigma_max, noise_level):
    dt, sd, _ = sde_coeffs(sigma, sigma_next, sigma_max, noise_level)
    sigma = float(sigma)
    return x * (1.0 + sd ** 2 / (2.0 * sigma) * dt) + v * (1.0 + sd ** 2 * (1.0 - sigma) / (2.0 * sigma)) * dt


def ode_step(x, v, sigma, sigma_next):
    return x + v * (float(sigma_next) - float(sigma))


def sde_step(x, v, sigma, sigma_next, sigma_max, noise_level, noise=None, generator=None):
    """One stochastic step. Returns (x_next, mean, step_std)."""
    x = x.float()
    v = v.float()
    mean = sde_mean(x, v, sigma, sigma_next, sigma_max, noise_level)
    _, _, step_std = sde_coeffs(sigma, sigma_next, sigma_max, noise_level)
    if noise is None:
        noise = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return mean + step_std * noise, mean, step_std


def token_logprob(x_next, mean, step_std):
    """Per-token (position) log-probability: Gaussian log density averaged over channels. Shapes [B, N, C] -> [B, N]."""
    lp = -((x_next.float() - mean.float()) ** 2) / (2.0 * step_std ** 2) - math.log(step_std) - 0.5 * LOG2PI
    return lp.mean(-1)


def region_ratio_loss(lp_new, lp_old, rid, adv, w, eps_lo=1e-4, eps_hi=1e-4, kl=None, beta=0.0):
    """Region-level clipped surrogate.

    lp_new, lp_old: [B, N] per-token log-probs (new with grad, old detached).
    rid: [B, N] long, region id per token. adv: [B, R] region advantages. w: [B, R] region weights (sum to 1 per row).
    kl: optional [B, N] per-token KL estimate to the reference policy. Returns (loss, stats)."""
    B, N = lp_new.shape
    R = adv.shape[1]
    delta = lp_new - lp_old.detach()
    loss = lp_new.new_zeros(())
    clipfrac = 0.0
    nreg = 0
    for i in range(B):
        li = lp_new.new_zeros(())
        for j in range(R):
            m = rid[i] == j
            n = int(m.sum())
            if n == 0 or float(w[i, j]) <= 0:
                continue
            d = delta[i][m].mean()
            rho = torch.exp(d)
            a = float(adv[i, j])
            unclipped = -a * rho
            clipped = -a * torch.clamp(rho, 1.0 - eps_lo, 1.0 + eps_hi)
            li = li + float(w[i, j]) * torch.maximum(unclipped, clipped)
            clipfrac += float((rho < 1.0 - eps_lo) | (rho > 1.0 + eps_hi))
            nreg += 1
        loss = loss + li
    loss = loss / B
    if kl is not None and beta > 0:
        loss = loss + beta * kl.mean()
    return loss, dict(clipfrac=clipfrac / max(1, nreg), mean_delta=float(delta.detach().mean()), regions=nreg)


def sample_timesteps(scheduler_sigmas, n_sde=None):
    """Indices of the steps to train on (all SDE steps by default)."""
    T = len(scheduler_sigmas) - 1
    return list(range(T)) if n_sde is None else list(range(min(T, n_sde)))
