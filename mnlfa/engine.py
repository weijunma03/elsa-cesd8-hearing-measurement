"""Two-factor binary probit MNLFA numerical engine.

No file/data access or fitting occurs on import. Callers supply authorized local
arrays. Anchors are identifying assumptions, not invariance findings. Marginal
likelihood weights are fixed case weights; this engine does not estimate survey
standard errors. All outputs are exploratory conditional-model quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr, roots_hermitenorm

STAGES = ("hearing_only", "latent_adjusted", "threshold_adjusted", "slope_adjusted")
FACTOR = np.array([0, 1, 1, 0, 0, 0, 0, 1], dtype=int)
ANCHORS = np.array([0, 3, 1, 2], dtype=int)
NONANCHORS = np.array([4, 5, 6, 7], dtype=int)
X_NAMES = ("hearing", "age10", "female")
ALL_PATTERNS = np.asarray(list(itertools.product([0., 1.], repeat=8)))
LOG_2PI = np.log(2 * np.pi)


@dataclass
class Grouped:
    groups: list
    n_rows: int
    weight_sum: float
    weighted_item_mean: np.ndarray


def group_rows(y, x, weights=None):
    """Aggregate equal covariates and equal eight-bit responses in memory.

    x columns: hearing (0/1), (age capped at 90 - 70)/10, female (0/1).
    n_rows retains the supplied row count; zero weights contribute nothing.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    if y.ndim != 2 or y.shape[1] != 8 or x.shape != (len(y), 3):
        raise ValueError("Expected y[n,8] and x[n,3]")
    if not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all():
        raise ValueError("Complete finite covariates and binary responses required")
    if not np.isin(x[:, [0, 2]], [0., 1.]).all():
        raise ValueError("Hearing and female must be binary 0/1")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(y),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Nonnegative finite weights with positive sum required")
    xu, xi = np.unique(x, axis=0, return_inverse=True)
    groups = []
    for i, xx in enumerate(xu):
        take = xi == i
        yy, yi = np.unique(y[take], axis=0, return_inverse=True)
        counts = np.bincount(yi, weights=w[take], minlength=len(yy))
        use = counts > 0
        if use.any():
            groups.append({"x": xx.copy(), "y": yy[use], "counts": counts[use]})
    return Grouped(groups, len(y), float(w.sum()), (w[:, None] * y).sum(axis=0) / w.sum())


class Model:
    """Simple-structure correlated two-factor model with fixed anchor set.

    P(Y_j=1 | theta,x) = Phi(b_j + x*delta_j
                               + exp(loga_j+x*omega_j)*theta[factor_j]).
    theta | x has mean beta*x, SD exp(gamma*x), and correlation
    tanh(r0+r_hearing*hearing). The reference x=(0,0,0) fixes both means at
    zero and both marginal SDs at one. Anchors have delta=omega=0 for all x.
    """
    def __init__(self, stage="hearing_only", q=21):
        if stage not in STAGES or not isinstance(q, (int, np.integer)) or q < 3:
            raise ValueError("Unknown stage or quadrature order below 3")
        self.stage, self.q = stage, int(q)
        self.k_lat = 1 if stage == "hearing_only" else 3
        self.k_delta = 3 if stage in ("threshold_adjusted", "slope_adjusted") else 1
        self.k_omega = 3 if stage == "slope_adjusted" else 1
        self.slices, self.names, self.bounds = {}, [], []
        self._add("b", [f"b_i{j+1}" for j in range(8)], (-6., 6.))
        self._add("loga", [f"loga_i{j+1}" for j in range(8)], (-2.5, 2.))
        self._add("beta", [f"mu_{f}_{X_NAMES[k]}" for f in ("affective", "somatic")
                           for k in range(self.k_lat)], (-3., 3.))
        self._add("gamma", [f"logsd_{f}_{X_NAMES[k]}" for f in ("affective", "somatic")
                            for k in range(self.k_lat)], (-1.25, 1.25))
        self._add("rho", ["atanh_rho_reference", "atanh_rho_hearing"], (-2.3, 2.3))
        self._add("delta", [f"intercept_i{j+1}_{X_NAMES[k]}" for j in NONANCHORS
                            for k in range(self.k_delta)], (-3., 3.))
        self._add("omega", [f"logslope_i{j+1}_{X_NAMES[k]}" for j in NONANCHORS
                            for k in range(self.k_omega)], (-1.25, 1.25))
        self.p = len(self.names)
        z, w = roots_hermitenorm(self.q)
        w = w / np.sqrt(2 * np.pi)
        self.z1, self.z2 = np.meshgrid(z, z, indexing="ij")
        self.z1, self.z2 = self.z1.ravel(), self.z2.ravel()
        self.quad_weights = np.outer(w, w).ravel()
        self.log_quad_weights = np.log(self.quad_weights)

    def _add(self, key, names, bound):
        start = len(self.names)
        self.names.extend(names)
        self.bounds.extend([bound] * len(names))
        self.slices[key] = slice(start, len(self.names))

    def to_components(self, t):
        t = np.asarray(t, float)
        if t.shape != (self.p,) or not np.isfinite(t).all():
            raise ValueError(f"Expected {self.p} finite parameters")
        beta, gamma = np.zeros((2, 3)), np.zeros((2, 3))
        delta, omega = np.zeros((8, 3)), np.zeros((8, 3))
        beta[:, :self.k_lat] = t[self.slices["beta"]].reshape(2, self.k_lat)
        gamma[:, :self.k_lat] = t[self.slices["gamma"]].reshape(2, self.k_lat)
        delta[NONANCHORS, :self.k_delta] = t[self.slices["delta"]].reshape(4, self.k_delta)
        omega[NONANCHORS, :self.k_omega] = t[self.slices["omega"]].reshape(4, self.k_omega)
        return {"b": t[self.slices["b"]].copy(), "loga": t[self.slices["loga"]].copy(),
                "beta": beta, "gamma": gamma, "rho": t[self.slices["rho"]].copy(),
                "delta": delta, "omega": omega}

    def pack(self, c):
        return np.r_[c["b"], c["loga"], np.asarray(c["beta"])[:, :self.k_lat].ravel(),
                     np.asarray(c["gamma"])[:, :self.k_lat].ravel(), c["rho"],
                     np.asarray(c["delta"])[NONANCHORS, :self.k_delta].ravel(),
                     np.asarray(c["omega"])[NONANCHORS, :self.k_omega].ravel()]

    def embed(self, t, from_model):
        """Copy an earlier-stage fit with all new coefficients exactly zero."""
        if STAGES.index(from_model.stage) > STAGES.index(self.stage):
            raise ValueError("Embedding only permits equal or expanding stages")
        return self.pack(from_model.to_components(t))

    def initial(self, grouped=None):
        t = np.zeros(self.p)
        t[self.slices["loga"]] = np.log(.8)
        t[self.slices["rho"]] = [np.arctanh(.5), 0.]
        if grouped is not None:
            from scipy.special import ndtri
            t[self.slices["b"]] = ndtri(np.clip(grouped.weighted_item_mean, .01, .99)) * np.sqrt(1+.8**2)
        return t

    def _node_values(self, c, x):
        x = np.asarray(x, float)
        mu, sd = c["beta"] @ x, np.exp(c["gamma"] @ x)
        rr = c["rho"] @ np.array([1., x[0]])
        rho = np.tanh(rr)
        # sech is stable for the bounded parameter/covariate region and does
        # not suffer cancellation when rho is close to one.
        sr = 1. / np.cosh(rr)
        latent = np.vstack((mu[0]+sd[0]*self.z1,
                            mu[1]+sd[1]*(rho*self.z1+sr*self.z2)))
        a = np.exp(c["loga"] + c["omega"] @ x)
        b = c["b"] + c["delta"] @ x
        eta = b[:, None] + a[:, None] * latent[FACTOR]
        return eta, a, latent, mu, sd, rho, sr

    def fg(self, t, grouped, return_group_scores=False):
        """Weighted mean negative marginal log likelihood and analytic gradient.

        Optional group scores are gradients of *negative summed* loglikelihood
        contributions, before normalization. They are not household scores.
        """
        c = self.to_components(t)
        gradient = np.zeros(self.p)
        total_nll = 0.
        group_scores = []
        for d in grouped.groups:
            x, y, counts = d["x"], d["y"], d["counts"]
            eta, a, latent, mu, sd, rho, sr = self._node_values(c, x)
            lp, ln = log_ndtr(eta), log_ndtr(-eta)
            logs = y @ lp + (1.-y) @ ln + self.log_quad_weights[None, :]
            ll = logsumexp(logs, axis=1)
            total_nll -= float(counts @ ll)
            weighted_post = np.exp(logs-ll[:, None]) * counts[:, None]
            pos, neg = y.T @ weighted_post, (1.-y).T @ weighted_post
            logphi = -.5 * (eta*eta + LOG_2PI)
            # Negative likelihood derivative with respect to eta_jq.
            ge = neg*np.exp(logphi-ln) - pos*np.exp(logphi-lp)
            gb = ge.sum(axis=1)
            ga = (ge*a[:, None]*latent[FACTOR]).sum(axis=1)
            gt = np.zeros((2, len(self.z1)))
            for factor in (0, 1):
                jt = FACTOR == factor
                gt[factor] = (ge[jt]*a[jt, None]).sum(axis=0)
            gmu = gt.sum(axis=1)
            glsd = (gt*(latent-mu[:, None])).sum(axis=1)
            grho = float(gt[1] @ (sd[1]*(sr*sr*self.z1-rho*sr*self.z2)))
            g = np.r_[gb, ga,
                      (gmu[:, None]*x[:self.k_lat]).ravel(),
                      (glsd[:, None]*x[:self.k_lat]).ravel(),
                      grho*np.array([1., x[0]]),
                      (gb[NONANCHORS, None]*x[:self.k_delta]).ravel(),
                      (ga[NONANCHORS, None]*x[:self.k_omega]).ravel()]
            gradient += g
            if return_group_scores:
                group_scores.append(g)
        result = (float(total_nll/grouped.weight_sum), gradient/grouped.weight_sum)
        if not np.isfinite(result[0]) or not np.isfinite(result[1]).all():
            raise FloatingPointError("Nonfinite likelihood or derivative")
        if return_group_scores:
            return (*result, np.asarray(group_scores))
        return result

    def pattern_probabilities(self, t, x, patterns=None):
        """Unconditional response-pattern probabilities given the covariates."""
        patterns = ALL_PATTERNS if patterns is None else np.asarray(patterns, float)
        eta = self._node_values(self.to_components(t), x)[0]
        logp = patterns @ log_ndtr(eta) + (1.-patterns) @ log_ndtr(-eta)
        return np.exp(logsumexp(logp+self.log_quad_weights[None, :], axis=1))

    def moments(self, t, x):
        """Return marginal means and joint-positive probabilities given x."""
        eta = self._node_values(self.to_components(t), x)[0]
        p = ndtr(eta)
        means = p @ self.quad_weights
        pairs = (p*self.quad_weights[None, :]) @ p.T
        np.fill_diagonal(pairs, means)
        return means, pairs

    def common_reference(self, t, x):
        """Item probabilities integrated over N(0,1), excluding latent impact.

        The item-specific affective/somatic coordinate is integrated separately;
        this is not observed prevalence nor a causal or clinical quantity.
        """
        c, x = self.to_components(t), np.asarray(x, float)
        b = c["b"] + c["delta"] @ x
        a = np.exp(c["loga"] + c["omega"] @ x)
        return ndtr(b/np.sqrt(1+a*a))

    def conditional_item_probabilities(self, t, x, latent):
        c, x, latent = self.to_components(t), np.asarray(x, float), np.asarray(latent, float)
        latent = np.atleast_2d(latent)
        if latent.shape[1] != 2:
            raise ValueError("latent must have affective and somatic columns")
        b, a = c["b"] + c["delta"] @ x, np.exp(c["loga"] + c["omega"] @ x)
        return ndtr(b[None, :]+a[None, :]*latent[:, FACTOR])

    def bounds_hits(self, t, tolerance=1e-5):
        out = []
        for name, value, (lo, hi) in zip(self.names, t, self.bounds):
            if value-lo <= tolerance or hi-value <= tolerance:
                out.append({"parameter": name, "value": float(value), "lower": lo, "upper": hi})
        return out
