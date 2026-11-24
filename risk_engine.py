"""
CapitalLens - core market-risk functions.

VaR/ES models, VaR backtesting (Kupiec, Christoffersen, Basel traffic light)
and Basel 2.5 internal-models capital.
"""
import numpy as np
import pandas as pd
from scipy import stats

Z99 = stats.norm.ppf(0.99)


# ---------------- VaR and ES models (all return positive loss numbers) ----------------
def hs_var(pnl, window=250, alpha=0.99):
    """Historical simulation VaR, forecast for day t using days t-window..t-1."""
    return -pnl.rolling(window).quantile(1 - alpha).shift(1)


def hs_es(pnl, window=250, alpha=0.975):
    """Historical Expected Shortfall: average loss beyond the (1-alpha) quantile."""
    def es(x):
        q = np.quantile(x, 1 - alpha)
        return -x[x <= q].mean()
    return pnl.rolling(window).apply(es, raw=True).shift(1)


def normal_var(pnl, window=250, alpha=0.99):
    """Parametric (variance-covariance) VaR with equal weights."""
    return (stats.norm.ppf(alpha) * pnl.rolling(window).std()).shift(1)


def ewma_vol(pnl, lam=0.94):
    """RiskMetrics exponentially weighted volatility, known at the end of each day."""
    return np.sqrt((pnl ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def ewma_var(pnl, lam=0.94, alpha=0.99):
    return (stats.norm.ppf(alpha) * ewma_vol(pnl, lam)).shift(1)


def filtered_hs_var(pnl, window=250, lam=0.94, alpha=0.99):
    """Filtered historical simulation: rescale past P&L by today's EWMA volatility."""
    vol = ewma_vol(pnl, lam).shift(1)          # vol forecast for each day
    z = pnl / vol                              # standardised P&L
    zq = z.rolling(window).quantile(1 - alpha).shift(1)
    return -zq * ewma_vol(pnl, lam).shift(1)


# ---------------- backtesting ----------------
def exceptions(pnl, var):
    """1 when the actual loss is bigger than the VaR forecast."""
    ok = var.notna()
    return ((-pnl > var) & ok).astype(int)[ok]


def kupiec_pof(n_exc, n_obs, p=0.01):
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value)."""
    x, n = n_exc, n_obs
    phat = x / n
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    elif x == n:
        lr = -2 * (n * np.log(p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(exc):
    """Christoffersen independence test: are exceptions clustered? Returns (LR, p-value)."""
    e = np.asarray(exc)
    prev, cur = e[:-1], e[1:]
    n00 = np.sum((prev == 0) & (cur == 0)); n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0)); n11 = np.sum((prev == 1) & (cur == 1))
    pi0 = n01 / max(n00 + n01, 1); pi1 = n11 / max(n10 + n11, 1); pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b):  # log-likelihood with safe logs
        return (a * np.log(1 - p) if a else 0) + (b * np.log(p) if b else 0)
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return lr, 1 - stats.chi2.cdf(lr, 1)


PLUS_FACTOR = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def traffic_light(n_exc_250):
    """Basel backtesting zones for 250 days at 99%: green 0-4, yellow 5-9, red 10+."""
    if n_exc_250 <= 4:
        return "Green", 0.0
    if n_exc_250 <= 9:
        return "Yellow", PLUS_FACTOR[n_exc_250]
    return "Red", 1.0


# ---------------- Basel 2.5 internal-models capital ----------------
def basel_capital(var1_series, svar1, n_exc_250, horizon=10):
    """
    Market-risk capital = max(VaR_t, m * avg60(VaR)) + max(sVaR_t, m * avg60(sVaR)),
    with 10-day VaR = 1-day VaR x sqrt(10) and m = 3 + plus factor from backtesting.
    sVaR uses a fixed stress window, so its 60-day average equals today's value.
    """
    zone, plus = traffic_light(n_exc_250)
    m = 3 + plus
    var10 = var1_series * np.sqrt(horizon)
    var_t, var_avg60 = var10.iloc[-1], var10.iloc[-60:].mean()
    svar10 = svar1 * np.sqrt(horizon)
    var_charge = max(var_t, m * var_avg60)
    svar_charge = max(svar10, m * svar10)
    return {"zone": zone, "plus_factor": plus, "multiplier": m,
            "VaR_10d_today": var_t, "VaR_10d_avg60": var_avg60, "VaR_charge": var_charge,
            "sVaR_10d": svar10, "sVaR_charge": svar_charge, "total_capital": var_charge + svar_charge}


def component_var(returns_window, positions, alpha=0.99):
    """Euler (parametric) component VaR: how much each position adds to portfolio VaR. Sums to total."""
    cov = returns_window.cov().values
    w = positions.values
    sigma_p = np.sqrt(w @ cov @ w)
    z = stats.norm.ppf(alpha)
    comp = w * (cov @ w) / sigma_p * z
    return pd.Series(comp, index=positions.index), z * sigma_p
