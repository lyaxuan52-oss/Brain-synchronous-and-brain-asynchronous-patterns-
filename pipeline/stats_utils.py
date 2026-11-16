# -*- coding: utf-8 -*-
"""Statistical utilities.

Everything here uses the *classic* / standard estimators from scipy and
statsmodels.  No residualisation-based partial correlation, no home-grown
p-value approximations.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --------------------------------------------------------------------------- #
# Partial correlation via direct multiple regression (M1-A / M3-A method)
# --------------------------------------------------------------------------- #
def ols_partial_corr(df, y, x, covars):
    """Fit  y ~ x + covars  by OLS and return the focal predictor statistics.

    Returns dict with unstandardised b, standardised beta, exact partial r
    (from the t statistic), t, p, n.  Partial r is derived directly from the
    regression t-statistic:  r = sign(b) * sqrt(t^2 / (t^2 + df_resid)).
    """
    cols = [y, x] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    if n < len(cols) + 5 or d[x].std() == 0 or d[y].std() == 0:
        return None
    X = sm.add_constant(d[[x] + list(covars)])
    try:
        res = sm.OLS(d[y].values, X.values).fit()
    except Exception:
        return None
    idx = 1  # x is first after const
    b = res.params[idx]
    se = res.bse[idx]
    t = res.tvalues[idx]
    p = res.pvalues[idx]
    dfres = int(res.df_resid)
    r = np.sign(b) * np.sqrt(t * t / (t * t + dfres)) if dfres > 0 else np.nan
    # standardised beta: z-score the continuous columns (keep binary as-is)
    dz = d.copy()
    for c in [y, x] + list(covars):
        s = dz[c].std()
        if s > 0 and dz[c].nunique() > 2:      # only standardise continuous
            dz[c] = (dz[c] - dz[c].mean()) / s
    # standardise x and y always (focal + outcome)
    for c in [y, x]:
        s = d[c].std()
        dz[c] = (d[c] - d[c].mean()) / s if s > 0 else d[c]
    Xz = sm.add_constant(dz[[x] + list(covars)])
    try:
        beta = sm.OLS(dz[y].values, Xz.values).fit().params[idx]
    except Exception:
        beta = np.nan
    return {"b": b, "se": se, "beta": beta, "r": r, "t": t, "p": p, "n": n,
            "df_resid": dfres}


def vif_table(df, predictors):
    """Variance inflation factors for a set of predictors (listwise complete)."""
    d = df[predictors].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(d)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.nan
        rows.append({"predictor": col, "VIF": v})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Multiple comparison correction (BH per family + BY global)
# --------------------------------------------------------------------------- #
def add_fdr(df, pcol="p", group=None, bh_col="p_BH", by_col="p_BY"):
    """Add BH-FDR (within `group`, or overall if None) and global BY-FDR columns."""
    df = df.copy()
    df[bh_col] = np.nan
    df[by_col] = np.nan
    valid = df[pcol].notna()
    # global BY across all valid tests
    if valid.sum() > 0:
        df.loc[valid, by_col] = multipletests(df.loc[valid, pcol], alpha=0.05,
                                              method="fdr_by")[1]
    # BH within family
    if group is None:
        if valid.sum() > 0:
            df.loc[valid, bh_col] = multipletests(df.loc[valid, pcol], alpha=0.05,
                                                  method="fdr_bh")[1]
    else:
        for _, gidx in df[valid].groupby(group).groups.items():
            sub = df.loc[gidx, pcol]
            df.loc[gidx, bh_col] = multipletests(sub, alpha=0.05, method="fdr_bh")[1]
    return df


def sig_marks(row, p="p", bh="p_BH", by="p_BY"):
    m = ""
    if pd.notna(row.get(p)) and row[p] < 0.05:
        m += "*"
    if pd.notna(row.get(p)) and row[p] < 0.01:
        m += "*"
    if pd.notna(row.get(bh)) and row[bh] < 0.05:
        m += "†"
    if pd.notna(row.get(by)) and row[by] < 0.05:
        m += "‡"
    return m


# --------------------------------------------------------------------------- #
# Effect sizes for disease outcomes
# --------------------------------------------------------------------------- #
def cohens_d(case_vals, ctrl_vals):
    a = np.asarray(pd.to_numeric(pd.Series(case_vals), errors="coerce").dropna())
    b = np.asarray(pd.to_numeric(pd.Series(ctrl_vals), errors="coerce").dropna())
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    return (a.mean() - b.mean()) / sp


def logistic_or(df, disease, x, covars=("age", "gender"), standardize_x=True):
    """Logit(disease) ~ x + covars. Returns OR (per 1 SD of x if standardize_x)."""
    cols = [disease, x] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    ncase = int((d[disease] == 1).sum())
    if d[disease].nunique() < 2 or ncase < 10:
        return None
    if standardize_x and d[x].std() > 0:
        d[x] = (d[x] - d[x].mean()) / d[x].std()
    X = sm.add_constant(d[[x] + list(covars)])
    try:
        res = sm.Logit(d[disease].values, X.values).fit(disp=0, maxiter=200)
    except Exception:
        return None
    if not res.mle_retvals.get("converged", False):
        return None
    b = res.params[1]
    se = res.bse[1]
    return {"logOR": b, "OR": np.exp(b), "ci_lo": np.exp(b - 1.96 * se),
            "ci_hi": np.exp(b + 1.96 * se), "p": res.pvalues[1],
            "n": len(d), "n_case": ncase}


# --------------------------------------------------------------------------- #
# Dose-response: grouped ANOVA + linear/nonlinear trend + Cohen's f
# --------------------------------------------------------------------------- #
def dose_response(df, outcome, group_col, covars):
    """OLS ANOVA of outcome across ordinal burden groups + trend tests.

    Fits (1) covariate-only model, (2) + categorical group (ANOVA / partial eta2,
    Cohen's f), (3) + linear group term (linear trend), (4) + quadratic (nonlinear).
    Returns a dict of statistics.  Groups are treated as ordered integers 0..k.
    """
    cols = [outcome, group_col] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if d[group_col].nunique() < 2 or len(d) < 30:
        return None
    y = d[outcome].values
    base = sm.add_constant(d[covars]) if covars else sm.add_constant(np.ones(len(d)))
    m0 = sm.OLS(y, base.values).fit()
    # categorical
    dummies = pd.get_dummies(d[group_col].astype(int), prefix="g", drop_first=True).astype(float)
    Xc = pd.concat([d[list(covars)].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    Xc = sm.add_constant(Xc)
    mc = sm.OLS(y, Xc.values.astype(float)).fit()
    # ANOVA F for group effect (extra sum of squares)
    df_num = dummies.shape[1]
    df_den = int(mc.df_resid)
    ss_diff = (m0.ssr - mc.ssr)
    F = (ss_diff / df_num) / (mc.ssr / df_den) if df_den > 0 and mc.ssr > 0 else np.nan
    p_anova = 1 - stats.f.cdf(F, df_num, df_den) if np.isfinite(F) else np.nan
    partial_eta2 = ss_diff / (ss_diff + mc.ssr) if (ss_diff + mc.ssr) > 0 else np.nan
    cohens_f = np.sqrt(partial_eta2 / (1 - partial_eta2)) if partial_eta2 < 1 else np.nan
    # linear trend
    Xl = sm.add_constant(pd.concat([d[list(covars)].reset_index(drop=True),
                                    d[group_col].reset_index(drop=True)], axis=1))
    ml = sm.OLS(y, Xl.values.astype(float)).fit()
    p_lin = ml.pvalues[-1]
    b_lin = ml.params[-1]
    # nonlinear (quadratic)
    dq = d[group_col].astype(float)
    Xq = sm.add_constant(pd.concat([d[list(covars)].reset_index(drop=True),
                                    dq.reset_index(drop=True),
                                    (dq**2).reset_index(drop=True).rename("g2")], axis=1))
    mq = sm.OLS(y, Xq.values.astype(float)).fit()
    p_quad = mq.pvalues[-1]
    grp_means = d.groupby(group_col)[outcome].agg(["mean", "std", "count"])
    return {"F": F, "p_anova": p_anova, "partial_eta2": partial_eta2,
            "cohens_f": cohens_f, "p_linear": p_lin, "b_linear": b_lin,
            "p_quadratic": p_quad, "n": len(d), "group_means": grp_means}


# --------------------------------------------------------------------------- #
# Bootstrap mediation with BCa CI  (numpy fast paths)
# --------------------------------------------------------------------------- #
def _ols_coef(X, y):
    """OLS coefficients via least squares (X already has intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _logit_coef(X, y, maxiter=100):
    """Newton-Raphson logistic regression; returns coef vector or raises."""
    beta = np.zeros(X.shape[1])
    for _ in range(maxiter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        grad = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def _med_point(Xa, Ycol_m, Xb, Ycol_y, y_binary):
    """Return (a, b, cprime). Xa: [1,x,cov]; Xb: [1,m,x,cov]."""
    a = _ols_coef(Xa, Ycol_m)[1]
    if y_binary:
        bb = _logit_coef(Xb, Ycol_y)
    else:
        bb = _ols_coef(Xb, Ycol_y)
    return a, bb[1], bb[2]


def bootstrap_mediation(df, x, m, y, covars, y_binary=False, n_boot=200, seed=12345,
                        jackknife_cap=1500):
    cols = [x, m, y] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    n = len(d)
    if n < 50:
        return None
    ones = np.ones((n, 1))
    xv = d[x].values[:, None]; mv = d[m].values[:, None]
    cv = d[list(covars)].values if covars else np.empty((n, 0))
    yv = d[y].values
    Xa = np.hstack([ones, xv, cv])
    Xb = np.hstack([ones, mv, xv, cv])
    Xc = np.hstack([ones, xv, cv])
    try:
        a, b, cprime = _med_point(Xa, mv[:, 0], Xb, yv, y_binary)
        c = (_logit_coef(Xc, yv) if y_binary else _ols_coef(Xc, yv))[1]
    except Exception:
        return None
    ab = a * b
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            ai, bi, _ = _med_point(Xa[idx], mv[idx, 0], Xb[idx], yv[idx], y_binary)
            boots[i] = ai * bi
        except Exception:
            boots[i] = np.nan
    boots = boots[np.isfinite(boots)]
    if len(boots) < n_boot * 0.5:
        return None
    lo, hi, method = _bca_ci(boots, ab, Xa, mv[:, 0], Xb, yv, y_binary, n, jackknife_cap)
    total = ab + cprime
    prop = ab / total if total != 0 else np.nan
    return {"a": a, "b": b, "cprime": cprime, "c_total": c, "ab": ab,
            "ci_lo": lo, "ci_hi": hi, "ci_method": method,
            "prop_mediated": prop, "n": n, "n_boot": len(boots),
            "sig": (lo > 0) or (hi < 0)}


def _bca_ci(boots, theta_hat, Xa, ym, Xb, yy, y_binary, n, jackknife_cap, alpha=0.05):
    """BCa CI. For n>jackknife_cap the acceleration uses an evenly-spaced
    subsample of leave-one-out estimates (delete-d jackknife approximation)."""
    try:
        z0 = stats.norm.ppf((boots < theta_hat).mean())
        if not np.isfinite(z0):
            raise ValueError
        idxs = (np.arange(n) if n <= jackknife_cap
                else np.linspace(0, n - 1, jackknife_cap).astype(int))
        jk = []
        allrows = np.ones(n, bool)
        for i in idxs:
            allrows[i] = False
            try:
                ai, bi, _ = _med_point(Xa[allrows], ym[allrows], Xb[allrows],
                                       yy[allrows], y_binary)
                jk.append(ai * bi)
            except Exception:
                pass
            allrows[i] = True
        jk = np.asarray(jk)
        jbar = jk.mean()
        num = np.sum((jbar - jk) ** 3)
        den = 6.0 * (np.sum((jbar - jk) ** 2) ** 1.5)
        acc = num / den if den != 0 else 0.0
        zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
        a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - acc * (z0 + zl)))
        a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - acc * (z0 + zu)))
        lo, hi = np.quantile(boots, a1), np.quantile(boots, a2)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError
        return lo, hi, ("BCa" if n <= jackknife_cap else "BCa(subsampled-acc)")
    except Exception:
        return (np.quantile(boots, alpha / 2), np.quantile(boots, 1 - alpha / 2),
                "percentile")


def ols_interaction(df, y, x, moderator, covars):
    """Fit  y ~ x + moderator + x:moderator + covars  by OLS.
    Returns the interaction term coefficient and p-value."""
    cols = [y, x, moderator] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(d) < len(cols) + 10 or d[x].std() == 0:
        return None
    d["_int"] = d[x] * d[moderator]
    preds = [x, moderator, "_int"] + list(covars)
    X = sm.add_constant(d[preds])
    try:
        res = sm.OLS(d[y].values, X.values).fit()
    except Exception:
        return None
    j = preds.index("_int") + 1
    return {"beta_int": res.params[j], "p_int": res.pvalues[j], "n": len(d)}


def logit_interaction(df, y, x, moderator, covars):
    """Logistic y ~ x + moderator + x:moderator + covars; interaction stats."""
    cols = [y, x, moderator] + list(covars)
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if len(d) < len(cols) + 10 or d[y].nunique() < 2:
        return None
    for c in [x] + list(covars):
        s = d[c].std()
        if s > 0 and d[c].nunique() > 2:
            d[c] = (d[c] - d[c].mean()) / s
    d["_int"] = d[x] * d[moderator]
    preds = [x, moderator, "_int"] + list(covars)
    X = sm.add_constant(d[preds])
    try:
        res = sm.Logit(d[y].values, X.values).fit(disp=0, maxiter=200)
    except Exception:
        return None
    j = preds.index("_int") + 1
    return {"beta_int": res.params[j], "p_int": res.pvalues[j], "n": len(d)}


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return np.nan, np.nan
    rho, p = stats.spearmanr(a[mask], b[mask])
    return rho, p
