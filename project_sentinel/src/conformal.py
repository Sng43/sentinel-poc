"""Split-conformal abstention for the binary SA-AKI risk score.

COMPOSER's signature safety trick, made honest: instead of forcing a risk
call on every patient, flag the ones the model is too uncertain about as
INDETERMINATE ("I don't know") rather than guessing.

This is plain split-conformal (LAC — Least-Ambiguous set-valued Classifier).
Nonconformity = 1 - P(true class). A threshold ``qhat`` is calibrated on a
held-out split; at inference, any score inside the band ``[1 - qhat, qhat]``
is ambiguous (the conformal prediction set is {0, 1}) and is abstained on.

Coverage guarantee: the true label lies in the prediction set with
probability >= 1 - alpha, distribution-free and finite-sample. Pure numpy —
no model dependency, so it loads and tests without LightGBM/libomp.

Fit and apply must use the *same* probability transform (e.g. both raw model
output, or both isotonic-calibrated) for the guarantee to hold.
"""

from __future__ import annotations

import numpy as np

INDETERMINATE = "INDETERMINATE"


def fit_conformal_threshold(y_cal, p_pos_cal, alpha: float = 0.1) -> float:
    """Calibrate the conformal threshold ``qhat`` on a held-out split.

    Parameters
    ----------
    y_cal : array of {0, 1}
        True labels on the calibration split.
    p_pos_cal : array
        Predicted P(class = 1) on the same split (same transform used at
        inference time).
    alpha : float
        Target miscoverage rate (0.1 -> 90% coverage).

    Returns
    -------
    float
        ``qhat``. The INDETERMINATE band is ``[1 - qhat, qhat]``; it is
        non-empty only when ``qhat >= 0.5`` (i.e. the model is not confident
        enough to always commit). A near-perfect / overfit model yields a
        small qhat and rarely abstains — which is the honest behaviour.
    """
    y = np.asarray(y_cal).ravel()
    p = np.asarray(p_pos_cal).ravel()
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("y_cal and p_pos_cal must be non-empty and the same length")
    scores = np.where(y == 1, 1.0 - p, p)  # nonconformity = 1 - P(true class)
    n = scores.size
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)  # finite-sample quantile
    return float(np.quantile(scores, level, method="higher"))


def conformal_label(p_pos: float, qhat: float) -> str:
    """Map a positive-class probability to POSITIVE / NEGATIVE / INDETERMINATE."""
    lo, hi = 1.0 - qhat, qhat
    if lo <= p_pos <= hi:
        return INDETERMINATE
    return "POSITIVE" if p_pos > hi else "NEGATIVE"


def indeterminate_rate(p_pos, qhat: float) -> float:
    """Fraction of scores that fall in the abstention band ``[1-qhat, qhat]``."""
    p = np.asarray(p_pos).ravel()
    lo, hi = 1.0 - qhat, qhat
    if lo > hi:
        return 0.0
    return float(((p >= lo) & (p <= hi)).mean())


if __name__ == "__main__":
    # Self-check: conformal coverage holds, and labels behave at the edges.
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    # deliberately imperfect classifier: signal + noise, clipped to [0, 1]
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.2, n), 0.0, 1.0)

    q = fit_conformal_threshold(y, p, alpha=0.1)
    assert 0.0 <= q <= 1.0, q

    # Empirical coverage of the prediction set must be >= ~1 - alpha.
    lo, hi = 1.0 - q, q
    covered = np.where(y == 1, p >= lo, p <= hi).mean()
    assert covered >= 0.88, covered  # target 0.90, small slack for randomness

    assert conformal_label(0.99, 0.6) == "POSITIVE"
    assert conformal_label(0.01, 0.6) == "NEGATIVE"
    assert conformal_label(0.50, 0.6) == INDETERMINATE
    assert 0.0 <= indeterminate_rate(p, q) <= 1.0

    print(f"OK  qhat={q:.3f}  coverage={covered:.3f}  band=[{lo:.3f}, {hi:.3f}]")
