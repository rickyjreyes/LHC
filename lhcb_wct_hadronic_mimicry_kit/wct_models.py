import numpy as np

# -----------------------------
# Basic model definitions
# -----------------------------

def wct_log_periodic(q2, A, k, phi, mu, sigma_w):
    """
    5-parameter WCT log-periodic ansatz:
        A exp[-(ln q2 - mu)^2/(2 sigma_w^2)] cos(k ln q2 + phi)

    q2 must be positive.
    """
    q2 = np.asarray(q2, dtype=float)
    ell = np.log(q2)
    sigma_w = max(float(sigma_w), 1e-9)
    env = np.exp(-0.5 * ((ell - mu) / sigma_w) ** 2)
    return A * env * np.cos(k * ell + phi)


def constant_shift(q2, c):
    q2 = np.asarray(q2, dtype=float)
    return np.full_like(q2, float(c), dtype=float)


def breit_wigner(q2, M, Gamma, amp=1.0, phase=0.0):
    """
    Complex Breit-Wigner tail in linear q^2:
        amp exp(i phase) / (q2 - M^2 + i M Gamma)

    Returns complex amplitude.
    """
    q2 = np.asarray(q2, dtype=float)
    return amp * np.exp(1j * phase) / (q2 - M**2 + 1j * M * Gamma)


def charm_tail(q2, a_jpsi=1.0, phi_jpsi=0.0, a_psi2s=0.35, phi_psi2s=1.0, scale=1.0, offset=0.0):
    """
    Minimal charm-tail toy model from J/psi and psi(2S) Breit-Wigner tails.

    Masses and widths in GeV. q2 in GeV^2.

    Note:
    This is a stress-test model, not a complete SM amplitude.
    It is intentionally realistic enough to create nonlinear tails but simple enough
    for null-injection testing.
    """
    jpsi = breit_wigner(q2, M=3.096900, Gamma=0.0000929, amp=a_jpsi, phase=phi_jpsi)
    psi2s = breit_wigner(q2, M=3.686097, Gamma=0.000294, amp=a_psi2s, phase=phi_psi2s)
    amp = jpsi + psi2s
    # Normalize shape to avoid huge scale from narrow poles.
    y = np.real(amp)
    y = y - np.mean(y)
    s = np.std(y)
    if s > 0:
        y = y / s
    return offset + scale * y


def smooth_background(q2, c0=0.0, c1=0.0, c2=0.0, q2_ref=6.0):
    """
    Smooth polynomial background in centered q2.
    """
    x = np.asarray(q2, dtype=float) - q2_ref
    return c0 + c1 * x + c2 * x**2


def chi2(y, yhat, sigma):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(sigma <= 0, np.nanmedian(sigma[sigma > 0]), sigma)
    return float(np.sum(((y - yhat) / sigma) ** 2))


def aic(chi2_value, n_params):
    return float(chi2_value + 2 * n_params)


def bic(chi2_value, n_params, n_obs):
    return float(chi2_value + n_params * np.log(max(n_obs, 1)))
