import numpy as np

import torch
from nptdms import TdmsFile

from scipy import optimize, ndimage, signal
from scipy.signal import savgol_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from typing import Tuple, Optional


def lin(x, m, n):
    """Simple linear function y = m*x + n."""
    return m * x + n

def rmed(x, window=11):
    """Running median filter for background correction."""
    pad = window // 2
    xpad = np.pad(x, pad, mode='edge')
    sw = np.lib.stride_tricks.sliding_window_view(xpad, window)
    return np.median(sw, axis=1)

# ------------------------ Core preprocessing class ------------------------

class ContactPointEstimator:
    """
    Estimate the contact point of AFM force–indentation curves
    using a two-step approach:
      1. Coarse detection via baseline deviation.
      2. Refinement by least-squares fitting to a power-law contact model.

    F_model = B + A * max(0, z - zc)**p
      - p = 1.5 for spherical tips (Hertz)
      - p = 2.0 for conical tips (Sneddon)

    Parameters
    ----------
    p : float
        Power-law exponent (1.5 for sphere, 2 for cone).
    smooth_win : int
        Window length for Savitzky–Golay smoothing (odd number).
    smooth_poly : int
        Polynomial order for Savitzky–Golay filter.
    pre_fraction : float
        Fraction of early data treated as baseline.
    sigma_thresh : float
        Threshold in baseline standard deviations for coarse detection.
    fit_span : float
        Fraction of data after coarse contact to include in fitting (0–1).
    """

    def __init__(self, p=1.5, smooth_win=11, smooth_poly=3,
                 pre_fraction=0.4, sigma_thresh=3.0, fit_span=0.4):

        self.p            = p
        self.smooth_win   = smooth_win
        self.smooth_poly  = smooth_poly
        self.pre_fraction = pre_fraction
        self.sigma_thresh = sigma_thresh
        self.fit_span     = fit_span

        # results
        self.result = None

    # ------------------------------------------------------------------ #
    def _smooth(self, y):
        """Apply optional Savitzky–Golay smoothing."""
        if self.smooth_win >= 5 and len(y) > self.smooth_win:
            return savgol_filter(y, self.smooth_win, self.smooth_poly)
        return y.copy()

    # ------------------------------------------------------------------ #
    def _coarse_detection(self, z, force):
        """Coarse contact-point detection via baseline deviation."""
        n_pre = max(3, int(len(z) * self.pre_fraction))
        baseline_mean = np.mean(force[:n_pre])
        baseline_std = np.std(force[:n_pre], ddof=1)
        threshold = baseline_mean + self.sigma_thresh * baseline_std

        mask = force > threshold
        if np.any(mask):
            coarse_idx = np.argmax(mask)
        else:
            # Fallback: maximum curvature
            d2 = np.gradient(np.gradient(force, z), z)
            coarse_idx = np.argmax(np.abs(d2))

        return coarse_idx, baseline_mean

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_model(z, f, p, zc0, A0, B0):
        """Fit the contact model F = B + A * max(0, z - zc)**p."""
        def residuals(params):
            A, zc, B = params
            delta = np.maximum(z - zc, 0)
            return B + A * delta**p - f

        z_min, z_max = np.min(z), np.max(z)
        lb = [0.0, z_min - 0.5*(z_max - z_min), -np.inf]
        ub = [np.inf, z_max + 0.5*(z_max - z_min), np.inf]

        res = least_squares(residuals, [A0, zc0, B0],
                            bounds=(lb, ub), method='trf', max_nfev=5000)

        # Covariance estimate
        j = res.jac
        try:
            JtJ = j.T @ j
            var = np.sum(res.fun**2) / max(1, (len(res.fun) - len(res.x)))
            cov = np.linalg.inv(JtJ) * var
            zc_std = np.sqrt(np.abs(cov[1, 1]))
        except np.linalg.LinAlgError:
            zc_std, cov = np.nan, None

        return res, zc_std

    # ------------------------------------------------------------------ #
    def fit(self, z, force):
        """Estimate contact point and model parameters from a curve."""
        z, f = np.asarray(z), np.asarray(force)
        f_smooth = self._smooth(f)

        coarse_idx, baseline_mean = self._coarse_detection(z, f_smooth)
        zc0 = z[coarse_idx]

        # fit window
        n = len(z)
        start = max(0, coarse_idx - int(0.02 * n))
        end = min(n, start + int(n * self.fit_span))
        z_fit, f_fit = z[start:end], f[start:end]

        # initial guesses
        A0 = (np.max(f_fit) - baseline_mean) / max(1e-12, (np.max(z_fit) - zc0)**self.p)
        B0 = baseline_mean

        res, zc_std = self._fit_model(z_fit, f_fit, self.p, zc0, A0, B0)
        A, zc, B = res.x

        self.result = dict(
            zc=zc, A=A, B=B, zc_std=zc_std,
            residual=res.fun, jac=res.jac, success=res.success
        )
        return self.result

    # ------------------------------------------------------------------ #
    def predict(self, z):
        """Return model force for given z (after fitting)."""
        if self.result is None:
            raise RuntimeError("Run .fit() first.")
        A, zc, B = self.result["A"], self.result["zc"], self.result["B"]
        delta = np.maximum(z - zc, 0)
        return B + A * delta**self.p



# ------------------------ Core preprocessing class ------------------------

class AFMForceCurvePreprocessor:
    """
    AFM Force Curve Preprocessor

    This class loads an AFM force curve (z, d) from a TDMS file,
    performs oscillation fitting, background subtraction, smoothing,
    and prepares the deflection data for contact-point estimation.

    Steps:
    1. Fit cosine to piezo motion z(t)
    2. Fit cosine (or Gaussian) to deflection d(t)
    3. Split data around maxima
    4. Smooth + background-correct deflection
    5. Call contact-point estimator
    6. Return F(\Delta) Force - depth

    Parameters
    ----------
    w_guess : float
        Approximate oscillation period (samples per cycle)
    A_guess : float
        Approximate amplitude of z-oscillation
    p_guess : float
        Phase offset guess
    gauss_sigma : int
        Gaussian smoothing sigma for deflection
    sg_window : int
        Savitzky-Golay window length
    sg_poly : int
        Savitzky-Golay polynomial order
    fit_window_left : float
        Fraction of curve to use on the left side for linear background
    fit_window_right : float
        Fraction of curve to use on the right side for linear background
    contact_params : dict
        Parameters forwarded to ContactPointEstimator
    """

    def __init__(self,
                 d_invols         = 100, ### (nm/V)
                 z_invols         = 100, ### (nm/V)
                 k                = 42,  ##  (N/m)
                 w_guess          = 333,
                 A_guess          = 100,
                 p_guess          = 0,
                 gauss_sigma      = 9,
                 sg_window        = 51,
                 sg_poly          = 3,
                 fit_window_left  = 0.5,
                 fit_window_right = 0.1,
                 contact_params   = None):

        self.d_invols         = d_invols
        self.z_invols         = z_invols
        self.k                = k
        self.w_guess          = w_guess
        self.A_guess          = A_guess
        self.p_guess          = p_guess
        self.gauss_sigma      = gauss_sigma
        self.sg_window        = sg_window
        self.sg_poly          = sg_poly
        self.fit_window_left  = fit_window_left
        self.fit_window_right = fit_window_right
        self.contact_params   = contact_params or dict(p=2, sigma_thresh=1, fit_span=1)

    # ------------------------ Cosine models ------------------------

    @staticmethod
    def z_cosine(x, w=333, A=100, p=0):
        """Cosine model for z(t)."""
        return -A * np.cos(2 * np.pi / w * (x - p))

    @staticmethod
    def d_cosine(x, w=333, A=100, x0=0, c=3):
        """Cosine model for deflection signal."""
        return A * np.cos(2 * np.pi / w * (x - x0)) + c

    @staticmethod
    def d_gaus(x, w=333, A=100, x0=0, c=3):
        """Gaussian model for deflection envelope."""
        return A * np.exp(-((x - x0)**2) / (2 * w**2)) + c

    # ------------------------ Fitting functions ------------------------

    def z_fit(self, t, z):
        """Fit cosine to piezo motion."""
        p0 = [t[np.argmax(z)] * 2, z.max(), 0]
        p, q = optimize.curve_fit(self.z_cosine, t, z, p0=p0, maxfev=1000000)
        return p, q

    def get_z_argmax(self, z):
        """Find z cosine fit and return phase shift."""
        t = np.arange(len(z))
        p = self.z_fit(t, z)[0]
        zf = self.z_cosine(t, *p)
        return np.argmax(zf), p[-1]

    def d_fit(self, t, d, pth=0, x0r=0.1):
        """Fit deflection cosine near its maximum."""
        pk = np.argmax(d[t[-1] // 3:t[-1] * 2 // 3]) + t[-1] // 3
        c = np.median(d[:len(d)//5])
        A = (d.max() - c) / 2
        hwhm = np.argmin(np.abs(d[t[:pk]] - (A + c)))
        w = (t[pk] - t[int(hwhm)]) * 4
        x0 = t[pk]
        c = c + A

        t_range = t[pk - w//6: pk + w//6]
        d_range = d[t[pk] - w//6: t[pk] + w//6]

        p0 = [w, A, x0, c]
        pb = ([-np.inf, -np.inf, t[-1]*(0.5 - x0r), -A*20],
              [np.inf, np.inf, t[-1]*(0.5 + x0r), A*20])

        try:
            p, q = optimize.curve_fit(self.d_cosine, t_range, d_range,
                                      p0=p0, bounds=pb, maxfev=100000)
        except RuntimeError:
            p, q = np.zeros((len(p0))), np.zeros((len(p0), len(p0)))
        return p, q, p0

    def get_d_argmax(self, d):
        """Return deflection cosine-fit parameters."""
        t = np.arange(len(d))
        p = self.d_fit(t, d)[0]
        return p

    # ------------------------ Data preparation ------------------------

    def preprocess_curve(self, FC_path, index=0, chunk_size=1000):
        """
        Preprocess a single AFM force curve segment from a TDMS file.

        Parameters
        ----------
        FC_path : str
            Path to the TDMS file.
        index : int
            Index of the segment to process.
        chunk_size : int
            Number of samples per segment.

        Returns
        -------
        dict : containing processed data and intermediate results
        """
        # --- Load data ---
        tdms = TdmsFile.open(FC_path)
        d = tdms["Forcecurve"].channels()[0]
        z = tdms["Forcecurve"].channels()[1]

        z0 = z[index*chunk_size:(index+1)*chunk_size]*self.z_invols
        d0 = d[index*chunk_size:(index+1)*chunk_size]*self.d_invols*self.k

        # --- Find oscillation centers ---
        zm = self.get_z_argmax(z0)[1] + chunk_size//2
        dm = self.get_d_argmax(d0)[2]

        # --- use shortest phase for left arc
        ms = np.min([zm,dm])
        ml = np.max([zm,dm])

        zl, zr = z0[:int(ms)], z0[int(ml):]
        dl, dr = d0[:int(ms)], d0[int(ml):]

        # --- Smooth deflection ---
        dg = ndimage.gaussian_filter(dl, self.gauss_sigma)
        dg = signal.savgol_filter(dg, self.sg_window, self.sg_poly)

        # --- Background correction ---
        W  = len(dg)//2
        W2 = len(dg)//10
        pl = np.polyfit(zl[   :W], dg[   :W], 1)
        pr = np.polyfit(zl[-W2: ], dg[-W2: ], 1)

        bkg    = lin(zl, *pl)
        dl_bkg = dl - bkg
        dg_bkg = dg - bkg

        # --- Contact point estimation ---
        estimator = ContactPointEstimator(**self.contact_params)
        result    = estimator.fit(zl, dl_bkg)
        cp        = result["zc"]
        thresh    = estimator.predict(zl)

        Delta0    = (zl-cp) - dl_bkg/self.k #
        Delta     = Delta0[Delta0>0]
        F         = dl_bkg[Delta0>0]
        Fs        = dg_bkg[Delta0>0]

        return dict(z_raw      = z0,
                    d_raw      = d0,
                    z_left     = zl,
                    d_left     = dl_bkg,
                    d_smooth   = dg_bkg,
                    background = bkg,
                    cp         = cp,
                    fit_result = result,
                    estimator  = estimator,
                    Force      = F,
                    ForceS     = Fs,
                    Delta      = Delta,
                    lenData    = len(d)//1000
        )


class TingElasticity:
    def __init__(self, R, dt):
        self.R = R
        self.dt = dt

    def estimate_contact(self, z, F):
        # smooth to remove noise
        F_smooth = savgol_filter(F, 21, 3)
        # compute derivative
        dFdz = np.gradient(F_smooth, z)
        # find sharp rise
        idx_contact = np.argmax(dFdz > np.mean(dFdz) + 3*np.std(dFdz))
        self.zc = z[idx_contact]
        self.Fc = F[idx_contact]
        return self.zc, self.Fc

    def compute_indentation(self, z, F):
        return (z - self.zc) - (F - self.Fc)

    def ting_force(self, t, E0, Einf, tau, delta):
        E = Einf + (E0 - Einf) * np.exp(-t / tau)
        kernel = np.gradient(delta**1.5, t)
        F = (4/3)*np.sqrt(self.R)*np.convolve(E, kernel, mode='full')[:len(t)]*self.dt
        return F

    def fit(self, t, z, F):
        delta = self.compute_indentation(z, F)
        p0 = [1000, 500, 0.05]  # initial guesses
        popt, _ = curve_fit(lambda t, E0, Einf, tau:
                            self.ting_force(t, E0, Einf, tau, delta),
                            t, F, p0=p0)
        return popt






class AFMContactPointBatch:
    """
    Vectorized contact-point estimation for large AFM datasets on GPU/MPS.
    """

    def __init__(self, tdms_path, z_invols=100, d_invols=100, k_sp=42,
                 curve_len=1000, device=None, p=1.5):
        self.tdms_path = tdms_path
        self.z_invols = z_invols
        self.d_invols = d_invols
        self.k_sp = k_sp
        self.curve_len = curve_len
        self.p = p
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    def load_data(self):
        tdms = TdmsFile.read(self.tdms_path)
        group = tdms["Forcecurve"]
        d = group.channels()[0].data.astype(np.float32)
        z = group.channels()[1].data.astype(np.float32)

        # Calibration
        d_nm = d * self.d_invols
        z_nm = z * self.z_invols
        F_nN = (d_nm * 1e-9 * self.k_sp) * 1e9  # nN

        n_curves = len(d_nm) // self.curve_len
        F = F_nN[:n_curves*self.curve_len].reshape(n_curves, self.curve_len)
        Z = z_nm[:n_curves*self.curve_len].reshape(n_curves, self.curve_len)

        # move to torch
        self.F = torch.tensor(F, device=self.device)
        self.Z = torch.tensor(Z, device=self.device)
        self.n_curves = n_curves

    def compute_contact_points(self, lr=1e-2, steps=500):
        """
        Vectorized gradient-based optimization of contact points.
        Returns CP in nm for each curve.
        """
        if not hasattr(self, "F"):
            self.load_data()

        # initialize learnable parameters per curve
        zc = torch.full((self.n_curves,), self.Z[:,0].mean(), device=self.device, requires_grad=True)
        A = torch.full((self.n_curves,), (self.F.max() / self.Z.max()**self.p).item(), device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([zc, A], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            # delta >= 0
            delta = torch.clamp(self.Z - zc[:, None], min=0)
            F_model = A[:, None] * delta**self.p
            loss = ((F_model - self.F)**2).mean()
            loss.backward()
            optimizer.step()
            # Optional: constrain zc to valid range
            zc.data = torch.clamp(zc.data, self.Z.min(), self.Z.max())
            A.data = torch.clamp(A.data, 0, None)

            if step % 50 == 0:
                print(f"Step {step}, loss={loss.item():.6f}")

        return zc.detach().cpu().numpy(), A.detach().cpu().numpy()
