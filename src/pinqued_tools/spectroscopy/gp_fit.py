import numpy as np

from pinqued_tools.spectroscopy.spectrum import SpectralData
from pinqued_tools.spectroscopy.field_reference import FieldReference
from pinqued_tools.spectroscopy.signal_simulator import GPPoissonSignalSimulator1D

from typing import Callable
from lmfit import minimize, Parameters
from numpy.typing import NDArray
from scipy.linalg import cholesky
from scipy.interpolate import BSpline
from scipy.sparse import diags
from scipy.integrate import cumulative_trapezoid
from numba import njit, prange

import matplotlib.pyplot as plt






class GPPoissonModel1D():
    def __init__(self, 
                 data: SpectralData,
                 field_ref: FieldReference,
                 signal_sim: GPPoissonSignalSimulator1D,
                 E_vec_init: NDArray,
                 l=1.0, sigma_f=50.0,
                 zero_bnd_efield: bool = True
                 ):
        """
        z: spatial coordinates (mm)
        freq: frequency axis (MHz)
        spectra: 2D array [z_bins, freq_bins]
        l: correlation lengthscale (related to Debye length)
        sigma_f: prior variance for potential
        """
        self.field_ref = field_ref
        self.signal_sim = signal_sim
    
        # Using x as the spatial coordinate per Axes1D definition
        self.x = data.axes.x if hasattr(data.axes, 'x') else data.axes.y
        # Integrate the initial macroscopic Electric Field guess to get the potential phi
        # NOTE: E_vec_init is in V/cm, x is in mm. We divide by 10 to get V/mm for integration.
        self.phi_vec_init = -cumulative_trapezoid(E_vec_init / 10.0, self.x, initial=0.0)
        self.f = data.axes.f
        spectra = data.signal
        self.data_max = np.max(spectra) # Use a single global max for normalization
        self.data = spectra / self.data_max
        self.px_size = np.abs(self.x[1] - self.x[0]) # Pixel size
        
        # Empirically estimate the standard deviation of the noise in the normalized data
        self.noise_sigma = np.std(np.diff(self.data, axis=-1)) / np.sqrt(2.0)
        if self.noise_sigma < 1e-8: self.noise_sigma = 1.0
        # Estimate the standard deviation of the noise in the normalized data
        if getattr(data, 'signal_err', None) is not None:
            self.noise_sigma = np.mean(data.signal_err / self.data_max)
        else:
            self.noise_sigma = np.std(np.diff(self.data, axis=-1)) / np.sqrt(2.0)
            if self.noise_sigma < 1e-8: self.noise_sigma = 1.0
        
        # 1. Bake in Poisson Logic: GP Prior on Potential phi(x)
        # We use a Matern 5/2 kernel because it is twice differentiable.
        self.K_phi = self._matern_kernel(self.x, self.x, l, sigma_f)
        # Regularize for inversion
        self.K_inv = np.linalg.inv(self.K_phi + 1e-6 * np.eye(len(self.x)))
        
        # Cholesky decomposition of K_inv to formulate the GP prior as sum of squares
        # (L_inv @ phi).T @ (L_inv @ phi) = phi.T @ K_inv @ phi
        self.L_inv = cholesky(self.K_inv, lower=False)
        self.zero_bnd_efield = zero_bnd_efield

    def _matern_kernel(self, x1, x2, l, sigma_f):
        d = np.abs(x1[:, None] - x2[None, :])
        arg = np.sqrt(5) * d / l
        return sigma_f**2 * (1 + arg + (arg**2)/3) * np.exp(-arg)

    def setup_params(self, base_params: Parameters) -> Parameters:
        """
        Adds the phi_vec parameters to an existing lmfit Parameters object.
        """
        params = base_params.copy()
        num_phi = len(self.phi_vec_init)
        
        # Prevent vanishing gradient if initialized with exact zeros
        if np.allclose(self.phi_vec_init, 0.0):
            # Assuming bulk plasma (zero potential) is at x[0]
            self.phi_vec_init = -5.0 * np.abs(self.x - self.x[0])
            
        for i, phi_val in enumerate(self.phi_vec_init):
            params.add(f'phi_{i}', value=phi_val)
            
        # Enforce phi -> 0 and E -> 0 at the boundary deep in the plasma.
        if num_phi > 1:
            params[f'phi_{num_phi - 1}'].set(value=0, vary=False)
            if self.zero_bnd_efield:
                params[f'phi_{num_phi - 2}'].set(value=0, vary=False)
        elif num_phi == 1:
            params[f'phi_{num_phi - 1}'].set(value=0, vary=False)
        
        # Ensure efield and grad_vec exist in params since signal_sim expects them
        if 'efield' not in params:
            params.add('efield', value=0.0, vary=False)
        if 'grad_vec' not in params:
            params.add('grad_vec', value=0.0, vary=False)
            
        return params

    def forward_physics(self, params):
        """Maps Potential -> Field -> Smeared Spectra."""
        # Reconstruct phi_vec array from the lmfit parameters
        phi_vec = np.array([params[f'phi_{i}'].value for i in range(len(self.x))])
        
        # E = -d_phi/dx. Spatial axis `x` is in mm, so gradient is V/mm.
        # Multiply by 10.0 to convert to V/cm (required by Stark reference).
        E_vec = -np.gradient(phi_vec, self.x) * 10.0
        # grad = dE/dx (Broadening driver) in (V/cm) / mm
        grad_vec = np.gradient(E_vec, self.x)
        
        # Fully vectorized 2D Spectrum Evaluation
        S_pred = self.signal_sim.holtsmark_spectrum(self.f, params, 
                                                    efield=E_vec, grad_vec=grad_vec)
        
        # Ensure predicted spectrum orientation matches experimental data
        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec

    def calc_uncertainty_bands(self, fit_result, n_samples=200, random_state=None):
        """
        Estimates the uncertainty bands (1-sigma standard deviation) for the physical 
        vectors by Monte Carlo sampling the parameter covariance matrix.
        """
        if getattr(fit_result, 'covar', None) is None:
            raise ValueError("Fit result does not contain a covariance matrix. "
                             "Uncertainties cannot be estimated.")
                             
        rng = np.random.default_rng(random_state)
        var_names = fit_result.var_names
        best_vals = np.array([fit_result.params[name].value for name in var_names])
        
        covar = np.array(fit_result.covar)
        # 1. Force perfectly symmetric
        covar = (covar + covar.T) / 2.0
        # 2. Clip negative eigenvalues to zero to strictly preserve parameter smoothness correlations
        eigvals, eigvecs = np.linalg.eigh(covar)
        if np.any(eigvals < 0):
            eigvals[eigvals < 0] = 0.0
            covar = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        # Sample parameter space using robust Singular Value Decomposition
        samples = rng.multivariate_normal(best_vals, covar, size=n_samples, method='svd')
        
        E_samples = np.zeros((n_samples, len(self.x)))
        p_copy = fit_result.params.copy()
        
        for i, sample in enumerate(samples):
            for name, val in zip(var_names, sample):
                param = p_copy[name]
                if val < param.min: val = param.min
                if val > param.max: val = param.max
                param.value = val
                
            res = self.forward_physics(p_copy)
            E_samples[i] = res[1] # E_vec is consistently the 2nd return element
            
        return np.std(E_samples, axis=0)

    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        S_pred, E_vec, _, phi_vec = self.forward_physics(params)
        
        # Normalize data identically to internal logic
        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            # Weight residuals strictly by the empirical noise standard deviation
            data_res = (difference / self.noise_sigma).flatten()
            scale_factor = self.noise_sigma
            data_res = (difference / scale_factor).flatten()
        else:
            data_err_norm = data_err / self.data_max
            scale_factor = np.mean(data_err_norm)
            data_res = (difference / data_err_norm).flatten()
            
        # Include GP smoothness penalty as "prior residuals", scaled to preserve model shape
        prior_res = (self.L_inv @ phi_vec) / scale_factor

        # Soft penalty to bound E-field without zeroing LM gradients
        max_E = np.max(self.field_ref.efield)
        overshoot = np.clip(E_vec - max_E, 0, None)
        undershoot = np.clip(-E_vec, 0, None)
        # Smooth quadratic penalty to prevent infinite Jacobian walls that break the optimizer
        bounds_penalty = (1e4 * (overshoot**2 + undershoot**2)) / scale_factor
        
        return np.concatenate([data_res, prior_res, bounds_penalty])


class BSplinePoissonModel1D():
    def __init__(self, 
                 data: SpectralData, #Input Stark map with frequency and 1 spatial axis
                 field_ref: FieldReference, # FieldReference instance to calculate Stark shifts
                 signal_sim: GPPoissonSignalSimulator1D, # Spectral signal simulator
                 E_vec_init: NDArray, # Initial guess of E-field distribution
                 E0_vec_init: NDArray|None = None, # Initial guess for Holtsmark field distribution
                 n_splines: int = 25, # Dimension of spline basis
                 spline_degree: int = 3,
                 smooth_param: float = 1e4, # Smoothing parameter for E-field spline
                 smooth_param_E0: float = 1e4, # Smoothing parameter for Holtsmark field spline
                 zero_bnd_efield: bool = True # If True, forces E-field to exactly 0 at the boundary
                 ):
        """
        Models the potential phi(x) using penalized B-splines (P-splines).

        z: spatial coordinates (mm)
        freq: frequency axis (MHz)
        spectra: 2D array [z_bins, freq_bins]
        n_splines: number of B-spline basis functions.
        spline_degree: degree of the B-spline (e.g., 3 for cubic).
        smooth_param: smoothing penalty weight for the potential phi. 
                      (Often needs to be 1e3 - 1e6 to overpower data noise).
        smooth_param_E0: smoothing penalty weight for the microfield E0.
                         (Often needs to be 1e3 - 1e6).
        """
        self.field_ref = field_ref
        self.signal_sim = signal_sim
    
        # Using x as the spatial coordinate per Axes1D definition
        self.x = data.axes.x if hasattr(data.axes, 'x') else data.axes.y
        if len(E_vec_init) != len(self.x):
            raise ValueError(f"Length of E_vec_init ({len(E_vec_init)}) must match spatial axis length ({len(self.x)}).")
        # Integrate the initial macroscopic Electric Field guess to get the potential phi
        # Note: E_vec_init is in V/cm, x is in mm. We divide by 10 to get V/mm for integration.
        self.phi_vec_init = -cumulative_trapezoid(E_vec_init / 10.0, self.x, initial=0.0)
        self.f = data.axes.f
        spectra = data.signal
        
        if E0_vec_init is None:
            self.E0_vec_init = np.full_like(self.x, 1.0)
        else:
            if len(E0_vec_init) != len(self.x):
                raise ValueError(f"Length of E0_vec_init ({len(E0_vec_init)}) must match spatial axis length ({len(self.x)}).")
            self.E0_vec_init = E0_vec_init
            
        self.n_splines = n_splines
        self.smooth_param = smooth_param
        self.smooth_param_E0 = smooth_param_E0
        self.zero_bnd_efield = zero_bnd_efield
        self.data_max = np.max(spectra) # Use a single global max for normalization
        self.data = spectra / self.data_max
        self.px_size = np.abs(self.x[1] - self.x[0]) # Pixel size
        
        # Empirically estimate the standard deviation of the noise in the normalized data
        self.noise_sigma = np.std(np.diff(self.data, axis=-1)) / np.sqrt(2.0)
        if self.noise_sigma < 1e-8: self.noise_sigma = 1.0
        # Estimate the standard deviation of the noise in the normalized data
        if getattr(data, 'signal_err', None) is not None:
            self.noise_sigma = np.mean(data.signal_err / self.data_max)
        else:
            self.noise_sigma = np.std(np.diff(self.data, axis=-1)) / np.sqrt(2.0)
            if self.noise_sigma < 1e-8: self.noise_sigma = 1.0
        
        # 1. Bake in P-Spline Logic: Penalized B-Spline Prior on Potential phi(x)
        self.k = spline_degree
        if n_splines <= self.k:
            raise ValueError("Number of splines must be greater than spline degree.")
        
        # Define knots for the B-spline basis. Use clamped knots for well-behaved boundaries.
        # For n_splines basis functions of degree k, we need n_splines - k - 1 interior knots.
        n_internal_knots = n_splines - self.k - 1
        internal_knots = np.linspace(self.x[0], self.x[-1], n_internal_knots + 2)[1:-1]
        self.knots = np.concatenate(([self.x[0]] * (self.k + 1), internal_knots, [self.x[-1]] * (self.k + 1)))

        # Precompute the B-spline basis matrices for instant evaluation via dot product
        self.B = np.zeros((len(self.x), n_splines))
        self.B_d1 = np.zeros((len(self.x), n_splines))
        self.B_d2 = np.zeros((len(self.x), n_splines))
        
        for i in range(n_splines):
            c = np.zeros(n_splines)
            c[i] = 1
            # BSpline is defined by knots, coefficients, and degree.
            spl = BSpline(self.knots, c, self.k, extrapolate=False)
            self.B[:, i] = spl(self.x)
            self.B_d1[:, i] = spl(self.x, nu=1)
            self.B_d2[:, i] = spl(self.x, nu=2)
        
        # Get pseudo-inverse to map from potential phi to spline coefficients c
        self.B_plus = np.linalg.pinv(self.B)
        self.c_init = self.B_plus @ self.phi_vec_init
        self.c_E0_init = self.B_plus @ self.E0_vec_init
        
        # Construct difference matrix for penalty on coefficients.
        # A 3rd-order difference penalizes the 2nd derivative of the E-field,
        # allowing the optimizer to form physically realistic linear E-fields (sheaths)
        # with ZERO penalty, eliminating the artificial "curved up" parabola effect. 
        if self.n_splines >= 4:
            self.D = diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(self.n_splines - 3, self.n_splines)).toarray()
        elif self.n_splines >= 3:
            self.D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(self.n_splines - 2, self.n_splines)).toarray()
        else:
            self.D = diags([-1.0, 1.0], [0, 1], shape=(self.n_splines - 1, self.n_splines)).toarray()



    def setup_params(self, base_params: Parameters) -> Parameters:
        """
        Adds the spline coefficient parameters to an existing lmfit Parameters object.
        """
        params = base_params.copy()
        
        # Model background and amplitude using B-splines to reduce parameter count
        init_amp = params['amp'].value if 'amp' in params else 100.0
        for i in range(self.n_splines):
            if f'c_b0_{i}' not in params:
                params.add(f'c_b0_{i}', value=1e-4)
            if f'c_b1_{i}' not in params:
                params.add(f'c_b1_{i}', value=1e-4)
            if f'c_amp_{i}' not in params:
                params.add(f'c_amp_{i}', value=init_amp, min=0.0)

        c_init = self.c_init.copy()
        c_E0_init = self.c_E0_init.copy()
        # Prevent vanishing gradient if initialized with exact zeros
        if np.allclose(c_init, 0.0):
            # Assuming bulk plasma (zero potential) is at x[0]
            phi_slope = -5.0 * np.abs(self.x - self.x[0])
            c_init = self.B_plus @ phi_slope

        # NOTE: Monotonic decreasng in phi(x) is not removed! 
        # This allows for more flexible fitting while still guaranteeing physical plausibility.
        # Enforce monotonically decreasing potential phi(x) by constraining B-spline 
        # coefficients. This strictly guarantees Electric Field >= 0 everywhere.
        params.add(f'c_{self.n_splines - 1}', value=0.0, vary=False)
        for i in range(self.n_splines - 2, -1, -1):
            delta_init = c_init[i] - c_init[i+1]
            params.add(f'delta_c_{i}', value=delta_init)
            params.add(f'delta_c_{i}', value=max(0.0, delta_init), min=0.0)
            params.add(f'c_{i}', expr=f'c_{i+1} + delta_c_{i}')
            
        # Constrain E0 to a physical ceiling
        for i in range(self.n_splines):
            params.add(f'c_E0_{i}', value=c_E0_init[i], min=1e-20, max=2.0, vary=True)
            
        # Enforce E -> 0 at the boundary deep in the plasma.
        # if self.n_splines > 1 and self.zero_bnd_efield:
        #     params[f'delta_c_{self.n_splines - 2}'].set(value=0.0, vary=True)
        
        if 'fshift' not in params:
            params.add('fshift', value=0.0)
        
        return params

    def forward_physics(self, params, coeffs=None):
        """Maps Potential -> Field -> Smeared Spectra."""
        if coeffs is None:
            # Reconstruct B-spline coefficients c from the lmfit parameters
            c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
            c_E0 = np.array([params[f'c_E0_{i}'].value for i in range(self.n_splines)])
            c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
            c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
            c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        else:
            c, c_E0, c_b0, c_b1, c_amp = coeffs

        # Fast evaluation using precomputed basis matrices (BLAS matrix multiplication)
        phi_vec = self.B @ c
        E0_vec = self.B @ c_E0
        E_vec = -(self.B_d1 @ c) * 10.0
        grad_vec = -(self.B_d2 @ c) * 10.0
        b0_vec = self.B @ c_b0
        b1_vec = self.B @ c_b1
        amp_vec = self.B @ c_amp
        
        fshift = params['fshift'].value if 'fshift' in params else 0.0
        f_shifted = self.f - fshift

        # 2D Spectrum Evaluation
        S_pred = self.signal_sim.holtsmark_spectrum(
            f_shifted, params, efield=E_vec, grad_vec=grad_vec, E0=E0_vec, amp=1.0)
        S_pred *= amp_vec[:, np.newaxis]
        S_pred += (b0_vec[:, np.newaxis] * f_shifted[np.newaxis, :] + b1_vec[:, np.newaxis])

        # Ensure predicted spectrum orientation matches experimental data
        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec, E0_vec

    def calc_uncertainty_bands(self, fit_result, n_samples=200, random_state=None):
        """
        Estimates the uncertainty bands (1-sigma standard deviation) for the physical 
        vectors by Monte Carlo sampling the parameter covariance matrix.
        """
        if getattr(fit_result, 'covar', None) is None:
            raise ValueError("Fit result does not contain a covariance matrix. "
                             "Uncertainties cannot be estimated.")
                             
        rng = np.random.default_rng(random_state)
        var_names = fit_result.var_names
        best_vals = np.array([fit_result.params[name].value for name in var_names])
        
        covar = np.array(fit_result.covar)
        # 1. Force perfectly symmetric
        covar = (covar + covar.T) / 2.0
        # 2. Clip negative eigenvalues to zero to strictly preserve parameter smoothness correlations
        eigvals, eigvecs = np.linalg.eigh(covar)
        if np.any(eigvals < 0):
            eigvals[eigvals < 0] = 0.0
            covar = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        # Sample parameter space using robust Singular Value Decomposition
        samples = rng.multivariate_normal(best_vals, covar, size=n_samples, method='svd')
        
        E_samples = np.zeros((n_samples, len(self.x)))
        p_copy = fit_result.params.copy()
        
        for i, sample in enumerate(samples):
            for name, val in zip(var_names, sample):
                param = p_copy[name]
                # Safely enforce existing parameter bounds (e.g., E0 > 0)
                if val < param.min: val = param.min
                if val > param.max: val = param.max
                param.value = val
                
            res = self.forward_physics(p_copy)
            E_samples[i] = res[1] # E_vec is consistently the 2nd return element
            
        return np.std(E_samples, axis=0)

    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        
        # Extract parameters once to avoid duplicate loops
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        c_E0 = np.array([params[f'c_E0_{i}'].value for i in range(self.n_splines)])
        c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
        c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
        c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        
        coeffs = (c, c_E0, c_b0, c_b1, c_amp)
        S_pred, E_vec, _, phi_vec, E0_vec = self.forward_physics(params, coeffs=coeffs)

        if data.shape != S_pred.shape:
            raise ValueError(f"Data shape mismatch! The fitter provided data of shape {data.shape}, "
                             f"but the model evaluated a grid of shape {S_pred.shape}. "
                             "Ensure the DataFitter is initialized with the exact same SpectralData "
                             "object that was used to initialize the model.")

        # Normalize data identically to internal logic
        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            # Weight residuals strictly by the empirical noise standard deviation
            data_res = (difference / self.noise_sigma).flatten()
            scale_factor = self.noise_sigma
            data_res = (difference / scale_factor).flatten()
        else:
            data_err_norm = data_err / self.data_max
            scale_factor = np.mean(data_err_norm)
            data_res = (difference / data_err_norm).flatten()
            
        # Include P-spline smoothness penalty as "prior residuals", scaled to preserve model shape
        prior_res = np.sqrt(self.smooth_param) * (self.D @ c) / scale_factor
        prior_res_E0 = np.sqrt(self.smooth_param_E0) * (self.D @ c_E0) / scale_factor
        prior_res_b0 = np.sqrt(self.smooth_param * 0.1) * (self.D @ c_b0) / scale_factor
        prior_res_b1 = np.sqrt(self.smooth_param * 0.1) * (self.D @ c_b1) / scale_factor
        prior_res_amp = np.sqrt(self.smooth_param * 0.1) * (self.D @ c_amp) / scale_factor
        
        return np.concatenate([data_res, prior_res, prior_res_E0, prior_res_b0, prior_res_b1, prior_res_amp])


# ------------------- NUMBA ACCELERATED MODEL -------------------

@njit(fastmath=True, cache=True)
def _bspline_eval_vectors_numba(c, c_E0, c_b0, c_b1, c_amp, B, B_d1, B_d2):

    # Fast BLAS matrix multiplications
    phi_vec = B @ c
    E0_vec = B @ c_E0
    E_vec = -(B_d1 @ c) * 10.0
    grad_vec = -(B_d2 @ c) * 10.0
    b0_vec = B @ c_b0
    b1_vec = B @ c_b1
    amp_vec = B @ c_amp
    return phi_vec, E0_vec, E_vec, grad_vec, b0_vec, b1_vec, amp_vec

@njit(parallel=True, fastmath=True, cache=True)
def _apply_bg_numba(S_pred, amp_vec, b0_vec, b1_vec, f_shifted):
    # Bypasses intermediate large memory allocations typical of NumPy broadcasting
    out = np.empty_like(S_pred)
    for i in prange(S_pred.shape[0]):
        for j in range(S_pred.shape[1]):
            out[i, j] = S_pred[i, j] * amp_vec[i] + b0_vec[i] * f_shifted[j] + b1_vec[i]
    return out

@njit(parallel=True, fastmath=True, cache=True)
def _apply_global_bg_numba(S_pred, amp, b0, b1, f_shifted):
    """
    Applies purely scalar background and amplitude parameters to the 2D spectrum.
    """
    out = np.empty_like(S_pred)
    for i in prange(S_pred.shape[0]):
        for j in range(S_pred.shape[1]):
            out[i, j] = S_pred[i, j] * amp + b0 * f_shifted[j] + b1
    return out

@njit(fastmath=True, cache=True)
def _calc_prior_res_numba(c, c_E0, c_b0, c_b1, c_amp, D, smooth_param, smooth_param_E0):
    # Calculate penalties efficiently
    prior_res = np.sqrt(smooth_param) * (D @ c)
    prior_res_E0 = np.sqrt(smooth_param_E0) * (D @ c_E0)
    prior_res_b0 = np.sqrt(smooth_param * 0.1) * (D @ c_b0)
    prior_res_b1 = np.sqrt(smooth_param * 0.1) * (D @ c_b1)
    prior_res_amp = np.sqrt(smooth_param * 0.1) * (D @ c_amp)
    return prior_res, prior_res_E0, prior_res_b0, prior_res_b1, prior_res_amp


class BSplinePoissonModel1D_numba(BSplinePoissonModel1D):
    """
    Numba-accelerated drop-in replacement for BSplinePoissonModel1D.
    """
    def forward_physics(self, params, coeffs=None):
        if coeffs is None:
            c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
            c_E0 = np.array([params[f'c_E0_{i}'].value for i in range(self.n_splines)])
            c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
            c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
            c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        else:
            c, c_E0, c_b0, c_b1, c_amp = coeffs

        # Evaluate vectors using optimized Numba
        phi_vec, E0_vec, E_vec, grad_vec, b0_vec, b1_vec, amp_vec = _bspline_eval_vectors_numba(
            c, c_E0, c_b0, c_b1, c_amp, self.B, self.B_d1, self.B_d2
        )
        
        fshift = params['fshift'].value if 'fshift' in params else 0.0
        f_shifted = self.f - fshift

        S_pred = self.signal_sim.holtsmark_spectrum(
            f_shifted, params, efield=E_vec, grad_vec=grad_vec, E0=E0_vec, amp=1.0)
        
        # Parallelly apply background offsets and spatial amplitudes 
        S_pred = _apply_bg_numba(S_pred, amp_vec, b0_vec, b1_vec, f_shifted)

        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec, E0_vec

    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        c_E0 = np.array([params[f'c_E0_{i}'].value for i in range(self.n_splines)])
        c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
        c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
        c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        
        coeffs = (c, c_E0, c_b0, c_b1, c_amp)
        S_pred, E_vec, _, phi_vec, E0_vec = self.forward_physics(params, coeffs=coeffs)

        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            data_res = difference.flatten()
            scale_factor = self.noise_sigma
            data_res = (difference / scale_factor).flatten()
        else:
            data_err_norm = data_err / self.data_max
            scale_factor = np.mean(data_err_norm)
            data_res = (difference / data_err_norm).flatten()
            
        # Extracted spline smoothing penalties calculated in Numba
        prior_res, prior_res_E0, prior_res_b0, prior_res_b1, prior_res_amp = _calc_prior_res_numba(
            c, c_E0, c_b0, c_b1, c_amp, self.D, self.smooth_param, self.smooth_param_E0
        )
        
        return np.concatenate([data_res, 
                               prior_res / scale_factor, 
                               prior_res_E0 / scale_factor, 
                               prior_res_b0 / scale_factor, 
                               prior_res_b1 / scale_factor, 
                               prior_res_amp / scale_factor])



#---------------------------------------------------------------------------------------------------
# GLOBAL E0 VERSION WITH NUMBA ACCELERATION (NO SPATIAL VARIATION IN HOLTSMARK FIELD)
@njit(fastmath=True, cache=True)
def _bspline_eval_vectors_numba_global_E0(c, c_b0, c_b1, c_amp, B, B_d1, B_d2):
    # Fast BLAS matrix multiplications
    phi_vec = B @ c
    E_vec = -(B_d1 @ c) * 10.0
    grad_vec = -(B_d2 @ c) * 10.0
    b0_vec = B @ c_b0
    b1_vec = B @ c_b1
    amp_vec = B @ c_amp
    return phi_vec, E_vec, grad_vec, b0_vec, b1_vec, amp_vec


@njit(fastmath=True, cache=True)
def _calc_prior_res_numba_global_E0(c, c_b0, c_b1, c_amp, D, smooth_param):
    # Calculate penalties efficiently
    prior_res = np.sqrt(smooth_param) * (D @ c)
    prior_res_b0 = np.sqrt(smooth_param * 0.1) * (D @ c_b0)
    prior_res_b1 = np.sqrt(smooth_param * 0.1) * (D @ c_b1)
    prior_res_amp = np.sqrt(smooth_param * 0.1) * (D @ c_amp)
    return prior_res, prior_res_b0, prior_res_b1, prior_res_amp


class BSplinePoissonModel1D_numba_globalE0(BSplinePoissonModel1D_numba):
    """
    Numba-accelerated model with a single global Holtsmark E0 parameter 
    (no spatial variation for the microfield).
    """
    def setup_params(self, base_params: Parameters) -> Parameters:
        params = super().setup_params(base_params)
        
        # Remove spatially varying c_E0_i parameters
        for i in range(self.n_splines):
            if f'c_E0_{i}' in params:
                params.pop(f'c_E0_{i}')
                
        # Add a single global E0 parameter
        mean_E0 = np.mean(self.E0_vec_init)
        params.add('E0', value=mean_E0, min=1e-20, max=5.0, vary=False)
        
        return params

    def forward_physics(self, params, coeffs=None, return_amp=False):
        if coeffs is None:
            c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
            c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
            c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
            c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
            E0_val = params['E0'].value
        else:
            c, E0_val, c_b0, c_b1, c_amp = coeffs

        # Evaluate vectors using optimized Numba
        phi_vec, E_vec, grad_vec, b0_vec, b1_vec, amp_vec = _bspline_eval_vectors_numba_global_E0(
            c, c_b0, c_b1, c_amp, self.B, self.B_d1, self.B_d2
        )
        E0_vec = np.full_like(self.x, E0_val)
        
        fshift = params['fshift'].value if 'fshift' in params else 0.0
        f_shifted = self.f - fshift

        S_pred = self.signal_sim.holtsmark_spectrum(
            f_shifted, params, efield=E_vec, grad_vec=grad_vec, E0=E0_vec, amp=1.0)
        
        # Parallelly apply background offsets and spatial amplitudes 
        S_pred = _apply_bg_numba(S_pred, amp_vec, b0_vec, b1_vec, f_shifted)

        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
        
        if return_amp:
            return S_pred, E_vec, grad_vec, phi_vec, E0_vec, amp_vec
        return S_pred, E_vec, grad_vec, phi_vec, E0_vec

    def residuals(self, params: Parameters, freq: NDArray, data: NDArray, data_err: NDArray|None = None) -> NDArray:
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines)])
        c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines)])
        c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines)])
        E0_val = params['E0'].value
        
        coeffs = (c, E0_val, c_b0, c_b1, c_amp)
        S_pred, E_vec, _, phi_vec, E0_vec = self.forward_physics(params, coeffs=coeffs)

        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            data_res = difference.flatten()
            scale_factor = self.noise_sigma
            data_res = (difference / scale_factor).flatten()
        else:
            data_err_norm = data_err / self.data_max
            scale_factor = np.mean(data_err_norm)
            data_res = (difference / data_err_norm).flatten()
            
        # Extracted spline smoothing penalties calculated in Numba
        prior_res, prior_res_b0, prior_res_b1, prior_res_amp = _calc_prior_res_numba_global_E0(
            c, c_b0, c_b1, c_amp, self.D, self.smooth_param
        )
        
        return np.concatenate([data_res, 
                               prior_res / scale_factor, 
                               prior_res_b0 / scale_factor, 
                               prior_res_b1 / scale_factor, 
                               prior_res_amp / scale_factor])







#------------------------LOWER RESOLUTION BACKGROUND---------------------------
@njit(fastmath=True, cache=True)
def _bspline_eval_vectors_numba_split_basis(c, c_b0, c_b1, c_amp, B, B_d1, B_d2, B_bg):
    # Fast BLAS matrix multiplications
    phi_vec = B @ c
    E_vec = -(B_d1 @ c) * 10.0
    grad_vec = -(B_d2 @ c) * 10.0
    b0_vec = B_bg @ c_b0
    b1_vec = B_bg @ c_b1
    amp_vec = B_bg @ c_amp
    return phi_vec, E_vec, grad_vec, b0_vec, b1_vec, amp_vec


@njit(fastmath=True, cache=True)
def _calc_prior_res_numba_split_basis(c, c_b0, c_b1, c_amp, D, D_bg, smooth_param, smooth_param_bg):
    # Calculate penalties efficiently
    prior_res = np.sqrt(smooth_param) * (D @ c)
    prior_res_b0 = np.sqrt(smooth_param_bg * 0.1) * (D_bg @ c_b0)
    prior_res_b1 = np.sqrt(smooth_param_bg * 0.1) * (D_bg @ c_b1)
    prior_res_amp = np.sqrt(smooth_param_bg * 0.1) * (D_bg @ c_amp)
    return prior_res, prior_res_b0, prior_res_b1, prior_res_amp


class BSplinePoissonModel1D_numba_globalE0_splitBg(BSplinePoissonModel1D_numba_globalE0):
    """
    Subclass that uses a decoupled (and typically smaller) number of B-spline 
    coefficients for background drifts and amplitude (b0, b1, amp) compared 
    to the main potential (phi) splines. This prevents overfitting the background
    while preserving high spatial resolution for the Electric field.
    """
    def __init__(self, 
                 data: SpectralData, 
                 field_ref: FieldReference, 
                 signal_sim: GPPoissonSignalSimulator1D, 
                 E_vec_init: NDArray, 
                 E0_vec_init: NDArray|None = None, 
                 n_splines: int = 32, 
                 n_splines_bg: int = 10,
                 spline_degree: int = 3,
                 smooth_param: float = 1e4, 
                 smooth_param_E0: float = 1e4, 
                 smooth_param_bg: float = 1.0,
                 zero_bnd_efield: bool = True):
        
        super().__init__(data, field_ref, signal_sim, E_vec_init, E0_vec_init, 
                         n_splines, spline_degree, smooth_param, smooth_param_E0, 
                         zero_bnd_efield)
        
        self.n_splines_bg = n_splines_bg
        self.smooth_param_bg = smooth_param_bg
        
        if n_splines_bg <= self.k and n_splines_bg > 1:
            raise ValueError("Number of background splines must be greater than spline degree or exactly 1.")
            
        # Handle single-coefficient case (flat spatial background) gracefully
        if n_splines_bg == 1:
            self.B_bg = np.ones((len(self.x), 1))
            self.D_bg = np.zeros((1, 1))
        else:
            n_internal_knots_bg = n_splines_bg - self.k - 1
            internal_knots_bg = np.linspace(self.x[0], self.x[-1], n_internal_knots_bg + 2)[1:-1]
            self.knots_bg = np.concatenate(([self.x[0]] * (self.k + 1), internal_knots_bg, [self.x[-1]] * (self.k + 1)))

            self.B_bg = np.zeros((len(self.x), n_splines_bg))
            for i in range(n_splines_bg):
                c_idx = np.zeros(n_splines_bg)
                c_idx[i] = 1
                spl = BSpline(self.knots_bg, c_idx, self.k, extrapolate=False)
                self.B_bg[:, i] = spl(self.x)
                
            if self.n_splines_bg >= 4:
                self.D_bg = diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(self.n_splines_bg - 3, self.n_splines_bg)).toarray()
            elif self.n_splines_bg >= 3:
                self.D_bg = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(self.n_splines_bg - 2, self.n_splines_bg)).toarray()
            else:
                self.D_bg = diags([-1.0, 1.0], [0, 1], shape=(self.n_splines_bg - 1, self.n_splines_bg)).toarray()

    def setup_params(self, base_params: Parameters) -> Parameters:
        params = super().setup_params(base_params)
        
        # Remove parent's background splines (they were populated up to self.n_splines)
        for i in range(self.n_splines):
            if f'c_b0_{i}' in params: params.pop(f'c_b0_{i}')
            if f'c_b1_{i}' in params: params.pop(f'c_b1_{i}')
            if f'c_amp_{i}' in params: params.pop(f'c_amp_{i}')
            
        init_amp = params['amp'].value if 'amp' in params else 100.0
        for i in range(self.n_splines_bg):
            params.add(f'c_b0_{i}', value=1e-4)
            params.add(f'c_b1_{i}', value=1e-4)
            params.add(f'c_amp_{i}', value=init_amp, min=0.0)
            
        return params

    def forward_physics(self, params, coeffs=None):
        if coeffs is None:
            c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
            c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines_bg)])
            c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines_bg)])
            c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines_bg)])
            E0_val = params['E0'].value
        else:
            c, E0_val, c_b0, c_b1, c_amp = coeffs

        phi_vec, E_vec, grad_vec, b0_vec, b1_vec, amp_vec = _bspline_eval_vectors_numba_split_basis(
            c, c_b0, c_b1, c_amp, self.B, self.B_d1, self.B_d2, self.B_bg
        )
        E0_vec = np.full_like(self.x, E0_val)
        
        fshift = params['fshift'].value if 'fshift' in params else 0.0
        f_shifted = self.f - fshift

        S_pred = self.signal_sim.holtsmark_spectrum(
            f_shifted, params, efield=E_vec, grad_vec=grad_vec, E0=E0_vec, amp=1.0)
        
        S_pred = _apply_bg_numba(S_pred, amp_vec, b0_vec, b1_vec, f_shifted)

        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec, E0_vec

    def residuals(self, params: Parameters, freq: NDArray, data: NDArray, data_err: NDArray|None = None) -> NDArray:
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        c_b0 = np.array([params[f'c_b0_{i}'].value for i in range(self.n_splines_bg)])
        c_b1 = np.array([params[f'c_b1_{i}'].value for i in range(self.n_splines_bg)])
        c_amp = np.array([params[f'c_amp_{i}'].value for i in range(self.n_splines_bg)])
        E0_val = params['E0'].value
        
        coeffs = (c, E0_val, c_b0, c_b1, c_amp)
        S_pred, E_vec, _, phi_vec, E0_vec = self.forward_physics(params, coeffs=coeffs)

        data_norm = data / self.data_max
        difference = data_norm - S_pred
        
        if data_err is None:
            data_res = difference.flatten()
            scale_factor = self.noise_sigma
            data_res = (difference / scale_factor).flatten()
        else:
            data_err_norm = data_err / self.data_max
            scale_factor = np.mean(data_err_norm)
            data_res = (difference / data_err_norm).flatten()
            
        prior_res, prior_res_b0, prior_res_b1, prior_res_amp = _calc_prior_res_numba_split_basis(
            c, c_b0, c_b1, c_amp, self.D, self.D_bg, self.smooth_param, self.smooth_param_bg
        )
        
        return np.concatenate([data_res, 
                               prior_res / scale_factor, 
                               prior_res_b0 / scale_factor, 
                               prior_res_b1 / scale_factor, 
                               prior_res_amp / scale_factor])


class BSplinePoissonModel1D_numba_globalE0_globalBg(BSplinePoissonModel1D_numba_globalE0):
    """
    Subclass that completely eliminates spatially varying background and 
    amplitude parameters. It models the background as a global linear drift 
    (b0, b1) and a single global amplitude scalar, drastically reducing parameter 
    count and preventing spatial jitter overfitting.
    """
    def setup_params(self, base_params: Parameters) -> Parameters:
        params = super().setup_params(base_params)
        
        # Remove all B-spline background and amplitude parameters inherited from parent
        for i in range(self.n_splines):
            if f'c_b0_{i}' in params: params.pop(f'c_b0_{i}')
            if f'c_b1_{i}' in params: params.pop(f'c_b1_{i}')
            if f'c_amp_{i}' in params: params.pop(f'c_amp_{i}')
            
        # Replace with single pure scalars
        init_amp = params['amp'].value if 'amp' in params else 100.0
        params.add('b0', value=1e-4)
        params.add('b1', value=1e-4)
        params.add('amp', value=init_amp, min=0.0)
            
        return params

    def forward_physics(self, params, coeffs=None):
        if coeffs is None:
            c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
            E0_val = params['E0'].value
            b0 = params['b0'].value
            b1 = params['b1'].value
            amp = params['amp'].value
        else:
            c, E0_val, b0, b1, amp = coeffs

        # Fast BLAS matrix multiplications for E-field only
        phi_vec = self.B @ c
        E_vec = -(self.B_d1 @ c) * 10.0
        grad_vec = -(self.B_d2 @ c) * 10.0
        E0_vec = np.full_like(self.x, E0_val)
        
        fshift = params['fshift'].value if 'fshift' in params else 0.0
        f_shifted = self.f - fshift

        S_pred = self.signal_sim.holtsmark_spectrum(
            f_shifted, params, efield=E_vec, grad_vec=grad_vec, E0=E0_vec, amp=1.0)
        
        # Apply scalar background in Numba
        S_pred = _apply_global_bg_numba(S_pred, amp, b0, b1, f_shifted)

        if S_pred.shape != self.data.shape and S_pred.T.shape == self.data.shape:
            S_pred = S_pred.T
            
        return S_pred, E_vec, grad_vec, phi_vec, E0_vec

    def residuals(self, params: Parameters, freq: NDArray, data: NDArray, data_err: NDArray|None = None) -> NDArray:
        # Extract parameters manually for the scalar case
        c = np.array([params[f'c_{i}'].value for i in range(self.n_splines)])
        coeffs = (c, params['E0'].value, params['b0'].value, params['b1'].value, params['amp'].value)
        
        S_pred, E_vec, _, phi_vec, E0_vec = self.forward_physics(params, coeffs=coeffs)
        
        difference = (data / self.data_max) - S_pred
        data_res = difference.flatten() if data_err is None else (difference / (data_err / self.data_max)).flatten()
            
        if data_err is None:
            scale_factor = self.noise_sigma
            data_res = (difference / scale_factor).flatten()
        else:
            data_err_norm = data_err / self.data_max
            scale_factor = np.mean(data_err_norm)
            data_res = (difference / data_err_norm).flatten()
            
        # Only penalize the electric field curvature, leaving background entirely penalty-free
        prior_res = np.sqrt(self.smooth_param) * (self.D @ c) / scale_factor
        
        return np.concatenate([data_res, prior_res])


class BSplinePoissonModel1D_numba_adaptive(BSplinePoissonModel1D_numba_globalE0):
    """
    Applies Spatially Adaptive Smoothing to the Electric Field.
    The smoothing penalty is relaxed in regions of high Electric Field 
    (where SNR is naturally lower) to allow the covariance matrix to capture
    the true, larger physical uncertainty.
    """
    def __init__(self, *args, adapt_strength: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Reconstruct the initial E-field
        E_init = -np.gradient(self.phi_vec_init, self.x) * 10.0
        self.update_adaptive_weights(E_init, adapt_strength)

    def update_adaptive_weights(self, E_current: NDArray, adapt_strength: float = 3.0):
        """
        Updates the smoothing matrix D based on a given Electric field array.
        This can be called iteratively between fits to refine the adaptive penalty.
        """
        # 1. Map the rows of the difference matrix D to the spatial grid
        M = self.D.shape[0]
        x_indices = np.linspace(0, len(self.x) - 1, M).astype(int)
        
        # 2. Extract E-field at these spatial locations (using absolute values)
        E_local = np.abs(E_current[x_indices])
        E_max = np.max(E_local) if np.max(E_local) > 1e-6 else 1.0
        
        # 3. Calculate weights (exponential drop-off in high E-field regions)
        # adapt_strength = 3.0 means the penalty drops to exp(-3) ~ 5% at E_max.
        weights = np.exp(-adapt_strength * (E_local / E_max))
        
        # 4. Reconstruct base D matrix depending on spline count
        if self.n_splines >= 4:
            base_D = diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(self.n_splines - 3, self.n_splines)).toarray()
        elif self.n_splines >= 3:
            base_D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(self.n_splines - 2, self.n_splines)).toarray()
        else:
            base_D = diags([-1.0, 1.0], [0, 1], shape=(self.n_splines - 1, self.n_splines)).toarray()
            
        # 5. Multiply each row of the standard base D matrix by its spatial weight
        self.D = weights[:, np.newaxis] * base_D
