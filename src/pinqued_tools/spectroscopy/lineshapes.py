#%%
import numpy as np

from numpy.typing import NDArray

from abc import ABC

from scipy.interpolate import interp1d
from scipy.integrate import quad

def gaussian(freq: NDArray, 
             fpos: float = 0.0, # Units of `freq`
             width: float = 20.0, # Units of `freq`
             amplitude: float = 1.0,
             normalized: bool = True) -> NDArray:
    '''
    Gaussian lineshape
    '''
    sigma = width / np.sqrt(8*np.log(2))
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    shape = np.exp( - 0.5 * ((freq - fpos) / sigma)**2)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape


def lorentzian(freq: NDArray, 
              fpos: float = 0.0, # Units of `freq`
              width: float = 20.0, # Units of `freq`
              amplitude: float = 1.0,
              normalized: bool = True) -> NDArray:
    '''
    Lorentzian lineshape
    '''
    norm = 2.0 / (np.pi * width)
    shape = 1.0 / (1.0 + (2.0 * (freq - fpos) / width)**2)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape


def holtsmarkian(freq: NDArray, 
                fpos: float = 0.0, # Units of `freq`
                width: float = 20.0, # Units of `freq`
                amplitude: float = 1.0,
                normalized: bool = True) -> NDArray:
    '''
    Holtsmark lineshape
    '''
    norm = (5.0/(2.0*np.pi))*np.sin(2.0*np.pi/5) / width
    arg = (2 * np.abs(freq - fpos) / width)**(2.5)
    shape = 1.0 / (1.0 + arg)
    if normalized:
        return amplitude * norm * shape
    return amplitude * shape

def lineshape(freq: NDArray, 
              params: dict):
    '''
    Any lineshape depending that is defined above
    the function itself must be passed as e.g. `func: lorentzian`
    '''
    # Extract spectral lineshape function from dictioary
    shape_function = params['func']  
    # remove spectral lineshape function `params` dict 
    function_parameters = {k: v for k, v in params.items() if k != 'func'}
    # Use `shape_function` to generate lineshape using `function_parameters`
    return shape_function(freq, **function_parameters)

def simulate_spectrum(freq: NDArray, 
                      params: list[dict],
                      return_shapes: bool = False) -> dict|NDArray:
    '''
    Simulates a spectrum based on the set of lineshapes provided as functions
    withing the list of dictionaries `params`. If `return_shapes` is True, then 
    the function returns dict with the following entries:
     'shapes_list' containing separate spectral lines
     'spectrum' sum of the spectral lines i.e. total spectrum.
    Otherwise, only total spectrum is returned as an numpy array.
    '''
    spectrum = np.zeros_like(freq)
    for p in params:
        spectrum += lineshape(freq, p)
    if return_shapes:
        shapes = []
        for p in params:
            shape = lineshape(freq, p)
            shapes.append(shape)
        return {'spectrum': spectrum, 'shapes_list': shapes}
    return spectrum

# ------------------- COMPLICATED LINESHAPES -------------------

class BaseSpectralLine(ABC):
    '''
    Base class for spectral lineshapes. 
    '''
    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def __call__(self, 
                 freq: NDArray, 
                 fpos: float = 0.0, # Units of `freq`
                 width: float = 20.0, # Units of `freq`
                 amplitude: float = 1.0) -> NDArray:
        raise NotImplementedError("Subclasses must implement the __call__ method.")

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d, RegularGridInterpolator, CubicSpline
from numba import njit, prange

# --Holtsmark lineshape --
@njit(parallel=True, fastmath=True)
def _fast_lorentzian_sum(freq: NDArray, 
                         shifts_flat: NDArray, 
                         weights_flat: NDArray, 
                         width: float, 
                         amplitude: float = 1.0) -> NDArray:
    '''
    Fast Lorentzian sum using Numba JIT compilation.
    '''
    spectrum = np.zeros(freq.shape[0], dtype=np.float64)
    gamma_half = width / 2.0
    gamma_half_sq = gamma_half**2
    
    for i in prange(freq.shape[0]):
        val = 0.0
        f = freq[i]
        for j in range(shifts_flat.shape[0]):
            detuning = f - shifts_flat[j]
            spectrum[i] += weights_flat[j] / (detuning**2 + gamma_half_sq)
    spectrum *= gamma_half_sq
        
    # Analytical normalization over all space prevents artificial 
    # amplitude explosion when a peak shifts outside the frequency window
    # We omit np.sum(weights_flat) here to ensure that if the microfield 
    # distribution is truncated (e.g. at high E0), the missing probability mass 
    # correctly reduces the peak amplitude rather than artificially inflating it.
    total_area = np.pi * gamma_half 
      
    if total_area > 0:
        return amplitude * spectrum / total_area
        
    return amplitude * spectrum

@njit(fastmath=True)
def _polyval(x: np.float64, coef: NDArray) -> np.float64:
    y = 0
    for j in range(coef.size):
        y = y*x + coef[j]
    return y

@njit(fastmath=True)
def _polyval_arr(x: NDArray, coef: NDArray) -> NDArray:
    y = np.zeros_like(x)
    for i in range(x.size):
        for j in range(coef.size):
            y[i] = y[i]*x[i] + coef[j]
    return y

class Poly():
    def __init__(self, coef):
        self.coef = coef

    def polyval(self, x):
        """Calculates polynomial with provided coefficients for every point x
           x - MUST be array!

           Note: numpy.polynomial.Polinomias has horrendous performance
           since it rescales x to a provided window and do other unnecessary things.
           numpu.poly1d has better performance but still about factor of 2 slower
        """
        return _polyval_arr(x, self.coef)

    def __call__(self, x):
        return self.polyval(x)

@njit(fastmath=True)  # note somehow parallel=True options make it factor of 10 slower
def ifleq_polyval(x: NDArray, limit: float, coef_leq: NDArray, coef_gt: NDArray) -> NDArray:
    y = np.zeros(x.size, dtype = np.float64)
    for i in range(x.size):
        if x[i] <= limit:
            y[i] = _polyval(x[i], coef_leq)
        else:
            y[i] = _polyval(x[i], coef_gt)
    return y

class StarkMap():
    """Do the Stark shift vs Electric field related calculation"""
    def __init__(self, Efield_reference, freq_shift_reference, maxStarkShiftMisMatch=1e-3):
        """Initialized with tabulated values of Stark shift vs Electric field
        The Efield_reference must be provided in the ascending order
        maxStarkShiftMisMatch - maximum allowed mistake by heuristic, in units of freq_shift_reference.

        Strongly assumes that Stark shifts push to negative for large
        Electric field values. I.e. similar to nD states.
        """
        self._Efield = Efield_reference.copy()  # Stark Electric field
        self._freq = freq_shift_reference.copy()  # frequency of stark shift
        #  Stark shift is quadratic for large enough Efield let's see if we can fit it with quadratic polynomial
        self._maxStarkShiftMisMatch = maxStarkShiftMisMatch
        self._approx_poly2ndOrder = Poly(np.polyfit(self._Efield[-3:], self._freq[-3:], 2))  # we can always make parabola on 3 points
        self._minEfild_for_poly2ndOderValidity = self._Efield[-3]
        self._approx_highOrder = None

        self._max_shift_indx = self._freq.argmax()
        self._max_shift_freq = self._freq[self._max_shift_indx]
        self._max_shift_Efield = self._Efield[self._max_shift_indx]

        self.monotonic = np.all(np.diff(self._freq) <= 0)  # is stark shift monotonically decreasing
        if not self.monotonic:
            print("The Stark shift are not monotonic, will split in two monotonic regions")

        # let's try to extend the range where 2nd degree polynomial still fits
        Np = len(self._Efield) 
        left_end = 0
        while (left_end < Np-3):
            Efmin = self._Efield[left_end]
            valid = self._Efield > Efmin
            p2 = Poly(np.polyfit(self._Efield[valid], self._freq[valid], 2))
            if np.abs(p2(self._Efield[valid]) - self._freq[valid]).max() < self._maxStarkShiftMisMatch:
                print(f"Stark map can be fitted with 2nd degree polynomial for E field > {Efmin}")
                self._approx_poly2ndOrder = p2
                self._minEfild_for_poly2ndOderValidity = self._Efield[left_end]
                break
            left_end += int(np.floor((Np - left_end)/2))

        # now let's find hight order approximation within tabulated values
        porder = 1
        maxTestOrder = min(20, Np)
        while(porder <= maxTestOrder):
            porder += 1
            phigh = Poly(np.polyfit(self._Efield, self._freq, porder))
            if np.abs(phigh(self._Efield) - self._freq).max() < self._maxStarkShiftMisMatch:
                print(f"Tabulated Stark map can be fitted with {porder} degree polynomial")
                self._approx_highOrder = phigh
                break

    def freq2Efield(self, freq, branch="falling"):
        """Calculates Electric field corresponding to a given shift
           
           Strongly assumes that Stark shifts push to negative for large
           Electric field values. I.e. similar to nD states.

           branch is used to chose are we working on "raising" or "falling" brunch
           of the Stark map. Note: that for monotonic StarkMap only "falling" makes sense
        """
        Ef = np.empty_like(freq)
        Ef[:] = np.nan
        # are there Stark shifts to get such frequencies
        if branch == "falling":
            reachable_freq = freq <= self._max_shift_freq
        if branch == "raising":
            reachable_freq = (self._freq[0] <= freq) & (freq <= self._max_shift_freq)
        if not np.any(reachable_freq):
            return Ef
        if branch == "falling":
            tab_mask = self._Efield >= self._max_shift_Efield
        else:  # raising branch
            tab_mask = self._Efield <= self._max_shift_Efield
        tabulated_freq = (self._freq[-1] <= freq) & (freq <= self._max_shift_freq)
        valid = tabulated_freq & reachable_freq
        if np.any(valid):
            # print("We are within tabulated values")
            if branch == "falling":
                tab_mask = self._Efield >= self._max_shift_Efield
                Ef[valid] = np.interp( -freq[valid], -self._freq[tab_mask], self._Efield[tab_mask])
            else:
                assert branch == "raising"
                tab_mask = self._Efield <= self._max_shift_Efield
                Ef[valid] = np.interp( freq[valid], self._freq[tab_mask], self._Efield[tab_mask])
        non_tabulated_freq = reachable_freq & ~tabulated_freq
        if np.any(non_tabulated_freq):
            assert branch == "falling"  # raising branch should not be beyond tabulated
            # FIXME: we now in the 2nd degree polynomial Stark shift
            # FIXME: we can find E cutoff from this polynomial in one step
            # searching for maximum field encompassing all leftover points
            fmin = freq.min()
            Ecut = 1.2*self._Efield[-1]
            Eprobe = np.linspace(0, Ecut, 2) # has to be array
            fprobe = self.Efield2freq(Eprobe)
            while np.all(fprobe > fmin):
                Ecut *= 1.2
                Eprobe = np.linspace(0, Ecut, 2)
                fprobe = self.Efield2freq(Eprobe)
            N_tab = 10000  # FIXME: do we need that many?
            E_tab = np.linspace(self._Efield[-1], Ecut, N_tab)
            f_tab = self.Efield2freq(E_tab)
            Ef[non_tabulated_freq] = np.interp( -freq[non_tabulated_freq], -f_tab, E_tab)
        return Ef

    def Efield2freq(self, Efield: NDArray):
        """Return Stark shifts array vs provided Electric field array"""
        f = np.empty_like(Efield)
        valid = Efield >= 0  # Efield is provided as magnitude, so negative numbers are illegal
        f[~valid] = np.nan
        f[valid] = ifleq_polyval(Efield[valid], self._minEfild_for_poly2ndOderValidity, self._approx_highOrder.coef, self._approx_poly2ndOrder.coef)
        return f

class HoltsmarkLine(BaseSpectralLine):
    '''
    Holtsmark lineshape as a subclass of `BaseSpectralLine`
    Supports 1D scalar, 2D vector, and ultra-fast 3D Look-Up Table (LUT) models.
    '''
    def __init__(self, 
                 efield_reference: NDArray,
                 stark_reference: NDArray,
                 normalized: bool = True,
                 n_efield_points: int = 1000):
        super().__init__(normalized)

        self.stark_interp = CubicSpline(efield_reference, stark_reference, extrapolate=False)
        self._efield_reference = efield_reference.copy()
        self._stark_reference= stark_reference.copy()
        self.stark_map = StarkMap(self._efield_reference, self._stark_reference)
        
        # Safe linear extrapolation up to 10x the calibration limit.
        # This prevents the Holtsmark tail from being truncated.
        max_ref_E = efield_reference[-1]
        # no need to over interpolate (it makes things slow and results are the same)
        self._dense_efield = np.zeros(len(self._efield_reference) + 1)
        self._dense_stark = np.zeros(len(self._efield_reference) + 1)
        self._dense_efield[:-1] = self._efield_reference[:]
        self._dense_stark[:-1] = self._stark_reference[:]
        self._dense_efield[-1] = self._efield_reference[-1]*10  # increase range by 10
        
        
        # FIXME: linear approximation is BAD for large Efield, it should be quadratic!
        # self._dense_stark[-1] = last_stark + last_slope * (self._dense_efield[-1] - max_ref_E)
        # FIXED: we now use quadratic extrapolation based on the last two points of the reference and their derivatives
        last_stark = self.stark_interp(max_ref_E)
        last_slope = self.stark_interp(max_ref_E, nu=1)
        last_second_deriv = self.stark_interp(max_ref_E, nu=2)
        self._dense_stark[-1] = last_stark + last_slope * (self._dense_efield[-1] - max_ref_E) + 0.5 * last_second_deriv * (self._dense_efield[-1] - max_ref_E)**2
        
        # 1. Pre-compute Holtsmark distribution and its analytical integral C(u)
        betas = np.linspace(0, 20.0, 2000)
        h_vals = np.array([self._integrate_holtsmark(b) for b in betas])
        self.H_beta_interp = interp1d(betas, h_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        
        self._dense_betas = np.linspace(0, 100.0, 200000)
        self._dense_h_vals = self.H_beta_interp(self._dense_betas)
        
        # Extend analytical tail for beta > 20.0 (H(beta) -> 1.496 * beta^-2.5)
        tail_mask = self._dense_betas > 20.0
        # FIXME: 1.496 is close but not precise, small discontinuity is visible
        self._dense_h_vals[tail_mask] = 1.496 / (self._dense_betas[tail_mask]**2.5)

        from scipy.integrate import cumulative_trapezoid
        integrand = np.zeros_like(self._dense_betas)
        integrand[1:] = self._dense_h_vals[1:] / self._dense_betas[1:]
        self._C_u_vals = cumulative_trapezoid(integrand, self._dense_betas, initial=0.0)

        self._lut_interpolator = None

    def _get_P_Etot(self, Etot_grid: NDArray, efield: float, E0: float) -> NDArray:
        """Exact analytical 1D projection of the 2D macroscopic + microfield sum."""
        assert efield >= 0
        assert E0 >= 0
        prob = np.empty_like(Etot_grid)
        valid = Etot_grid >= 0  # this E field magnitude distribution
        prob[~valid] = np.nan
        E0 = max(E0, 1e-1)
        
        if efield < 1e-6:
            betas = Etot_grid / E0
            prob[valid] = np.interp(betas[valid], self._dense_betas, self._dense_h_vals, left=0.0, right=0.0)
            return (1.0 / E0) * prob

        u_max = (Etot_grid[valid] + efield) / E0
        u_min = np.abs(Etot_grid[valid] - efield) / E0

        C_max = np.interp(u_max, self._dense_betas, self._C_u_vals, left=0.0, right=self._C_u_vals[-1])
        C_min = np.interp(u_min, self._dense_betas, self._C_u_vals, left=0.0, right=self._C_u_vals[-1])

        prob[valid] = (Etot_grid[valid] / (2.0 * efield * E0)) * (C_max - C_min)
        return prob

    def build_lut(self, 
                  freq_grid: NDArray, 
                  efield_grid: NDArray, 
                  E0_grid: NDArray, 
                  width_grid: NDArray,  # <-- Added width array
                  base_model: str = '2d'):
        """
        Pre-calculates a 4D Lineshape (E0, efield, width, freq) Look Up Table or library for instant recall.

        Available models:
         - '2d' - for smart/quick lineshape calculations (2d is bad name, should be 'smart'
         - 'bruteforce' - oversampled case usually slower
        """
        print(f"Building 4D Lineshape Library: {len(E0_grid)}x{len(efield_grid)}x{len(width_grid)}x{len(freq_grid)} points...")
        
        library = np.zeros((len(E0_grid), len(efield_grid), len(width_grid), len(freq_grid)))

        if base_model == '2d':
            for j, efield in enumerate(efield_grid):
                for i, E0 in enumerate(E0_grid):
                    for k, width in enumerate(width_grid): 
                        library[i, j, k, :] = self.line2d(freq_grid, efield, width, E0, 1.0)
        elif base_model == "bruteforce":
            for j, efield in enumerate(efield_grid):
                for i, E0 in enumerate(E0_grid):
                    for k, width in enumerate(width_grid): 
                        library[i, j, k, :] = self.line2d(freq_grid, efield, width, E0, 1.0, bruteforce=True)
        else:
            raise ValueError(f"Build lut with {base_model=} is not implemented")
                
        # 3. Create the 4-Dimensional Interpolator
        self._lut_freq = freq_grid.copy()
        self._lut_freq_max = self._lut_freq.max()
        self._lut_freq_min = self._lut_freq.min()
        # The lookup will be performed in E0, efield, and width space. It will return spectrum
        self._lut_interpolator = RegularGridInterpolator(
            (E0_grid, efield_grid, width_grid),
            library, 
            bounds_error=False, 
            fill_value=0.0
        )

        print("LUT Build Complete.")

    def save_lut(self, file_path: str) -> None:
        """
        Saves the generated 4D Look-Up Table to a compressed numpy .npz file.
        """
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` first.")
            
        grid_E0, grid_E, grid_W, grid_f = self._lut_interpolator.grid
        np.savez_compressed(
            file_path,
            E0_grid=grid_E0,
            efield_grid=grid_E,
            width_grid=grid_W,
            freq_grid=grid_f,
            library=self._lut_interpolator.values
        )
        print(f"LUT successfully saved to {file_path}")

    def save_lut_hdf5(self, file_path: str, group_name: str = 'holtsmark_lut') -> None:
        """
        Saves the generated 4D Look-Up Table to an HDF5 file.
        """
        import h5py
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` first.")
            
        grid_E0, grid_E, grid_W, grid_f = self._lut_interpolator.grid
        
        with h5py.File(file_path, 'a') as f:
            # Remove the group if it already exists to overwrite it cleanly
            if group_name in f:
                del f[group_name]
            group = f.create_group(group_name)
            group.create_dataset('E0_grid', data=grid_E0)
            group.create_dataset('efield_grid', data=grid_E)
            group.create_dataset('width_grid', data=grid_W)
            group.create_dataset('freq_grid', data=grid_f)
            group.create_dataset('library', data=self._lut_interpolator.values, compression='gzip', compression_opts=9)
        print(f"LUT successfully saved to {file_path} in group '{group_name}'")

    def line_lut(self, freq: NDArray, 
                 efield: float|NDArray = 0.0, 
                 width: float|NDArray = 20.0, 
                 E0: float|NDArray = 3.0, 
                 amplitude: float|NDArray = 1.0) -> NDArray:
        """
        Instant lineshape extraction from the pre-computed 4D Look-Up Table.
        Supports scalar inputs or 1D arrays for spatial variations.

        If 1D arrays submitted for Efield, width, or E0 then this arrays are brodcasted to each other
        and result is 2D array with sets of spectra linked to counter of (Efield, width, E0) list
        """
        if self._lut_interpolator is None:
            raise RuntimeError("LUT not initialized. Call `build_lut()` before using model='lut'.")
            
        efield = np.asarray(efield)
        width = np.asarray(width)
        E0 = np.asarray(E0)
        
        # Safely clip query parameters to the LUT grid boundaries to prevent out-of-bounds 
        # linear extrapolation, which can produce negative values or empty gaps in the spectra.
        grid_E0, grid_E, grid_W = self._lut_interpolator.grid
        E0 = np.clip(E0, grid_E0[0], grid_E0[-1])
        efield = np.clip(efield, grid_E[0], grid_E[-1])
        width = np.clip(width, grid_W[0], grid_W[-1])
        freq = np.clip(freq, self._lut_freq_min, self._lut_freq_max)
        # FIXME: I think it is better to fail with asserts, then do clip. --Eugeniy

        # Scalar parameters: return 1D frequency spectrum
        if efield.ndim == 0 and width.ndim == 0 and E0.ndim == 0:
            query_points = np.zeros(3)
            query_points[0] = E0
            query_points[1] = efield
            query_points[2] = width
            lut_spectrum = self._lut_interpolator(query_points).squeeze()
            spectrum = np.interp(freq, self._lut_freq, lut_spectrum)
            return amplitude * spectrum
        
        # Array parameters (spatial grid): return 2D array (spatial x frequency)
        # Safely broadcast arrays before meshing
        E0_b, efield_b, width_b, = np.broadcast_arrays(E0, efield, width)
        
        query_points = np.zeros((efield_b.size, 3))
        query_points[:, 0] = E0_b.ravel()
        query_points[:, 1] = efield_b.ravel()
        query_points[:, 2] = width_b.ravel()
        
        lut_spectra = self._lut_interpolator(query_points)
        # spectra = np.empty_like(lut_spectra)
        spectra = np.zeros((lut_spectra.shape[0], freq.shape[0]))
        for i in range(lut_spectra.shape[0]):
            spectra[i] = np.interp(freq, self._lut_freq, lut_spectra[i,:])

        if np.ndim(amplitude) > 0:
            amplitude = amplitude[:, np.newaxis]
            
        return amplitude * spectra

    def __call__(self, freq: NDArray, 
                 efield: float|NDArray = 0.0, 
                 width: float|NDArray = 20.0, 
                 E0: float|NDArray = 3.0,     
                 amplitude: float|NDArray = 1.0,
                 model: str = 'lut') -> NDArray:
        
        if model == '1d':
            return self.line1d(freq, efield, width, E0, amplitude)
        elif model == '2d':
            return self.line2d(freq, efield, width, E0, amplitude)
        elif model == 'lut':
            return self.line_lut(freq, efield, width, E0, amplitude)
        else:
            raise ValueError("Invalid model. Choose '1d', '2d', or 'lut'.")

    def line1d(self, freq: NDArray, 
                 efield: float = 0.0, 
                 width: float = 20.0, 
                 E0: float = 3.0, 
                 amplitude: float = 1.0) -> NDArray:
        
        E0_safe = max(E0, 1e-6)
        E_max = min(efield + 20.0 * E0_safe, self._dense_efield[-1])
        Etot_grid = np.linspace(0.0, max(E_max, 1e-3), 10000)
        dEtot = Etot_grid[1] - Etot_grid[0]

        E_m = Etot_grid - efield
        valid_mask = E_m >= 0
        betas = E_m / E0_safe
        H_vals = np.interp(betas, self._dense_betas, self._dense_h_vals, left=0.0, right=0.0)
        weights_flat = (1.0 / E0_safe) * H_vals * dEtot * valid_mask

        shifts_flat = np.interp(Etot_grid, self._dense_efield, self._dense_stark)

        return _fast_lorentzian_sum(freq, shifts_flat, weights_flat, width, amplitude)
    
    def line2d(self, freq: NDArray, 
               efield: float = 0.0, 
               width: float = 20.0, 
               E0: float = 3.0, 
               amplitude: float = 1.0,
               bruteforce = False,
               _branch = None) -> NDArray:
        """
        Generates the lineshape using a 2D vector summation of the external 
        DC field and the isotropic Holtsmark microfield.

        The freq MUST be sorted in the ascending order

        _branch is used internally to select raising or falling part of Stark Map,
                can take values None (default), "raising", and "falling"
        """
        # FIXME: for small E0 values p(E) looks like delta function
        # and we might miss the peak during sampling
        E0_safe = max(E0, 1e-6)
        if bruteforce:
            # worst case scenario, we cannot predict required grid
            E_max = min(efield + 20.0 * E0_safe, self._dense_efield[-1])
            Etot_grid = np.linspace(0.0, max(E_max, 1e-3), 10000)
            dEtot = Etot_grid[1] - Etot_grid[0]

            weights_flat = self._get_P_Etot(Etot_grid, efield, E0_safe) * dEtot
            shifts_flat = self.stark_map.Efield2freq(Etot_grid)
            return _fast_lorentzian_sum(freq, shifts_flat, weights_flat, width, amplitude)

        if (not self.stark_map.monotonic) and (_branch is None):
            lineshape_f = self.line2d( freq, efield, width, E0, amplitude, bruteforce, _branch = "falling")
            lineshape_r = self.line2d( freq, efield, width, E0, amplitude, bruteforce, _branch = "raising")
            return lineshape_r + lineshape_f
        else:
            if (self.stark_map.monotonic) and (_branch is None):
                _branch = "falling"  # for monotonic Stark Map frequency fall with Efield increase
            min_freq = freq[0] - 20*width
            max_freq = freq[-1] + 20*width
            
            # limit tested frequency to reachable by StarkMap
            max_freq = min(max_freq, self.stark_map._max_shift_freq)
            if _branch == "raising":
                min_freq = max(self.stark_map._freq[0], min_freq)
            if min_freq >= max_freq:
                # we are really testing for min_freq == max_freq, no candidates to test
                return freq*0  # lineshape strength is zero in this case

            shifts_flat = np.linspace(min_freq, max_freq, max(100, int(np.ceil((max_freq - min_freq)/width*20))))
            Etot_grid = self.stark_map.freq2Efield(shifts_flat, branch = _branch)

            # select only achievable Electric fields in Holtsmark distribution
            mask = (~np.isnan(Etot_grid))
            Etot_grid = Etot_grid[mask]
            if np.sum(mask) < 2:
                print("not enough hits of the matching E field on our frequency grid")
                return freq*0  # lineshape strength is zero in this case
            shifts_flat = shifts_flat[mask]

            dEtot = np.zeros_like(Etot_grid)
            dEtot[:-1] = np.abs(np.diff(Etot_grid))  # protect against falling branch case
            # trapezoid dE approximation since we are doing integral and dE is not even
            dEtot[-1] = 0
            dEtot[1:] += dEtot[:-1]
            dEtot /= 2

            weights_flat = self._get_P_Etot(Etot_grid, efield, E0_safe) * dEtot
        return _fast_lorentzian_sum(freq, shifts_flat, weights_flat, width, amplitude)
    
    def _integrate_holtsmark(self, beta):
        """Rigorous Holtsmark integral definition."""
        if beta == 0: return 0.0
        integrand = lambda x: x * np.sin(beta * x) * np.exp(-(x**1.5))
        result, _ = quad(integrand, 0, np.inf, limit=200)
        return (2.0 * beta / np.pi) * result
        

#%%
if __name__=='__main__':
    # Apply custom plotting style
    import matplotlib.pyplot as plt
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    # 1. Define list with parameters for each spectral line
    params = [
        {'func': gaussian, #<---- NOTE: spectral line function is a dict entry
         'fpos': 0.0,
         'width': 20,
         'normalized': False},
        {'func': lorentzian, 
         'fpos': 0.0,
         'width': 20,
         'normalized': False},
        {'func': holtsmarkian, 
         'fpos': 0.0,
         'width': 20,
         'normalized': False},
    ]

    # 2. Frequency detunings -100 to 100 MHz
    x = np.linspace(-100,100,1000)

    # 3. Plot all available spectral lines
    labels = ['Gauss', 'Lorentz', 'Holtsmark']
    fig, ax = plt.subplots()
    for p, ll in zip(params, labels):
        y = lineshape(x, p) # Calculate spectral lineshape
        ax.plot(x,y, linewidth=1.5, label=ll)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('EIT Signal $S$ (arb. units)')
    ax.legend()
# %%
