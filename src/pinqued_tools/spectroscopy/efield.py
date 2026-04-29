'''
Contains classes for electric field reconstruction
from Stark-split Rydberg EIT spectra 

Author: Mykhailo Vorobiov
'''
#%%
from typing import Callable, Dict
from numpy.typing import NDArray

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from scipy.interpolate import CubicSpline
from lmfit import minimize, Parameters, fit_report

from pinqued_tools.spectroscopy.spectrum import SpectralData, Axes0D
from pinqued_tools.spectroscopy.lineshapes import HoltsmarkLine



# ---------------- PRE-CALCULATED REFERENCE STARK MAP ----------------------
class FieldReference():
    '''
    Class reads Rydberg levels positions vs E-field and interpolate 
    between calculated values for arbitrary field within the valid reference range.
    Additionally calculates gradient df/dE to account for Stark broadening.
    '''
    def __init__(self, 
                 csv_path: str,
                 atol = 0.3,
                 n_interp = 4):
        '''
        Reads a file with reference dependence of Stark split positions 
        of Rydberg levels vs. E-field as calculated by ARC or any other 
        method. 
        '''

        self._atol = atol
        self._n_interp = n_interp

        # 1. Read refernce values into a dictionary
        arc_ref = pd.read_csv(csv_path).to_dict('list')

        # 2. Separate E-field column and create an extended domain for interpolation
        original_efield_array = np.array(arc_ref['E-field (V/cm)'])
        efield_mirrored = -np.flip(original_efield_array[1:])

        self._efield_interpolation_domain = np.concatenate((efield_mirrored, original_efield_array))
        self._efield = np.array(arc_ref['E-field (V/cm)'])

        # 3. Delete E-field column from the original dictionary
        del arc_ref['E-field (V/cm)']

        # 3. Define dictionary with frequency detunings only
        self._detunings = {}
        self._detunings_interpolation_domain = {}
        for key, detuning in arc_ref.items():
            original_detuning = np.array(detuning)
            detuning_mirrored = np.flip(original_detuning[1:])
            detuning_interpolation_domain = np.concatenate((detuning_mirrored, original_detuning))
            self._detunings[key] = original_detuning
            self._detunings_interpolation_domain[key] = detuning_interpolation_domain
            
    @property
    def efield(self) -> NDArray:
        return self._efield
    
    @property
    def detunings(self) -> Dict[str, NDArray]:
        return self._detunings
   
    @property
    def level_labels(self) -> list[str]:
        return list(self._detunings.keys())
    
    def interp(self, efield: float, method='spline') -> list[tuple[float, float]]:
        '''
        Interpolates between points of the reference for a given E-field.
        Calculates 1st derivative of the reference f(E) dependence.
        '''
        if method == 'poly':
            return self.interp_poly(efield)
        elif method == 'spline':
            return self.interp_spline(efield)
        else:
            raise ValueError("Invalid interpolation method. Choose 'poly' or 'spline'.")

    def interp_poly(self, efield: float) -> list[tuple[float, float]]:
        '''
        Interpolates between points of the reference for a given E-field.
        Calculates 1st derivative of the reference f(E) dependence.
        '''
        efield_reference = self._efield_interpolation_domain
        detunings_reference = self._detunings_interpolation_domain
        

        # 3. Extract a portion of the reference that is closest to the E-field
        #    value `efield` 
        closest_field = np.isclose(efield, efield_reference, atol=self._atol)
        idx_tmp = np.where(closest_field)
        closest_field_idx = np.min(idx_tmp)
        lower_lim_field = closest_field_idx - self._n_interp
        upper_lim_field = closest_field_idx + self._n_interp + 1

        x = efield_reference[lower_lim_field:upper_lim_field]

        detunings_interpolated = []
        for value in detunings_reference.values():
            y = value[lower_lim_field :upper_lim_field]

        # 4. Interpolate reference values in-between the known values
        #    using a second degree polynomial and define a polynomial object
            poly_coefs = np.polyfit(x, y, 2)
            polynomial = np.poly1d(poly_coefs)

        # 5. Calculate peak position with its 1st and 2nd derivatives wrt E-field
            f = polynomial(efield) # freq. position at `efield`
            df_de = polynomial.deriv(1)(efield) # first derivative
            detunings_interpolated.append((f, df_de))

        return detunings_interpolated

    def interp_spline(self, efield: float) -> list[tuple[float, float]]:
        '''
        Alternative interpolation method using cubic splines.
        '''
        efield_reference = self._efield_interpolation_domain
        detunings_reference = self._detunings_interpolation_domain

        detunings_interpolated = []
        for value in detunings_reference.values():
            cs = CubicSpline(efield_reference, value)
            f = cs(efield) # freq. position at `efield`
            df_de = cs.derivative(1)(efield) # first derivative
            detunings_interpolated.append((f, df_de))

        return detunings_interpolated
    

#----------------------- SPECTRUM SIMULATOR -------------------------
class SignalSimulator():
    '''
    Class that based on the Rydberg levels Stark splitting 
    from generated by the `FieldReference` class simulates EIT spectra.
    '''
    def __init__(self, 
                 reference: FieldReference,
                 lineshape_func: Callable|None = None
                 ):
        self._reference = reference
        self._lineshape_func = lineshape_func

        if lineshape_func is None:
            print('No lineshape function provided, using Holtsmark lineshape by default.')
            self._hline_list = self._holtsmark_spectrum_prepare()
            print(len(self._hline_list))

    def _holtsmark_spectrum_prepare(self) -> list[HoltsmarkLine]:
        '''
        Simulate EIT signal for a given electric field using the Holtsmark lineshape.
        '''
        line_keys = self._reference.level_labels
        efield_reference = self._reference.efield
        stark_reference = self._reference.detunings[line_keys[0]]
        
        hline_list = []
        for key in line_keys:
             print(f'Preparing Holtsmark line for {key}...')
             stark_reference = self._reference.detunings[key]
             hline = HoltsmarkLine(efield_reference, stark_reference)
             # 1. Define your 4D parameter space
             freq_grid = np.linspace(-1200, 400, 300)  # <-- The frequency axis for the LUT
             efield_grid = np.linspace(0.0, 45.0, 30)
             E0_grid = np.linspace(0.01, 20.0, 20)
             width_grid = np.linspace(20.0, 50.0, 10)  # <-- The new dimension
             hline.build_lut(freq_grid, efield_grid, E0_grid, width_grid, base_model='2d')
             hline_list.append(hline)
             print(f'Finished preparing Holtsmark line for {key}.')

        print(f'Ready to simulate spectra with Holtsmark spectrum! Number of lines {len(hline_list)}')
        return hline_list

    
    def holtsmark_spectrum(self, 
                           freq: NDArray, 
                           params: Parameters
                           ) -> NDArray:
        '''
        Simulate EIT signal for a given electric field using the Holtsmark lineshape.
        '''
        efield = params['efield'].value
        scale_factor = params['amp'].value
        width = params['width'].value
        E0 = params['E0'].value
        r_amp = [params[f'rel_amp_{i}'].value for i in range(len(self._hline_list))]
        spectrum = np.zeros_like(freq)
        for hline, ai in zip(self._hline_list, r_amp):
            spectrum += hline(freq, efield, width, E0, ai)
        spectrum *= scale_factor
        return spectrum

    def holtsmark_spectrum_bg(self, 
                           freq: NDArray, 
                           params: Parameters
                           ) -> NDArray:
        '''
        Simulate EIT signal for a given electric field using the Holtsmark lineshape.
        '''
        signal = self.holtsmark_spectrum(freq, params)
        bg = self.bg_drifts(freq, params, poly_terms=2)
        spectrum = signal + bg
        return spectrum

    def signal(self, 
               freq: NDArray, 
               params: Parameters,
               **kwargs) -> NDArray[np.float64]:
        '''
        Simulate EIT signal for a given electric field
        '''
        scale_factor = params['amp'].value
        width_0 = params['width_0'].value
        gradE_dr = params['gradE_dr'].value
        efield = params['efield'].value

        ref = self._reference.interp(efield)

        spectrum = np.zeros_like(freq)
        r_amp = [params[f'rel_amp_{i}'].value for i in range(len(ref))]
        for (fpos, df_de), amp in zip(ref, r_amp):
            width = width_0  - df_de * gradE_dr
            spectrum += amp * self._lineshape_func(freq, fpos, width, **kwargs)
        spectrum *= scale_factor
        return spectrum
    
    def bg_drifts(self, 
                  freq: NDArray, 
                  params: Parameters,
                  poly_terms: int = 2,
                  **kwargs):
        coefs = [params[f'b{i}'].value for i in range(poly_terms) if params[f'b{i}'] is not None]
        poly = np.poly1d(coefs)
        if len(coefs) < poly_terms:
            return np.zeros_like(freq)
        return poly(freq)

    def signal_with_bg(self, 
                       freq: NDArray, 
                       params: Parameters,
                       poly_terms: int = 3,
                       **kwargs) -> NDArray[np.float64]:
        signal = self.signal(freq, params, **kwargs)
        bg = self.bg_drifts(freq, params, poly_terms=poly_terms)
        return signal + bg

    def signal_with_bg_shifted(self, 
                       freq: NDArray, 
                       params: Parameters,
                       poly_terms: int = 3,
                       **kwargs) -> NDArray[np.float64]:
        f_shifted = freq - params['f_shift'].value
        signal = self.signal(f_shifted, params, **kwargs)
        bg = self.bg_drifts(f_shifted, params, poly_terms=poly_terms)
        return signal + bg




# --------------------  CLASS GENERATING RESIDUALS FOR MODEL FITTING ---------------     
class FitModel():
    '''
    Class for fitting experimental EIT spectra using the SignalSimulator.
    '''
    def __init__(self, 
                 fit_func: Callable
                 ):
        self._fit_func = fit_func
        
    def residuals(self, 
                  params: Parameters,
                  freq: NDArray, 
                  data: NDArray, 
                  data_err: NDArray|None = None
                  ) -> NDArray:
        signal = self._fit_func(freq, params)
        difference = data - signal
        if data_err is None:
            return difference
        return difference / data_err


# --------------- CLASS THAT RUNS FITTING --------------------
class DataFitter():
    def __init__(self, 
                 data: SpectralData,
                 model: FitModel):
        self._data = data
        self._model = model
    
    def set_data(self, data: SpectralData):
        self._data = data

    def fit(self, params: Parameters):
        result = minimize(self._model.residuals, 
                          params, 
                          args=(self._data.axes.f, 
                                self._data.signal, 
                                self._data.signal_err))
        return result




#%%
if __name__=='__main__':
    # ----------------- Usage example ----------------------
    from pinqued_tools.spectroscopy.lineshapes import holtsmarkian
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    # Read reference Rydberg splittings
    ref_path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Scripts\\calculated_stark_maps\\stark_map_25D_MHz.csv'
    ref = FieldReference(ref_path)

    # define parameters of the spectrum
    params = {'efield': 0.6,
              'amp': 150, 
              'width_0': 30, 
              'gradE_dr': 2,
              'rel_amp': [0.6, 0.6, 1.0, 1.0, 1.0],
              'b0': 1e-3, 'b1': 1e-2, 'b3': 1e-4}
    
    params_sim = Parameters()
    for key, value in params.items():
        if key == 'rel_amp':
            continue
        params_sim.add(key, value=value)
    
    params_lmfit = Parameters()
    params_lmfit.add('efield', value=params['efield'], min=-0.1)
    params_lmfit.add('amp', value=params['amp']-20.0)
    params_lmfit.add('width_0', value=params['width_0']+10.0)
    params_lmfit.add('gradE_dr', value=params['gradE_dr']-1.0)


    # Instantiate signal simulator object
    sim = SignalSimulator(ref, holtsmarkian)


    # Generate detunings
    freq = np.linspace(200, -1500, 700)

    # Simulate signal
    signal = sim.signal(freq, params=params_sim, normalized=True)

    sigma = 1.0
    noise = np.random.normal(loc=0, scale=sigma, size=signal.shape)/(signal+1)
    signal_err =  sigma/(signal+1)
    signal_noise = signal + noise

    spectrum = SpectralData(signal=signal_noise, 
                            axes = Axes0D(f=freq),
                            signal_err=signal_err)

    fm = FitModel(sim)
    df = DataFitter(spectrum, fm).fit(params_lmfit)
    print(fit_report(df))
    print(df.params)

    # Plot results
    fig, ax = plt.subplots(figsize=(4,2))
    ax.set_title(f'Simulated EIT spectrum ($E = ${params["efield"]:.1f} V/cm)')
    ax.plot(freq, signal, linewidth=1.5)
    ax.fill_between(y1=signal, x=freq, y2=-2, color='C0', alpha=0.2)
    ax.scatter(x=freq, y=signal_noise, 
               marker='.', s=5,
               color='C3', alpha=0.5)
    # ax.plot(freq, df.best_fit, linewidth=1.5, color='C1')
    ef =  ref.interp(params['efield'])
    for label, amp, (fpos, _) in zip(ref.level_labels, params['rel_amp'], ef):
        ax.axvline(x=fpos, color='C3', linestyle='--')
    ax.set_xlabel('Detuning (MHz)')
    ax.set_ylabel('EIT Signal $S$ (%)')

# %%
