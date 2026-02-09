'''
Author: Mykhailo Vorobiov 
This file contains collection of functions for 
Langmuir probe analysis
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from lmfit import Model
from scipy.signal import find_peaks
import os

#%%

def generate_dummy_data():
    """
    Generates a synthetic noisy Langmuir probe trace with electron saturation.
    """
    V = np.linspace(-10, 20, 150)
    V_p = 10.0  # Plasma Potential
    T_e = 3.0   # Effective electron temperature in eV for the exponential part
    I_ion_sat = -1.0
    I_ion_slope = 0.02
    I_e_sat_base = 2.0
    I_e_sat_slope = 0.05
    
    # Model for ion current part
    I_ion = I_ion_sat + I_ion_slope * V

    # Model for electron current part with saturation
    # Exponential rise below Vp, linear increase above Vp
    I_electron = np.where(
        V <= V_p,
        I_e_sat_base * np.exp((V - V_p) / T_e),
        I_e_sat_base + I_e_sat_slope * (V - V_p)
    )
    
    I_true = I_ion + I_electron
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.01, V.shape)
    I_measured = I_true + noise
    
    return V.reshape(-1, 1), I_measured

def langmuir_subtract_ion_current(V_measured: np.ndarray, I_measured: np.ndarray, 
                                  tolerance=0.1):
    """
    Fits a straight line to an ion current and subtracts it from the measured current.
    """
    # 1. Fit a straight line to the ion current
    ion_slope = 0
    ion_intercept = 0
    for sample in range(3,len(V_measured)):
        x = V_measured[0:sample]
        y = I_measured[0:sample]
        slope, intercept = np.polyfit(x, y, 1)
        if np.abs((ion_slope - slope)/ion_slope) < tolerance:
            ion_slope = slope
            ion_intercept = intercept   
            break
        ion_slope = slope
        ion_intercept = intercept
    
    # 2. Subtract ion current from the measured current
    I_corrected = I_measured - (ion_slope * V_measured + ion_intercept)
    
    return I_corrected

def ei_parts_separator_index(I_measured: np.ndarray):
    "Extracts electron and ion current parts from the combined probe trace."
    # 1. Find index of minimal current value in the absolute current
    I_min_idx = np.argmin(np.abs(I_measured))
    return I_min_idx

def electron_ion_parts(V_measured: np.ndarray, I_measured: np.ndarray):
    "Extracts electron and ion current parts from the combined probe trace."
    # 1. Find index of minimal current value in the absolute current
    I_min_idx = np.argmin(np.abs(I_measured))+1
    print(f'Min index: {I_min_idx}')

    # 2. Extract ion part
    V_ion = V_measured[:I_min_idx]
    I_ion = I_measured[:I_min_idx]

    # 3. Extract electron part
    V_electron = V_measured[I_min_idx:]
    I_electron = I_measured[I_min_idx:]

    return V_ion, I_ion, V_electron, I_electron


def fit_ion_current(V_measured: np.ndarray, I_measured: np.ndarray,
                    tolerance=0.1, fit_type='linear', return_fit=False, 
                    V_ion_cutoff=-10, min_sample_number=1):
    """Fits ion parts with a straight line or exponential function."""

    V_ion, I_ion, _, _ = electron_ion_parts(V_measured, I_measured)

    if fit_type == 'linear':
        # Fit a straight line to the ion current
        ion_slope = 0
        ion_intercept = 0
        for sample in range(min_sample_number,len(V_measured)):
            x = V_ion[0:sample]
            y = np.log(-I_ion[0:sample])
            slope, intercept = np.polyfit(x, y, 1)
            if sample==min_sample_number: ion_slope = slope
            slope_fractional_change = np.abs((ion_slope - slope)/ion_slope)
            print(slope_fractional_change)
            if slope_fractional_change > tolerance:
                ion_slope = slope
                ion_intercept = intercept
                break
            ion_slope = slope
            ion_intercept = intercept
        ion_fit = -np.exp(ion_slope * V_measured + ion_intercept)
    elif fit_type == 'exponential':
        # Fit an exponential function to the ion current
        def exp_func(x, a, b, c):
            # return a * np.exp(b*x) + c
            return np.power((x-c),b)
        V_ion_fit = V_ion[V_ion < V_ion_cutoff]
        I_ion_fit = I_ion[V_ion < V_ion_cutoff]
        model = Model(exp_func)
        params = model.make_params(a=1, b=1.3, c=1e-7)
        result = model.fit(I_ion_fit, params, x=V_ion_fit)
        ion_fit = result.eval(x=V_measured)
    else:
        raise ValueError("Invalid fit_type. Choose 'linear' or 'exponential'.")
    
    I_corrected = I_measured - ion_fit
    if return_fit:
        return I_corrected, ion_fit
    
    return I_corrected

def fit_langmuir_gp(V_measured: np.ndarray, I_measured: np.ndarray,
                    n_grid=200):
    """Fits a Gaussian Process to the data and calculates the 1st and 2nd derivatives."""

    # 1. Reshape and Scale Data (Crucial for GP stability)
    # ---------------------------------------------------
    V_scaler = StandardScaler()
    I_scaler = StandardScaler()
    
    # Sklearn requires 2D arrays (N, 1)
    X_train = V_scaler.fit_transform(V_measured.reshape(-1, 1))
    y_train = I_scaler.fit_transform(I_measured.reshape(-1, 1)).ravel() # .ravel() for 1D target
    
    # 2. Configure and Fit GP
    # ---------------------------------------------------
    # We use a compound kernel: RBF (signal) + WhiteKernel (noise)
    # The length_scale of the RBF kernel is crucial. A larger length_scale
    # forces a smoother fit, which is necessary for a stable 2nd derivative.
    # We constrain the lower bound to prevent the optimizer from choosing a value
    # that is too small and leads to overfitting the noise.
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-4, 100.0)) \
           + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-7, 1e-5))
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.0)
    gp.fit(X_train, y_train)
    
    # 3. Predict on Dense Grid
    # ---------------------------------------------------
    # Create grid in scaled space
    X_grid = np.linspace(X_train.min(), X_train.max(), n_grid).reshape(-1, 1)
    
    # Predict normalized mean and std
    y_pred_scaled, sigma_scaled = gp.predict(X_grid, return_std=True)
    
    # 4. Analytical First and Second Derivative Calculation
    # ---------------------------------------------------
    # Extract learned hyperparameters
    rbf_kernel = gp.kernel_.k1
    l = rbf_kernel.length_scale
    
    # Precompute alpha weights (inv(K) * y)
    alpha = gp.alpha_
    
    # d_y_scaled / d_x_scaled and d2_y_scaled / d_x_scaled^2
    d_y_scaled = np.zeros(X_grid.shape[0])
    d2_y_scaled = np.zeros(X_grid.shape[0])
    
    for i, x_star in enumerate(X_grid):
        diff = x_star - X_train.flatten()
        k_val = np.exp(-(diff**2) / (2 * l**2))
        
        # First derivative
        d_k = -(diff / l**2) * k_val
        d_y_scaled[i] = np.dot(d_k, alpha)

        # Second derivative
        d2_k = (1 / l**2) * ((diff**2 / l**2) - 1) * k_val
        d2_y_scaled[i] = np.dot(d2_k, alpha)
    
    # 5. Inverse Transform and Rescale
    # ---------------------------------------------------
    # Recover physical units for V and I
    V_grid = V_scaler.inverse_transform(X_grid).flatten()
    I_pred = I_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Scale sigma (uncertainty)
    sigma = sigma_scaled * I_scaler.scale_[0]
    
    # Rescale Derivatives
    scale_factor_d1 = I_scaler.scale_[0] / V_scaler.scale_[0]
    dI_dV = d_y_scaled * scale_factor_d1
    
    scale_factor_d2 = I_scaler.scale_[0] / (V_scaler.scale_[0]**2)
    d2I_dV2 = d2_y_scaled * scale_factor_d2
    
    return V_grid, I_pred, dI_dV, d2I_dV2, sigma, gp, V_scaler, I_scaler


def floating_potential(V_grid: np.ndarray, I_raw: np.ndarray, return_index=False):
    V_float_idx = ei_parts_separator_index(I_raw)
    V_float = V_grid[V_float_idx]
    if return_index:
        return V_float, V_float_idx
    return V_float


def plasma_potential(V_grid: np.ndarray, dI_dV: np.ndarray, V_drop=20, **kwargs):
    dI_dV_cropped = dI_dV[V_grid < V_drop]
    pidx, pinfo = find_peaks(dI_dV_cropped, **kwargs)
    if len(pidx) == 0:
        print("No peaks found.")
        return None
    plasma_potential = V_grid[pidx[0]]
    derivative = dI_dV[pidx[0]]
    return plasma_potential, derivative


def electron_temperature(V_grid: np.ndarray, dlnI_dV: np.ndarray, 
                         V_plasma: float, V_float: float,
                         return_index=False, V_guard=0.1):
    array_mask_1 = V_grid < V_plasma - V_guard
    array_mask_2 = V_grid[array_mask_1] >= V_float
    dlnI_dV_cropped = dlnI_dV[array_mask_1]
    dlnI_dV_cropped = dlnI_dV_cropped[array_mask_2]
    min_idx = np.argmin(dlnI_dV_cropped)
    print(min_idx)
    # if len(max_idx) == 0:
    #     print("No peaks found.")
    #     return None
    plt.plot(dlnI_dV_cropped)
    electron_temperature = 1/dlnI_dV_cropped[min_idx]
    print(f'Beam temperature: {1/dlnI_dV_cropped[0]:.2f} eV')
    if return_index:
        return electron_temperature, min_idx
    return electron_temperature


def fit_electron_slope(V_measured: np.ndarray, I_measured: np.ndarray,I_sigma: np.ndarray,
                       V_float: float, 
                       Te: float, T_ebeam: float):
    """
    Fits the electron current slope with a two-temperature model.
    """
    def model_func(x, Ie, T1):
        return  np.log(Ie) + (x - V_float) / T1

    model = Model(model_func)
    params = model.make_params(Ie=1e-6, T1=Te)
    params['Ie'].min = 1e-8
    params['T1'].min = 0.1
    
    result = model.fit(np.log(I_measured), params, x=V_measured, weights=1/I_sigma)
    
    return result



def plasma_parameters(V_raw: np.ndarray, I_raw: np.ndarray, probe_area=1, **kwargs):

    V_float_idx = ei_parts_separator_index(I_raw)
    V_float = V_raw[V_float_idx]

    I_corrected, ion_fit = fit_ion_current(V_raw, I_raw, tolerance=5e-3, V_ion_cutoff=-8,
                                           fit_type='linear', return_fit=True, min_sample_number=4)
    
    # Prepare the data for the GP fit (using the electron current)
    el_mask = V_raw>=V_float
    ion_mask = V_raw<=V_float

    V_electron = V_raw[el_mask]
    I_electron_fit = I_corrected[el_mask]
    I_electron = I_raw[el_mask]

    V_ion = V_raw[ion_mask]
    I_ion = I_raw[ion_mask]
    I_ion_fit = ion_fit[ion_mask]

     # 2. Fit with Gaussian Process
    V_fit, I_fit, dI_dV, d2I_dV2, sigma, gp, V_scaler, I_scaler = fit_langmuir_gp(V_electron, I_electron_fit, n_grid=400)

    # 3. Process 2nd derivative and Find Plasma Potential
    dlnI_dV = dI_dV/I_fit # Logarithmic derivative of current wrt voltage (~q/kT)
    V_plasma, dI_dV_max = plasma_potential(V_fit[V_fit>0], dI_dV[V_fit>0], V_drop=5, prominence=1e-7, width=[1,150])
    Te = electron_temperature(V_fit, dlnI_dV, 
                              V_plasma=V_plasma, V_float=V_float,
                              V_guard=0)
    
    # Calculate fit
    tol = 0.5
    V_mask = np.where((V_fit < V_plasma + tol) & (V_fit > V_plasma - tol))
    fit_res = fit_electron_slope(V_fit[V_mask], I_fit[V_mask], 30*sigma[V_mask],
                                 V_float, Te, Te)
    fit_params = fit_res.params
    fit_line = fit_params['Ie'].value + (V_fit - V_float) / fit_params['T1'].value 
    # fit_res.plot()
    # print(fit_res.fit_report())

    # 4. Calculate Residuals
    # Predict on the original training points to get residuals
    V_train_scaled = V_scaler.transform(V_electron.reshape(-1, 1))
    I_fit_on_train_scaled = gp.predict(V_train_scaled, return_std=False)
    I_fit_on_train = I_scaler.inverse_transform(I_fit_on_train_scaled.reshape(-1, 1)).flatten()
    I_residuals = I_electron_fit - I_fit_on_train
    

    # Print extracted plasma parameters
    print(f'Floating potential: {V_float:.2f} V')
    print(f'Plasma potential: {V_plasma:.2f} V')
    print(f'Electron temperature: {Te:.2f} eV')
    print(f'Electron density: ???? cm^-3')

    output_dict = {'V_raw': V_raw, 'I_raw': I_raw,
                   'V_float': V_float, 'V_plasma': V_plasma,
                   'Te': Te,
                   'V_train': V_electron, 'I_train': I_electron_fit,
                   'V_electron': V_electron, 'I_electron': I_electron,
                   'I_ion': I_ion, 'I_ion_fit': I_ion_fit,
                   'V_fit': V_fit, 'I_fit': I_fit,
                   'dI_dV': dI_dV, 'd2I_dV2': d2I_dV2, 'dlnI_dV': dlnI_dV,
                   'dI_dV_max': dI_dV_max,
                   'sigma': sigma*30,
                   'V_ion': V_ion, 'I_ion': I_ion,
                   'I_ion_fit': I_ion_fit,
                   'residuals': I_residuals,
                   'fit_result': fit_res}
    return output_dict




#%%
if __name__ == "__main__":
    # 1. Generate or Load Data
    # (optional) Generate dummy data
    # V_raw, I_raw = generate_dummy_data()

    # Read real data 

    path = "G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-10-21\\data\\"
    date = "2025-10-21"

    data_numbers = [6,8,9,11,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33]

    electron_temperatures = []
    electron_temperatures_err = []
    plasma_potentials = []
    floating_potentials = []


    for data_number in data_numbers:
        file_name = f"data-lprobe-{date}_{data_number}.csv"
        data_path =  os.path.join(path, file_name)
        V_raw, I_raw, _ = np.genfromtxt(data_path, delimiter=',', skip_header=2, unpack=True)



        plasma_dict = plasma_parameters(V_raw, I_raw)

        current_best = plasma_dict['fit_result'].best_values['Ie']
        temp_electron_best = plasma_dict['fit_result'].best_values['T1']
        fit_line = current_best*np.exp((V_raw-plasma_dict['V_float'])/temp_electron_best)

        # 5. Create Plots
        # Create a figure with a GridSpec layout



        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 2], width_ratios=[2, 1])

        fig.suptitle(f'Langmuir Probe Analysis via Gaussian Process\
                     \nData Set #{data_number} ({date})', fontsize=16)

        # Add subplots to the grid
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[0, 1])  # Ion current plot

        # Hide x-tick labels for upper plots sharing the x-axis
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # --- Top Plot: I-V Curve ---
        # ax1.plot(V_fit, I_slope_fit, color='C3')
        ax1.plot(V_raw, fit_line, color='C3', linestyle='--', linewidth=2)
        ax1.plot(plasma_dict['V_electron'], plasma_dict['I_electron'], color='gray', alpha=0.5, label="Raw current")
        ax1.scatter(plasma_dict['V_train'], plasma_dict['I_train'], label='Measured Electron Current', 
                    c='k', marker='.', s=20, zorder=5, alpha=0.5)
        ax1.plot(plasma_dict['V_fit'], plasma_dict['I_fit'], 'r-', linewidth=2, label='GP Mean Fit')
        ax1.fill_between(plasma_dict['V_fit'],plasma_dict['I_fit'] - 1.96*plasma_dict['sigma'], plasma_dict['I_fit'] + 1.96*plasma_dict['sigma'], 
                         color='r', alpha=0.2, label='Uncertainty (type B)')
        ax1.axvline(plasma_dict['V_plasma'], color='g', linestyle='-.', linewidth=2, 
                    label=f'Plasma Potential $Vp = {plasma_dict['V_plasma']:.2f}$ V')
        ax1.set_yscale('log')
        ax1.set_ylabel('Electron Current (A)')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim([1e-6,5e-4])
        ax1.set_xlim([-6,15])
        ax1.legend()

        # --- Middle Plot: Residuals ---
        ax2.plot(plasma_dict['V_train'], plasma_dict['residuals'], 'k.', markersize=4)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1)
        ax2.set_ylabel('Residuals (A)')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # --- Bottom Plot: First and Second Derivatives ---
        ax3_twin = ax3.twinx() # Create a twin y-axis

        # Plot First Derivative on the left axis (ax3)
        p1, = ax3.plot(plasma_dict['V_fit'], 1/plasma_dict['dlnI_dV'], 'm-', linewidth=2, label='$T_e$')
        ax3.set_xlabel('Probe Voltage (V)')
        ax3.set_ylabel('Electron temperature (eV)', color='m')
        ax3.tick_params(axis='y', labelcolor='m')
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Plot Second Derivative on the right axis (ax3_twin)
        p2, = ax3_twin.plot(plasma_dict['V_fit'], plasma_dict['dI_dV'], 'b-', linewidth=2, label='$dI/dV$')
        ax3_twin.set_ylabel('$dI/dV$ (A/V)', color='b')
        ax3_twin.tick_params(axis='y', labelcolor='b')
        ax3_twin.set_ylim(bottom=-1e-5)

        # Mark the plasma potential
        ax3.axvline(plasma_dict['V_plasma'], color='g', linestyle='-.', linewidth=2, 
                    label=f'Plasma Potential $V_p = {plasma_dict['V_plasma']:.2f}$ V')
        # ax3.axhline(1/Te, color='g', linestyle='-.', linewidth=2, 
                    # label=f'Temp $T_e = {Te:.2f}$ eV')

        # Combine legends for ax3 and ax3_twin
        plots = [p1, p2]
        ax3.legend(plots, [p.get_label() for p in plots], loc='best')
        ax3.set_ylim(bottom=0, top=5)
        ax3_twin.set_ylim(bottom=-1e-6)


        # --- Ion current plot ---
        ax4.scatter(plasma_dict['V_ion'], plasma_dict['I_ion'], label='Measured Ion Current', 
                    c='k', marker='.', s=2)
        # ax4.scatter(V_ion, (-I_ion)**(4/3), label='Measured Ion Current', 
        #             c='k', marker='.', s=2)
        ax4.plot(plasma_dict['V_ion'],plasma_dict['I_ion_fit'], 'r-', linewidth=2, label='Ion sat. fit')
        # ax4.plot(V_ion, 1e-3*np.power((-V_ion + 5), -1.1)-1.2e-4)
        ax4.set_ylabel('Ion Current (A)')
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend()

        text_string = f'Electron temp: $T_e = {plasma_dict["fit_result"].params["T1"].value:.2f}\\pm{plasma_dict["fit_result"].params["T1"].stderr:.2f}$ eV\n\
Plasma potential: $V_p = {plasma_dict["V_plasma"]:.2f}$ V\n\
Floating potential: $V_f = {plasma_dict["V_float"]:.2f}$ V'
        fig.text(x=0.7, y=0.4, s=text_string, fontsize=16)

        # Adjust the layout

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig.savefig(f'data_lprobe_{date}_{data_number}.png')

        electron_temperatures.append(plasma_dict['fit_result'].params['T1'].value)
        electron_temperatures_err.append(plasma_dict['fit_result'].params['T1'].stderr)
        plasma_potentials.append(plasma_dict['V_plasma'])
        floating_potentials.append(plasma_dict['V_float'])

# %%

import pandas as pd

table_dict = {'#': data_numbers,
              'Electron temperature (eV)': electron_temperatures,
              'Electron temperature error (eV)': electron_temperatures_err,
              'Plasma potential (V)': plasma_potentials,
              'Floating potential (V)': floating_potentials}

table = pd.DataFrame(table_dict)
table
# %%
table.plot(x='#', y='Electron temperature (eV)', marker='o',
            xlabel='Measurement #', ylabel='$T_e$ (eV)')

# %%
# table.plot(x='#', y='Plasma potential (V)')
#

# %%
