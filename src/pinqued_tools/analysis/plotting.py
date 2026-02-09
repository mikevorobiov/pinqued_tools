'''
Author: Mykhailo Vorobiov
This file contains functions that modify and predefine 
certain types of plotting styles.
'''

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_mpl_style(style='tex'):
    ...


def langmuir_probe_plot(data_dict: dict, figsize=(12, 9)):
    """
    Functions plots fitting procedure for 
    cylindrical Langmuir probe data
    """

    # 1. Unpack data from input dictionary
    # ------------------------------------
    data_number = data_dict['data_number']
    date = data_dict['date']
    V_electron = data_dict['V_raw']
    I_electron = data_dict['I_raw']
    V_train = data_dict['V_train']
    I_train = data_dict['I_train']
    V_fit = data_dict['V_fit']
    I_fit = data_dict['I_fit']
    dI_dV = data_dict['dI_dV']
    d2I_dV2 = data_dict['d2I_dV2']
    sigma = data_dict['sigma']
    V_plasma = data_dict['V_plasma']
    I_ion_fit = data_dict['I_ion_fit']
    V_ion = data_dict['V_ion']
    I_ion = data_dict['I_ion']
    residuals = data_dict['residuals']

    # 2. Create Plots
    # ------------------------------------
    # Create a figure with a GridSpec layout
    fig = plt.figure(figsize=(12, 9))
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
    ax1.plot(V_electron, I_electron, color='gray', alpha=0.5, label="Raw current")
    ax1.scatter(V_train, I_train, label='Measured Electron Current', 
                c='k', marker='.', s=20, zorder=5, alpha=0.5)
    ax1.plot(V_fit, I_fit, 'r-', linewidth=2, label='GP Mean Fit')
    ax1.fill_between(V_fit, I_fit - 1.96*sigma, I_fit + 1.96*sigma, 
                     color='r', alpha=0.2, label='95% Confidence Interval')
    ax1.axvline(V_plasma, color='g', linestyle='-.', linewidth=2, 
                label=f'Plasma Potential $Vp = {V_plasma:.2f}$ V')
    ax1.set_yscale('log')
    ax1.set_ylabel('Electron Current (A)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Middle Plot: Residuals ---
    ax2.plot(V_train, residuals, 'k.', markersize=4)
    ax2.axhline(0, color='r', linestyle='--', linewidth=1)
    ax2.set_ylabel('Residuals (A)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Bottom Plot: First and Second Derivatives ---
    ax3_twin = ax3.twinx() # Create a twin y-axis

    # Plot First Derivative on the left axis (ax3)
    p1, = ax3.plot(V_fit, dI_dV/I_fit, 'm-', linewidth=2, label='$dI/dV$')
    ax3.set_xlabel('Probe Voltage (V)')
    ax3.set_ylabel('$dI/dV$ (A/V)', color='m')
    ax3.tick_params(axis='y', labelcolor='m')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Plot Second Derivative on the right axis (ax3_twin)
    p2, = ax3_twin.plot(V_fit, d2I_dV2_processed, 'b-', linewidth=2, label='Processed $d^2I/dV^2$')
    p3, = ax3_twin.plot(V_fit, d2I_dV2, color='c', ls='--', alpha=0.5, label='Original $d^2I/dV^2$')
    ax3_twin.set_ylabel('$d^2I/dV^2$ (A/V$^2$)', color='b')
    ax3_twin.tick_params(axis='y', labelcolor='b')
    ax3_twin.set_ylim(bottom=-1e-5)

    # Mark the plasma potential
    ax3.axvline(V_plasma, color='g', linestyle='-.', linewidth=2, 
                label=f'Plasma Potential $V_p = {V_plasma:.2f}$ V')
    
    # Combine legends for ax3 and ax3_twin
    plots = [p1, p2, p3]
    ax3.legend(plots, [p.get_label() for p in plots], loc='best')
    # ax3.set_ylim(bottom=-1e-6)
    ax3_twin.set_ylim(bottom=-1e-6)


    # --- Ion current plot ---
    ax4.scatter(V_ion, I_ion, label='Measured Ion Current', 
                c='k', marker='.', s=2)
    # ax4.scatter(V_ion, (-I_ion)**(4/3), label='Measured Ion Current', 
    #             c='k', marker='.', s=2)
    ax4.plot(V_ion, I_ion_fit, 'r-', linewidth=2, label='Fit')
    # ax4.plot(V_ion, 1e-3*np.power((-V_ion + 5), -1.1)-1.2e-4)
    ax4.set_ylabel('Ion Current (A)')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()

    # Adjust the layout

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()



