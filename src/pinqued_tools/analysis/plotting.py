'''
Author: Mykhailo Vorobiov
This file contains functions that modify and predefine 
certain types of plotting styles.
'''

import matplotlib as mpl
import matplotlib.pyplot as plt


def _mpl_prod_style():
    '''
    Sets production style plots rendering
    no LaTeX rendering allowed by default.
    '''
    _mpl_nature_style()
    plt.rcParams.update({
        'figure.figsize': [8, 7]
    })

def _mpl_nature_style(use_tex=False):
    '''
    Plots look like rendered with LaTeX 
    regardless of actual rendering engine.
    
    :param use_tex: Enables (True) or disables (False) LaTeX rendering 
    '''
    plt.rcParams.update({
        'text.usetex': use_tex,
        'figure.figsize': [3.38, 3.38],
        'lines.linewidth': 1,
        'lines.linewidth': 1,
        'xtick.top': True,
        'ytick.right': True, 
        'xtick.direction': 'in', 
        'ytick.direction': 'in',
        'axes.xmargin': 0.01,
        'axes.ymargin': 0.01,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'axes.labelsize': 9,
        'font.family': 'serif', 
        'font.serif': 'Arial',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.4,
        'legend.frameon': False,
        'figure.dpi': 150,  # Sets display DPI
        'savefig.dpi': 300, # Sets saved image DPI
    })

def _mpl_tex_style(use_tex=False):
    '''
    Plots style similar to Nature papers
    
    :param use_tex: Enables (True) or disables (False) LaTeX rendering 
    '''
    plt.rcParams.update({
        'text.usetex': use_tex,
        'figure.figsize': [3.38, 3.38],
        'lines.linewidth': 1,
        'xtick.top': True,
        'ytick.right': True, 
        'xtick.direction': 'in', 
        'ytick.direction': 'in',
        'axes.xmargin': 0.01,
        'axes.ymargin': 0.01,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'axes.labelsize': 14,
        'font.family': 'serif', 
        'font.serif': 'cmr10',
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'axes.unicode_minus': False,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.4,
        'figure.dpi': 150,  # Sets display DPI
        'savefig.dpi': 300, # Sets saved image DPI
    })


def set_mpl_style(style='tex', use_tex=False):
    '''
    Select predefined matplotlib plotting style
    from suggested. 
    
    :param style: Custom style alias name
    :param use_tex: Render with LaTeX (True) or no (False).
    '''
    if style=='tex': # TeX like style
        _mpl_tex_style(use_tex)
    elif style=='nature': # Nature like style
        _mpl_nature_style(use_tex)
    elif style=='prod': # Production style for daily plotting
        _mpl_prod_style()

def lprobe_plot(data_dict: dict, 
                ylim, xlim,
                figsize=(12, 9)
                ):
    """
    Function plots fitting procedure for 
    cylindrical Langmuir probe data
    """

    # 1. Unpack data from input dictionary
    # ------------------------------------
    data_number = data_dict['data_number']
    date = data_dict['date']
    V_raw_samples = data_dict['V_raw']
    I_raw_samples = data_dict['I_raw']
    V_electron = data_dict['V_train']
    I_electron = data_dict['I_train']
    V_electron_fit = data_dict['V_fit']
    I_electron_fit = data_dict['I_fit']
    dI_dV = data_dict['dI_dV']
    # d2I_dV2 = data_dict['d2I_dV2']
    sigma = data_dict['sigma']
    V_plasma = data_dict['V_plasma']
    I_ion_fit = data_dict['I_ion_fit']
    V_ion = data_dict['V_ion']
    I_ion = data_dict['I_ion']
    residuals = data_dict['residuals']

    dlnI_dV = dI_dV / I_electron_fit
    temp_electron = 1 / dlnI_dV

    # 2. Create Plots
    # ------------------------------------
    # Create a figure with a GridSpec layout
    fig = plt.figure(figsize=figsize)
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
    # ax1.plot(V_electron, fit_line, color='C3', linestyle='--', linewidth=2)
    ax1.plot(V_electron, I_electron, color='gray', alpha=0.5, label="Raw current")
    ax1.scatter(V_electron_fit, I_electron_fit, label='Measured Electron Current', 
                c='k', marker='.', s=20, zorder=5, alpha=0.5)
    ax1.plot(V_electron_fit, I_electron_fit, 'r-', linewidth=2, label='GP Mean Fit')
    ax1.fill_between(V_electron_fit, I_electron_fit - 1.96*sigma, I_electron_fit + 1.96*sigma, 
                     color='r', alpha=0.2, label='Uncertainty (type B)')
    ax1.axvline(V_plasma, color='g', linestyle='-.', linewidth=2, 
                label=f'Plasma Potential $Vp = {V_plasma}$ V')
    ax1.set_yscale('log')
    ax1.set_ylabel('Electron Current (A)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax1.legend()
    # --- Middle Plot: Residuals ---
    ax2.plot(V_electron, residuals, 'k.', markersize=4)
    ax2.axhline(0, color='r', linestyle='--', linewidth=1)
    ax2.set_ylabel('Residuals (A)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    # --- Bottom Plot: First and Second Derivatives ---
    ax3_twin = ax3.twinx() # Create a twin y-axis
    # Plot First Derivative on the left axis (ax3)
    p1, = ax3.plot(V_electron_fit, temp_electron, 'm-', linewidth=2, label='$T_e$')
    ax3.set_xlabel('Probe Voltage (V)')
    ax3.set_ylabel('Electron temperature (eV)', color='m')
    ax3.tick_params(axis='y', labelcolor='m')
    ax3.grid(True, linestyle='--', alpha=0.6)
    # Plot Second Derivative on the right axis (ax3_twin)
    p2, = ax3_twin.plot(V_electron_fit, dI_dV, 'b-', linewidth=2, label='$dI/dV$')
    ax3_twin.set_ylabel('$dI/dV$ (A/V)', color='b')
    ax3_twin.tick_params(axis='y', labelcolor='b')
    ax3_twin.set_ylim(bottom=-1e-5)
    # Mark the plasma potential
    ax3.axvline(V_plasma, color='g', linestyle='-.', linewidth=2, 
                label=f'Plasma Potential $V_p = {V_plasma:.2f}$ V')
    # ax3.axhline(1/Te, color='g', linestyle='-.', linewidth=2, 
                # label=f'Temp $T_e = {Te:.2f}$ eV')
    # Combine legends for ax3 and ax3_twin
    plots = [p1, p2]
    ax3.legend(plots, [p.get_label() for p in plots], loc='best')
    ax3.set_ylim(bottom=0, top=5)
    ax3_twin.set_ylim(bottom=-1e-6)
    # --- Ion current plot ---
    ax4.scatter(V_ion, I_ion, label='Measured Ion Current', 
                c='k', marker='.', s=2)
    # ax4.scatter(V_ion, (-I_ion)**(4/3), label='Measured Ion Current', 
    #             c='k', marker='.', s=2)
    ax4.plot(V_ion, I_ion_fit, 'r-', linewidth=2, label='Ion sat. fit')
    # ax4.plot(V_ion, 1e-3*np.power((-V_ion + 5), -1.1)-1.2e-4)
    ax4.set_ylabel('Ion Current (A)')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()
#     text_string = f'Electron temp: $T_e = {plasma_dict["fit_result"].params["T1"].value:.2f}\\pm{plasma_dict["fit_result"].params["T1"].stderr:.2f}$ eV\n\
# Plasma potential: $V_p = {plasma_dict["V_plasma"]:.2f}$ V\n\
# Floating potential: $V_f = {plasma_dict["V_float"]:.2f}$ V'
    # fig.text(x=0.7, y=0.4, s=text_string, fontsize=16)
    # Adjust the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    return fig




