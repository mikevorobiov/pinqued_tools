'''
Classses for spectral data storage and processing

Author: Mykhailo Vorobiov
'''
#%%
import numpy as np
from datetime import datetime

from dataclasses import dataclass, field, fields, asdict
from typing import Dict
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import copy



import pybaselines as pbl
from scipy.signal import find_peaks
# -------------------- Axes classes -------------------------
@dataclass 
class BaseAxes(ABC):
    """Abstract base class for all Axes dataclasses."""
    f: NDArray # Frequency coordinate

@dataclass
class Axes0D(BaseAxes):
    units: Dict[str, str] = field(default_factory=lambda: {'f': 'MHz'})

@dataclass
class Axes1D(BaseAxes):
    x: NDArray # Spatial coordinate
    units: Dict[str, str] = field(default_factory=lambda: {'x': 'mm', 'f': 'MHz'})

@dataclass
class Axes2D(BaseAxes):
    x: NDArray # Spatial coordinate x
    y: NDArray # Spatial coordinate y
    units: Dict[str, str] = field(default_factory=lambda: {'x': 'mm', 'y': 'mm', 'f': 'MHz'})

# ----------------- Spectral Data classes -----------------
@dataclass
class SpectralData():
    '''
    Class stores any set of spectral data
    '''
    signal: NDArray
    axes: BaseAxes
    signal_err: NDArray|None = None
    units: Dict[str, str] = field(default_factory=lambda: {'signal': '%', 'signal_err': '%'})
    metadata: dict|None = None

    def __post_init__(self):
        self.metadata = self.metadata or {'signal dims': f'{self.signal.shape}'}

    def __repr__(self) -> str:
        pstring = f"<{self.__class__.__name__} at {hex(id(self))}>\n"
        total_mem = 0

        pstring += "--- Data Arrays ---\n"
        
        mem_signal = self.signal.nbytes
        total_mem += mem_signal
        pstring += f"signal: shape={self.signal.shape}, unit={self.units.get('signal', 'N/A')}, mem={mem_signal / 1024**2:.3f} MB\n"

        if self.signal_err is not None:
            mem_err = self.signal_err.nbytes
            total_mem += mem_err
            pstring += f"signal_err: shape={self.signal_err.shape}, unit={self.units.get('signal_err', 'N/A')}, mem={mem_err / 1024**2:.3f} MB\n"

        pstring += "\n--- Axes ---\n"
        axes_fields = fields(self.axes)
        for f in axes_fields:
            if f.name == 'units':
                continue
            ax_val = getattr(self.axes, f.name)
            if isinstance(ax_val, np.ndarray):
                mem_ax = ax_val.nbytes
                total_mem += mem_ax
                pstring += f"{f.name}: shape={ax_val.shape}, unit={self.axes.units.get(f.name, 'N/A')}, mem={mem_ax / 1024**2:.3f} MB\n"

        if self.metadata:
            pstring += "\n--- Metadata ---\n"
            for k, v in self.metadata.items():
                pstring += f"{k}: {v}\n"

        pstring += "-------------------\n"
        pstring += f"Total memory: {total_mem / 1024**2:.3f} MB\n"
        return pstring
    
    def add_metadata(self, key: str, value: str):
        '''
        Adds metadata entry to class
        TODO: This method must be reimplemented and refactored
                currently there is problem that it does not align with the dates
        '''
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value


# ------------------ Classes for processing Spectral Data -------------------
class SpectralDataProcessor():
    '''
    Class for processing spectral data.
    Spectral data can be
        0d - signle spectrum
        1d - 1d spatial map (f, x)
        2d - 2d spatial map (f, x, y)
    '''

    AX_ALIAS = ['f', 'x', 'y']

    def __init__(self, data: SpectralData):
        self._data = copy.deepcopy(data)
        self._data.signal = self._data.signal.astype(np.float32)
        self._data.metadata = {'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' Processing...'}

    @property
    def data(self,) -> SpectralData:
        return self._data        
    
    def preprocess(self, background_image: SpectralData|None = None):
        '''
        Turns raw spectral image from the camera into the relative intensity dip signal.
        Addiitonaly the function calculates signal error.
        '''
        # Extract very first image away from the intensity dips
        if background_image is not None:
            self._data.signal -= background_image.signal
            singnal_err_squared = self._data.signal + background_image.signal
            err_signal_ratio2 = singnal_err_squared / self._data.signal**2
        else:
            err_signal_ratio2 = 1.0 / self._data.signal
        
        # Calculate relative intensity dip signal
        bg_image = self._data.signal[0]
        relative_signal = self._data.signal / bg_image

        # Calculate signal error
        # σ_s = I/I0 * √((σ_I/I)^2 + (σ_I0/I0)^2)
        err_bg_ratio2 = 1.0 / bg_image
        relative_signal_err = relative_signal * np.sqrt(err_signal_ratio2 + err_bg_ratio2)
        
        # Update data
        self._data.signal = 1.0 - relative_signal
        self._data.signal_err = relative_signal_err

    def remove_fmean(self, samples = range(10)) -> None:
        '''
        Removes mean value of the selected samples calculated along the frequency axis.
        '''
        # 1. Extract signal samples to be averaged
        samples = np.take(self._data.signal, samples, axis=0)
        
        # 2. Calculate average and subtract from the signal array
        mean = np.mean(samples, axis=0, keepdims=True)
        mean_full = np.repeat(mean, self._data.signal.shape[0], axis=0)
        self._data.signal -= mean_full

        # 3. Add message to metadata that removal of mean has been performed
        if self._data.metadata is not None:
            self._data.metadata['fmean'] = f'Subtracted mean of f {samples.shape[0]}'
        else:
            self._data.metadata = {'fmean': f'Subtracted mean of f {samples.shape[0]}'}

    def _bin(self, array, px_per_bin: int) -> NDArray:
        '''
        Private fucntion to bin a 1d signal array
        '''
        n = array.shape[0]
        arr_reshaped = array.reshape(n // px_per_bin, px_per_bin, *array.shape[1:])
        return np.mean(arr_reshaped, axis=1)
    
    def _bin_error(self, array, px_per_bin: int) -> NDArray:
        '''
        Private fucntion to calcualte errors for binned 1d signal array
        '''
        # Propagate error when the signal samples are binned
        n = array.shape[0]
        arr_reshaped = array.reshape(n // px_per_bin, px_per_bin, *array.shape[1:])
        # Calculate error of the binned signal (standard error of the mean) as
        #  σ = 1/N * √ { ∑σ^2 } 
        signal_error = np.sqrt(np.sum(arr_reshaped**2, axis=1)) / px_per_bin
        return signal_error

    def bin(self,
            px_per_bin: int = 2,
            axis: int = 0) -> None:
        '''
        Performs spatial binning of the spectral data.
        So far limited to axis arrays with dimenstions propto powers of 2
        Axes idices:
            0 - f
            1 - x
            2 - y
        '''
        if self._data.signal.shape[axis] % px_per_bin != 0:
            print(f'Cannot perfrom binning along axis {self.AX_ALIAS[axis]} with {px_per_bin} pixels per bin.')
            print(f'Make sure axis length is dvisible by {px_per_bin}.')
            raise ValueError

        # 1. Bin signal and signal error if present
        self._data.signal = np.apply_along_axis(self._bin, axis, 
                                                self._data.signal, 
                                                px_per_bin=px_per_bin)
        if self._data.signal_err is not None:
            self._data.signal_err = np.apply_along_axis(self._bin_error, axis, 
                                                    self._data.signal_err, 
                                                    px_per_bin=px_per_bin)
        # 2. Adjust axes accordingly (reduce number of samples)
        ax_dict = asdict(self._data.axes)
        for i,k in enumerate(ax_dict.keys()):
            if i==axis:
                print(k)
                ax_dict[k] = self._bin(ax_dict[k], px_per_bin=px_per_bin)
                # 3. Add message about binning to metadata
                if self._data.metadata is not None:
                    self._data.metadata['binning'] = f'Binning applied along axis {self.AX_ALIAS[axis]} with {px_per_bin} pixels per bin'
                else:
                    self._data.metadata = {'binning': f'Binning applied along axis {self.AX_ALIAS[axis]} with {px_per_bin} pixels per bin'}
        
        # 4. Update axes
        if isinstance(self._data.axes, Axes0D):
            self._data.axes = Axes0D(**ax_dict)
        elif isinstance(self._data.axes, Axes1D):
            self._data.axes = Axes1D(**ax_dict)
        elif isinstance(self._data.axes, Axes2D):
            self._data.axes = Axes2D(**ax_dict) 

    def remove_baseline(self, **kwargs):
        pass

    def denoise(self):
        pass


class Calibration():
    '''
    Class for processing of calibration trace data.
    In the case of Ryderbg nD states it fits two peaks with
    known separations and determines time-to-frequency 
    conversion factor
    '''
    def __init__(self,
                 time: NDArray|None = None,
                 signal: NDArray|None = None, 
                 trigger_signal: NDArray|None = None,
                 d_peaks_separation: float = 1.0):
        
        self._d_peaks_separation = d_peaks_separation
        
        self._time = time
        self._signal = signal
        self._trigger_signal = trigger_signal

        self._background = None
        self._conversion_factor = None
        self._main_peak_pos = None

    def set_data(self, 
                 time: NDArray, 
                 signal: NDArray,
                 trigger_signal: NDArray|None = None
                 ):
        self._time = time
        self._signal = signal
        self._trigger_signal = trigger_signal

    def correct_trigger_event(self, trigger_threshold: float = 2.5):
        if self._trigger_signal is None or self._time is None:
            raise ValueError("Trigger signal and time data must be set before correcting.")
        # Implementation for correcting trigger event by 
        # finding the first index where the trigger signal exceeds the threshold
        trigger_indices = np.where(self._trigger_signal > trigger_threshold)[0]
        if len(trigger_indices) == 0:
            raise ValueError("No trigger found in the trigger signal.")
        trigger_index = trigger_indices[0]
        time_shift = self._time[trigger_index]
        self._time = self._time - time_shift    
        
    def remove_baseline_fastchom(self,
                        fixed_threshold: float = 1.5e-4,
                        half_window: int = 2):
        if self._time is None or self._signal is None:
            raise ValueError("Time and signal data must be set before removing baseline.")
        baseline_fitter = pbl.Baseline(x_data = self._time)

        bg_fit, params = baseline_fitter.fastchrom(self._signal, 
                                                   half_window, 
                                                   threshold=fixed_threshold)
        yd = self._signal - bg_fit

        self._background = bg_fit
        self._signal = yd

    def time_to_freq(self, **find_peaks_kwargs):
        if self._signal is None or self._time is None:
            raise ValueError("Signal and time data must be available for time-to-frequency conversion.")
        # 1. Find peaks in the clean signal
        ppos, ppar = find_peaks(self._signal, **find_peaks_kwargs)
        if len(ppos) < 2:
            raise ValueError("At least two peaks must be found for time-to-frequency conversion.")
        # 2. Calculate time separation between the peaks
        t_peaks_separation = self._time[ppos[1]] - self._time[ppos[0]]
        # 3. Calculate time-to-frequency conversion factor
        conversion_factor = self._d_peaks_separation / t_peaks_separation
        self._conversion_factor = conversion_factor
        self._main_peak_pos = ppos[1]
        return conversion_factor, ppos[0], ppos[1]

    @property
    def time(self):
        if self._time is None:
            raise ValueError("Time data has not been shifted yet. Call match_trigger() first.")
        return self._time

    @property
    def background(self):
        if self._background is None:
            raise ValueError("Baseline has not been removed yet. Call remove_baseline() first.")
        return self._background
    
    @property
    def signal(self):
        if self._signal is None:
            raise ValueError("Clean signal has not been calculated yet. Call remove_baseline() first.")
        return self._signal
    
    @property
    def freq(self):
        if self._time is None or self._conversion_factor is None:
            raise ValueError("Time data and conversion factor must be set for frequency calculation.")
        return (self._time - self._time[self._main_peak_pos]) * self._conversion_factor

#%%
if __name__=='__main__':
    # ------------------  Usage example ----------------
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from datetime import datetime
    from pinqued_tools.analysis.plotting import set_mpl_style
    set_mpl_style()

    from pinqued_tools.data.io import SpectralDataH5Handler, DataManager
    h5_handler = SpectralDataH5Handler()

    dm = DataManager(base_path='./data')


    # 1. Create mock data
    # single spectrum (f, signal)
    sdata0 = SpectralData(signal=10 + np.random.poisson(lam=100, size=256)*10.1,
                          signal_err=np.sqrt(100 + np.random.poisson(lam=100, size=256)*10.1),
                         axes=Axes0D(f=np.linspace(0,10,256)),
                         metadata={'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    # 1d spectral map (f, x, signal)
    sdata = SpectralData(signal=100 + np.random.poisson(lam=100, size=(256,256))*10.1,
                         axes=Axes1D(x=np.linspace(0,10,256), 
                                     f=np.linspace(0,100,256)))

    # 2. Process data using processor class
    sproc = SpectralDataProcessor(sdata)
    sproc.preprocess() # convert to relative dip intensity and calculate resulting error
    sproc.remove_fmean()
    sproc.bin(px_per_bin=16, axis=1)

    dm.save(sproc.data, 'spec_data_processed', ext='.h5')


    # 3. Print info about data containers
    print(sdata0)
    print(sproc.data)

    # Plot 0D (single spectrum)
    fig, ax = plt.subplots()
    ax.errorbar(x=sdata0.axes.f, y=sdata0.signal, yerr=sdata0.signal_err, 
                linestyle='None', marker='o', markersize=2, alpha=0.6)
    
    # Plot 1D (spatial-frequnecy map) raw data
    fig, ax = plt.subplots()
    ax.pcolormesh(sdata.axes.x, sdata.axes.f, sdata.signal, cmap='jet')

    # Plot 1D (spectrum with error) preprocessed, i.e. converted to relative dip intensity
    fig, ax = plt.subplots()
    ax.errorbar(x=sproc.data.axes.f, y=sproc.data.signal[:,10], yerr=sproc.data.signal_err[:,10],
                linestyle='None', marker='o', markersize=2, alpha=0.6)

# %%
