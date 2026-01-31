'''
Docstring for pinqued_tools.analysis.lprobe
'''
from scipy.interpolate import UnivariateSpline


class lprobe():
    '''
    Class for processing cylindrical Langmuir probe I-V sweeps 
    '''
    def __init__(self, 
                 probe_diameter = float,
                 probe_length = float,
                 iv_sweep = dict|None) -> None:
        self.probe_diameter = probe_diameter # Probe diameter in mm
        self.probe_length = probe_length # Probe length in mm
        self.iv_sweep = iv_sweep # I-V sweep of the probe in Amp and Volts

        self.iv_sweep_spline = None
        self.iv_sweep_grad = None
        self.iv_sweep_grad2 = None

    def set_iv_sweep(self, iv_sweep: dict):
        '''
        Sets I-V sweep data in the form of a dictionary
        {'V':   [float], 'I':   [float]} 
        '''
        self.iv_sweep = iv_sweep

    def _spline_smooth_grad(self, smooth=0.1):
        '''
        Private function calculates spline for IV sweep
        returns spline object and its 1st and 2nd 
        derivatives as spline objects too.

        :param smooth: Spline smoothing parameter.
        '''
        if self.iv_sweep is None:
            print('ERROR: L-probe IV curve has not been set. \
                  Use `set_iv_sweep` method.')
            return None
        x, y = self.iv_sweep.items()
        spline = UnivariateSpline(x,y,smooth)

        self.iv_sweep_spline = spline
        self.iv_sweep_grad = spline.derivative(n=1) 
        self.iv_sweep_grad2 = spline.derivative(n=2)
    
    def _spline_resample(self, x):
        smooth_sweep = self.iv_sweep_spline(x)
        smooth_grad = self.iv_sweep_grad(x)
        smooth_grad2 = self.iv_sweep_grad2(x)
        return smooth_sweep, smooth_grad, smooth_grad2


    def get_palsma_params(self):
        '''
        If an I-V sweep data of the Langmuir probe has been
        set with the `set_iv_sweep` method, then this function
        calculates plasma paramteres according to 
        F. F. Chen's "Lecture notes on Langmuir probe diagnostics" 
        https://www.seas.ucla.edu/~ffchen/Publs/Chen210R.pdf
        '''
        pass