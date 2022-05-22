import numpy as np
from scipy.signal import butter, lfilter
from typing import Optional

"""
At this step refined color signals are computed from raw signals by surpressing
noise and artefacts
"""

class TooLowCutoff(Exception):
    pass

class TooHighNumberPoints(Exception):
    pass

def butter_lowpass_filter(signal:np.array, 
                          cutoff:float, 
                          fs:Optional[float]=30.0, 
                          order:Optional[int]=5) -> np.array:
    """
    Butterworth lowpass filter
    """
    def butter_lowpass(cutoff:float, fs:float, order:int):
        if cutoff < 4:
            raise TooLowCutoff("Cutoff lower than the celling of heart rate band width")
        nyq = 0.5*fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    signal_filtered = lfilter(b, a, signal)
    return signal_filtered

def moving_average(signal:np.array,
                   n_points:int) -> np.array:
    """
    Calculates the moving average of a signal
    """
    if n_points > 12:
        raise TooHighNumberPoints("Risk of removing pulse rate information.")
    signal_averaged = np.convolve(signal, np.ones(n_points), 'valid') / n_points
    #Just to avoid changing signal dimensions
    signal[:len(signal_averaged)] = signal_averaged
    return signal
