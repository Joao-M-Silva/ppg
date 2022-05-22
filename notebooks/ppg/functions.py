from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

def power_spectrum(signal:np.array,
                   plot=True,
                   **kwargs) -> pd.DataFrame:
    if "df_power" in kwargs:
        df_power = kwargs["df_power"]
    else:
        #Calculate the fourier transform of the signal
        fourier_transform = np.fft.rfft(signal)
        sampling_rate = 30.0
        #Compute the power spectrum
        power_spectrum = np.square(np.abs(fourier_transform))
        frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
        df_power = pd.DataFrame({'frequency':frequency, 
                                'power_spectrum':power_spectrum})
        df_power = df_power.query('0.4 <= frequency <= 6').reset_index(drop=True)
    if plot:
        plt.figure(figsize=(20,10))
        plt.grid()
        plt.plot(
            df_power['frequency'], 
            df_power['power_spectrum']
        )
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.plot(
            [0.65, 0.65], 
            [0, max(df_power['power_spectrum'])], 
            'k-',
        )
        plt.plot(
            [4, 4], 
            [0, max(df_power['power_spectrum'])], 
            'k-',
        )
        plt.show()
    return df_power

def _select_source(ica_sources:List[np.array]) -> Tuple[np.array, int]:
    """
    Choose the component with the most prominent
    peak in the heart rate bandwidth
    """
    def map_function(signal:np.array) -> float:
        df_power = power_spectrum(signal, plot=False)
        return df_power['power_spectrum'].max()

    maximum_peaks = list(map(map_function, ica_sources))
    max_peak = max(maximum_peaks)
    max_peak_index = maximum_peaks.index(max_peak)
    print('Max Peak Source: ', max_peak_index)
    return ica_sources[max_peak_index], max_peak_index

def ppg_by_ica_sources(df:pd.DataFrame,) -> pd.DataFrame:
    """
    Independent Component Analysis to extract three sources.
    One of them will be considered the PPG.
    """
    X = df[['B_F', 'G_F', 'R_F']].values
    ica = FastICA(n_components=3)
    #Source Signals
    S = ica.fit_transform(X)
    print('Mixing Matrix')
    print(ica.mixing_)
    df['ICA_Source_0'] = S[:,0]
    df['ICA_Source_1'] = S[:,1]
    df['ICA_Source_2'] = S[:,2]
    ica_sources = [
        df['ICA_Source_0'].values,
        df['ICA_Source_1'].values,
        df['ICA_Source_2'].values,
    ]
    df['ICA_Best_Source'], _ = _select_source(ica_sources)
    return df

def ppg_by_green_signal(df:pd.DataFrame,) -> pd.DataFrame:
    """
    The PPG is simply the green signal.
    """
    df['Green_Source'] = df['G_F'] 
    return df

def ppg_by_grd(df:pd.DataFrame,) -> pd.DataFrame:
    """
    The PPG is the difference between the green and red channels
    """
    df['GRD_Source'] = df['G_F'] - df['R_F']
    return df

def select_ppg(df:pd.DataFrame) -> str:
    """
    Selection of the PPG signal with the highest peak in the
    heart rate bandwidth
    """
    possible_sources = [(df[col].values, col)
                        for col in df.columns 
                        if "Source" in col]
    _, index =  _select_source([source[0] for source in possible_sources])
    source_col = possible_sources[index][1]
    return source_col


