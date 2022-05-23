import numpy as np
import pandas as pd
from .functions import power_spectrum
from typing import Optional, Tuple, List
import scipy.stats
import seaborn as sns
import plotly.graph_objects as go
from enum import Enum, auto

class FPSError(Exception):
    pass

class TimeUnit(Enum):
    SECONDS = auto()
    MILISECONDS = auto()

class PPG:
    def __init__(self, 
                 df_ppg:pd.DataFrame, 
                 source_name:str,):
        self.ppg = PPG._process_ppg(df_ppg)
        self.source_name = source_name
        self.power_spectum = power_spectrum(
            signal=df_ppg[self.source_name].values, 
            plot=False
            )
        self.predictions = None
    
    def __str__(self):
        return f"PPG derived from {self.source_name}."

    @staticmethod
    def _process_ppg(ppg:pd.DataFrame) -> pd.DataFrame:
        #Only consider data after 10 seconds of recording
        ppg = ppg.query('10.0 < Time_s < 174.0').reset_index(drop=True)
        delta_ms = int(ppg.loc[1, 'Time_ms'] - ppg.loc[0, 'Time_ms'])
        if delta_ms not in [33, 34]:
            raise FPSError("Frequency different than 30 Hz.")
        return ppg
    
    @staticmethod
    def _process_reference(reference:pd.DataFrame) -> pd.DataFrame:
        reference = reference.rename(columns={"Time[ms]":"Time_ms"})
        reference['Time_s'] = reference['Time_ms']*0.001
        #Only consider data after 10 seconds of recording
        reference = reference.query('10.0 < Time_s < 174.0').reset_index(drop=True)
        delta_ms = int(reference.loc[1, 'Time_ms'] - reference.loc[0, 'Time_ms'])
        if delta_ms not in [33, 34]:
            raise FPSError("Frequency different than 30 Hz.")
        return reference

    def display_power_spectum(self, source:Optional[str]=None):
        if source:
            source_name = source
        else:
            source_name = self.source_name
        power_spectrum(
            signal=self.ppg[source_name], 
            plot=True, 
            )

    def display_signals(
            self,
            columns=List[str],
            time_range:Optional[Tuple[float, float]]=None,
            time_unit:Optional[TimeUnit]=TimeUnit.SECONDS,
            width:Optional[float]=0.5,
        ):
        fig = go.Figure()
        if time_unit == TimeUnit.SECONDS:
            time_col = 'Time_s'
        elif time_unit == TimeUnit.MILISECONDS:
            time_col = 'Time_ms'

        if time_range is not None:
            min_time = time_range[0]
            max_time = time_range[1]
            df = self.ppg.query(f"@min_time <= `{time_col}` <= @max_time")
        else:
            df = self.ppg
        for col in columns:
            fig.add_trace(
                    go.Scatter(
                    x=df[time_col],
                    y=df[col],
                    name=col,
                    line=dict(width=width)
                )
            )
        return fig

    @staticmethod
    def _folds_indices(array:np.array, n_folds:int) -> np.array:
        folds_indices = []
        for fold_index, fold in enumerate(np.array_split(array, n_folds)):
            folds_indices.append(np.ones(len(fold))*fold_index)
        return np.concatenate(folds_indices)
        
    def _estimate_heart_rate(self, n_folds:Optional[int]=10) -> pd.DataFrame:
        #Split the data into folds
        self.ppg['Folds'] = PPG._folds_indices(self.ppg['Time_ms'], n_folds)
        heart_rate_dict = {
            'Folds':[],
            'HR[Hz]':[],
            'HR[bpm]':[],
        }
        #Estimate the heart rate for each fold
        for fold_index in self.ppg['Folds'].unique():
            ppg_fold = self.ppg.query("Folds == @fold_index")
            #Calculate the power spectrum
            fold_power_spectum = power_spectrum(
                    signal=ppg_fold[self.source_name].values, 
                    plot=False
                )
            #Get the frequency correspondent to the heighest peak
            max_peak_index = fold_power_spectum['power_spectrum'].idxmax()
            heart_rate_hz = fold_power_spectum.iloc[max_peak_index]['frequency']
            heart_rate_bpm = heart_rate_hz*60

            heart_rate_dict['Folds'].append(fold_index)
            heart_rate_dict['HR[Hz]'].append(heart_rate_hz)
            heart_rate_dict['HR[bpm]'].append(heart_rate_bpm)

        return pd.DataFrame(heart_rate_dict)

    def results(self, reference:pd.DataFrame, n_folds:int) -> pd.DataFrame:
        reference = PPG._process_reference(reference)
        assert self.ppg.shape[0] == reference.shape[0]
        reference['Folds'] = PPG._folds_indices(reference['Time_ms'], n_folds)
        #Estimate heart beat
        df_HR = self._estimate_heart_rate(n_folds=n_folds)
        #Add reference to the Heart Rate dataframe
        hr_references = []
        for fold_index in df_HR['Folds']:
            hr_oxymeter_bpm = reference.query("Folds == @fold_index")['HR[bpm]'].mean()
            hr_references.append(hr_oxymeter_bpm)
        df_HR['Reference[bpm]'] = hr_references
        df_HR['Reference[Hz]'] = df_HR['Reference[bpm]'] * (1/60)
        self.predictions = df_HR
        return df_HR
    
    @staticmethod
    def MAE(results) -> float:
        predictions = results['HR[bpm]'].values
        references = results['Reference[bpm]'].values
        difference_array = abs(np.subtract(predictions, references))
        return difference_array.mean()
    
    @staticmethod
    def MSR(results:pd.DataFrame) -> float:
        predictions = results['HR[bpm]'].values
        references = results['Reference[bpm]'].values
        difference_array = np.subtract(predictions, references)
        squared_array = np.square(difference_array)
        return squared_array.mean()

    @staticmethod
    def pearson_correlation(results:pd.DataFrame, plot:Optional[bool]=True) -> float:
        #Calculate pearson correlation between HR predictions and references
        predictions = results['HR[bpm]'].values
        references = results['Reference[bpm]'].values 
        pearson_correlation, _ = scipy.stats.pearsonr(predictions, references) 
        if plot:
            #Plot reference against predictions
            fig = sns.lmplot(
                x='HR[bpm]', 
                y='Reference[bpm]', 
                data=results
                )
        return pearson_correlation

    






