from utils import Colors, post_process_data
from preprocessing import butter_lowpass_filter, moving_average
from ppg import (ppg_by_ica_sources, ppg_by_grd, ppg_by_green_signal,
                  select_ppg, PPG)
import pandas as pd
from pathlib import Path
from typing import Optional


"""
Create a Pipeline for Heart Rate Prediction
"""

def pipeline(
        ppg_path:Path, 
        raw_path:Path,
        lowpass_filter:Optional[bool]=True,
        n_points:Optional[int]=12,
        n_folds:Optional[int]=10,):
    experiment_path = ppg_path.parent
    #Load the BGR signals and the reference data
    df_ppg = pd.read_csv(ppg_path)
    df_raw = pd.read_csv(raw_path)
    #Process PPG data
    df_ppg = post_process_data(df_ppg)
    #Signal Processing
    for color in [Colors.BLUE, Colors.GREEN, Colors.RED]:
        #Lowpass Filter
        if lowpass_filter:
            df_ppg[f"{color.value}_F"] = butter_lowpass_filter(
                signal=df_ppg[color.value].values,
                cutoff=4,
                fs=30,
                order=5
            )
        else:
            df_ppg[f"{color.value}_F"] = df_ppg[color.value].values
        
        #Moving Average
        if n_points:
            df_ppg[f"{color.value}_F"] = moving_average(
                signal=df_ppg[f"{color.value}_F"],
                n_points=n_points
            )
        else:
            pass

    #Extracting PPG from color signals
    #Consider PPG to be simply the green signal
    df_ppg = ppg_by_green_signal(df_ppg)
    #Consider PPG to simply be the difference between 
    #the green and red channels
    df_ppg = ppg_by_grd(df_ppg)
    #Apply Independent Component Analysis
    df_ppg = ppg_by_ica_sources(df_ppg)

    #Choose the PPG signal for heart beat estimation
    best_signal = select_ppg(df_ppg)
    #Predict heart beat
    ppg = PPG(
        df_ppg=df_ppg,
        source_name=best_signal,
    )
    print(ppg)
    ppg.results(
            reference=df_raw,
            n_folds=n_folds,
        )
    
    return ppg


    

    






