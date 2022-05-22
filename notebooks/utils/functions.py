from time import time
import plotly.graph_objects as go
from typing import List, Optional, Tuple
import pandas as pd
from sklearn import preprocessing
import numpy as np
from .enumerators import Colors, TimeUnit

def post_process_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the color channels signals
    """
    x = df[['B', 'G', 'R']].values
    x_scaled = preprocessing.StandardScaler().fit_transform(x)
    time_ms = df['Time[ms]']
    time_s = time_ms*0.001
    df_pp = pd.DataFrame(x_scaled, columns=['B', 'G', 'R'])
    df_pp['Time_ms'] = time_ms
    df_pp['Time_s'] = time_s
    return df_pp

def _display_utils(df:pd.DataFrame,
                   time_unit:TimeUnit,
                   time_range:Optional[Tuple[float, float]]=None) -> Tuple[pd.DataFrame, str]:
    if time_unit == TimeUnit.SECONDS:
        time_col = 'Time_s'
    elif time_unit == TimeUnit.MILISECONDS:
        time_col = 'Time_ms'

    if time_range is not None:
        min_time = time_range[0]
        max_time = time_range[1]
        df = df.query(f"@min_time <= `{time_col}` <= @max_time")
    
    return df, time_col

def display_columns(df:pd.DataFrame,
                    columns:List[str],
                    time_range:Tuple[float, float]=None,
                    time_unit:Optional[TimeUnit]=TimeUnit.SECONDS,
                    width:Optional[float]=0.5,):
    """
    Display signals
    """
    fig = go.Figure()
    df, time_col = _display_utils(df=df,
                                  time_unit=time_unit, 
                                  time_range=time_range)

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

def display_color_channels(df:pd.DataFrame,
                           colors:List[Colors],
                           time_unit:Optional[TimeUnit]=TimeUnit.SECONDS,
                           time_range:Optional[Tuple[float, float]]=None,
                           width:Optional[float]=0.5,):
    """
    Display datasets
    """
    fig = go.Figure()
    df, time_col = _display_utils(df=df,
                                  time_unit=time_unit, 
                                  time_range=time_range)

    for color in colors:
        color_disp = color.name.lower().replace("_f", "")
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[color.value],
                name=color.name.title(),
                line=dict(color=color_disp,
                          width=width)
            )
        )
    return fig

def draw_histogram(df:pd.DataFrame, colors:List[Colors]):
    fig = go.Figure()
    for color in colors:
        fig.add_trace(go.Histogram(x=df[color.value]))
    
    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    return fig









        
