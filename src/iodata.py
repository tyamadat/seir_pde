# io.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

from pathlib import Path

import numpy as np
import pandas as pd

def load_data(path, prefecture, metrics, period):
    """ Returns data for particular metrics during designated period in a prefecture. 

    Parameters
    ----------
    path: Path
        path to the data
    prefecture : str
    metrics : str
    period : Tuple(int, int)
        the day count from the beginning of the data (start, end)

    Return
    ----------
    data : np.ndarray, shape=(period[1]-period[0], )
    """
    data = pd.read_csv(path)
    data = data[data['prefectureNameE']==prefecture][metrics].to_numpy()
    data = data[period[0]:period[1]]
    return data

def load_data_matrix(path, network_range, metrics, period):
    """ Returns data for particular metrics during designated period in several prefectures. 

    Parameters
    ----------
    network_range : Tuple(int, int)
        the range of prefecture indexes to be considered. 
        range[0] ~ range[1] : both included. 
    metrics : str
    period : Tuple(int, int)
        the day count from the beginning of the data (start, end)

    Return
    ----------
    data : np.ndarray, shape=(period[1]-period[0], range[1]-range[0]+1)
    """
    data = pd.read_csv(path)
    data_matrix = []
    for i in range(network_range[0], network_range[1]+1):
        data_matrix.append(data[data['prefectureNumber']==i][metrics].to_numpy()[period[0]:period[1]])
    data_matrix = np.array(data_matrix).T # transpose for reshape to (period, range)
    return data_matrix

def load_population(path, prefecture):
    """ Returns population in a given prefecture. 

    Parameters
    ----------
    prefecture : str

    Return
    ----------
    population : int
    """
    data = pd.read_csv(path, header=0, index_col=0)
    return int(data.loc[prefecture])

def load_population_arr(path, network_range):
    """ Returns a dataframe which contains population in each prefecture.

    Parameters
    ----------
    network_range : Tuple(int, int)
        the range of prefecture indexes to be considered. 
        range[0] ~ range[1] : both included. 

    Return
    ----------
    population_arr : np.ndarray, shape=(range[1]-range[0]+1, )
    """
    data = pd.read_csv(path, header=0)
    data.index = range(1, 48) # convert index to the prefecture index (1~47)
    data_arr = np.array(data.loc[network_range[0]:network_range[1]]['Population'].tolist(), dtype=int)
    return data_arr

def load_prefecture_network(path):
    """ Returns a dataframe which contains each prefecture name and \
    the name of connected prefectures. 

    Return
    ----------
    df : pd.DataFrame
        index : prefecture indexes, \
        colums: [Prefecture, Adjuscent, ...]
    """
    data = pd.read_csv(path, header=0, index_col=0)
    return data
