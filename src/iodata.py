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
    data : np.ndarray
    """
    data = pd.read_csv(path)
    data = data[data['prefectureNameE']==prefecture][metrics].to_numpy()
    data = data[period[0]:period[1]]
    return data

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
