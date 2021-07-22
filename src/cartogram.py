# cartogram.py
# Contact: Tetsuya Yamada <tetsuyamada1222@gmail.com>

"""
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def local_metric(pref1, pref2, pref_cood):
    """ Returns local (geographical) distance between prefecture 1 and 2.

    Parameters
    ----------
    pref1 : str
        Prefecture 1
    pref2 : str
        Prefecture 2
    pref_cood : dict[str, Tuple[float, float]]
        Dictionary of coordinates of all 47 prefectures. 

    Return
    ----------
    d : float
        Geographical distance between prefecture 1 and 2.
    """
    y1, x1 = pref_cood[pref1]
    y2, x2 = pref_cood[pref2]
    vec = np.array([x2, y2]) - np.array([x1, y1])
    d = np.linalg.norm(vec)
    return d

def calc_theta(pref_cood, local_connectivity_dic, xt=139.69167, yt=35.68944):
    """ Returns array of the bearing of a linke between two prefectures. 

    Parameters
    ----------
    pref_cood : dict[str, Tuple[float, float]]
        Dictionary of coordinates of all 47 prefectures. 
    local_connectivity_dic : dict[str, List[str, ...]]
        Dictionary of adjacent prefectures to be considered. 
    xt : float
        Longitude of the center (default: Tokyo).
    yt : float
        Latitude of the center (default: Tokyo). 

    Return
    ----------
    theta : np.ndarray, shape=(num_links, )
        Array of the bearing of a link between two prefectures in azimuth.
    """
    theta_lst = []

    # Tokyo to other prefectures : effective distance
    for pref in pref_list_noTokyo:
        yj, xj = pref_cood[pref]
        vec = np.array([xj, yj])-np.array([xt, yt])
        theta_lst.append(np.arctan2(vec[0], vec[1]))
        
    # Connection between other prefectures : geographical distance
    for pref in local_connectivity_dic.keys():
        yi, xi = pref_cood[pref]
        for pref_adj in local_connectivity_dic[pref]:
            yj, xj = pref_cood[pref_adj]
            vec = np.array([xj, yj]) - np.array([xi, yi])
            theta_lst.append(np.arctan2(vec[0], vec[1]))

    theta = np.array(theta_lst)
    return theta

def calc_A_t(eff_df, e_coef, local_connectivity_dic):
    """ Returns components needed for constructing simultaneous equations.

    Parameters
    ----------
    eff_df : pd.DataFrame, columns=(Prefecture, RMSE_minx)
        DataFrame that contains information on effective distance. 
    e_coef : float
        A coefficient to convert effective distance into geographical distance.
    local_connectivity_dic : dict
        Dictionary of adjacent prefectures to be considered. 

    Returns
    ----------
    A : np.ndarray, shape=(num_links, 46)
        Matrix to express simultaneous equations of 
        tsin - (xj-xi) and tcos - (yj-yi).
    t : np.ndarray, shape=(num_links, )
        Array of distances between two prefectures. 
    """
    t_lst = []
    A_lst = []
    nlink = 0
    
    # Tokyo to other prefectures : effective distance
    for pref in pref_list_noTokyo:
        j = pref_num_noTokyo[pref]
        eff = eff_df[eff_df['Prefecture']==pref]['RMSE_minx'].values[0]
        t_lst.append(eff * e_coef)
        Aij = np.zeros(46, dtype=int).tolist()
        Aij[j] = 1
        A_lst.append(Aij)
        nlink += 1
    
    # Connection between other prefectures : geographical distance
    for pref in local_connectivity_dic.keys():
        i = pref_num_noTokyo[pref]
        for pref_adj in local_connectivity_dic[pref]:
            j = pref_num_noTokyo[pref_adj]
            t = local_metric(pref, pref_adj, pref_latlng)
            t_lst.append(t)
            
            Aij = np.zeros(46, dtype=int).tolist()
            Aij[j] = 1
            Aij[i] = -1
            A_lst.append(Aij)
            nlink += 1
    
    A = np.array(A_lst, dtype=int).reshape(nlink, 46)
    t = np.array(t_lst)
    return A, t

def calc_xy(t, theta, xt=139.69167, yt=35.68944):
    """ Returns distance from the reference point on x and y axes.

    Parameters
    ----------
    t : float
        Distance from the reference point.
    theta : float
        Bearing from the reference point in azimuth. 
    xt : float
        x coordinate of the reference point (default: Tokyo). 
    yt : float
        y coordinate of the reference point (default: Tokyo)

    Returns
    ----------
    tsin : float
        Distance on x axis. 
    tcos : float
        Distance on y axis.
    """
    tsin = t * np.sin(theta)
    tcos = t * np.cos(theta)
    tsin[:46] = tsin[:46] + xt
    tcos[:46] = tcos[:46] + yt
    return tsin, tcos

def transform(eff_df, e_coef, local_connectivity_dic, cood_ini, weight=None):
    """ Returns coordinates of each prefecture after distortion. 

    Parameters
    ----------
    eff_df : pd.DataFrame
        DataFrame that contains information on effective distance. 
    e_coef : float
        A coefficient to convert effective distance into geographical distance.
    local_connectivity_dic : dict
        Dictionary of adjacent prefectures to be considered. 
    cood_ini : dict[str, Tuple[float, float]]
        Initial coordinates of each prefecture. 
    weight : np.ndarray, shape=(num_links, )
        Array of weight for each links.         

    Return
    ----------
    pref_cood : dict[str, Tuple[float, float]]
        Dictionary of coordinates of all 47 prefectures. 
    """
    A, t = calc_A_t(eff_df, e_coef, local_connectivity_dic)
    theta = calc_theta(cood_ini, local_connectivity_dic)
    if weight is None:
        weight = np.ones(t.shape[0], dtype=int)
    weight = np.sqrt(np.diag(weight))

    while True:
        tsin, tcos = calc_xy(t, theta, xt=139.69167, yt=35.68944)
        A = np.dot(weight, A)
        tsin = np.dot(tsin, weight)
        tcos = np.dot(tcos, weight)
        x_arr = np.linalg.lstsq(A, tsin, rcond=None)[0]
        y_arr = np.linalg.lstsq(A, tcos, rcond=None)[0]
        cood_lst = [(y_arr[i], x_arr[i]) for i in range(x_arr.shape[0])]
        pref_cood = dict(zip(pref_list_noTokyo, cood_lst))
        theta_new = calc_theta(pref_cood, local_connectivity_dic)
        theta_diff = np.abs(theta_new - theta)
        if np.all(theta_diff < 0.01):
            return pref_cood
        theta = theta_new

def plot_map(ax, pref_cood, connect=False, local_connectivity_dic=None, xt=139.69167, yt=35.68944):
    """ Plot distorted map. 

    Parameters
    ----------
    ax : matplotlib.axes
    pref_cood : dict[str, Tuple[float, float]]
        Dictionary of coordinates of all 47 prefectures. 
    connect : bool
        Display links between prefectures. 
    local_connectivity_dic : dict
        Dictionary of adjacent prefectures to be considered. 
    xt : float
        x coordinate of the reference point (default: Tokyo). 
    yt : float
        y coordinate of the reference point (default: Tokyo)

    Return
    ----------
    ax : matplotlib.axes
    """
    for pref in pref_cood.keys():
        y, x = pref_cood[pref]
        ax.scatter(x, y, color=pref_color_region[pref], s=100)
        if connect:
            if pref in local_connectivity_dic.keys():
                for pref_adj in local_connectivity_dic[pref]:
                    ya, xa = pref_cood[pref_adj]
                    ax.plot([x, xa], [y, ya], color='skyblue')
    ax.scatter(xt, yt, color='black', s=100)
    return ax
