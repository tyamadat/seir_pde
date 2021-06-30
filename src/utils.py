# utils.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import networkx as nx

import models


class Prior:
    """ Log prior probability for estimating posterior distributions. 
    """
    def __init__(self):
        pass

    def uniform(self, val, range):
        """

        Parameters
        ----------
        val : float
        range : Tuple (float, float)

        Returns
        ---------
        log prior probability : float
            0.0 (=log1) or -inf (=log0)
        """
        if range[0] < val < range[1]:
            return 0.0
        return -np.inf


class Likelihood:
    """ Log-likelihood functions for estimating posterior distributions. 
    """
    def __init__(self):
        pass

    def gaussian(self, y, y_model, tau):
        """ Returns log_likelihood value for gaussian distribution. 

        Parameters
        ----------
        y : float
            observed value
        y_model : float
            expected value (mean of the gaussian distribution)
        tau : float
            1/tau is the variance of the gaussian distribution

        Return
        ----------
        log likelihood : float
        """
        return -0.5 * np.sum((y - y_model) ** 2 / tau + np.log(tau))


class Calibration:
    def __init__(self):
        pass


class NewCase(Calibration):
    def __init__(self):
        pass

    def i(self, sol, xi):
        _, _, i, _ = sol
        return xi * i


class PositiveRate(Calibration):
    def __init__(self):
        pass

    def linear(self, sol, xi):
        _, _, i, _ = sol
        return xi * i


class PDE:
    """
    """
    def __init__(self):
        self.model = models.SeirPde()

    def fitting(self, sol, y_obs, xi, x_range, y_range, metrics='MSE'):
        """ Returns list of fitting values for each x (distance from epicenter).

        Parameters
        ----------
        sol : np.ndarray, shape=(len(t), 4, len(x))
            solution for the PDEs
        y_obs : np.ndarray, shape=(duration, )
        x_range : Tuple(int, int)

        y_range : Tuple(int, int)
            the range of y_model and y_obs to separate before and after the peak

        Return
        ----------
        eval_arr : np.ndarray, shape=(x_range[1]-x_range[0], )
            List of fitting values for each x
        """
        leny = y_obs.shape[0]
        y_obs = np.cumsum(y_obs[y_range[0]:y_range[1]])

        eval_list = []
        for x in range(x_range[0], x_range[1]):
            u_list = sol[:, :, x]
            s, e, i, r = u_list[:, 0], u_list[:, 1], u_list[:, 2], u_list[:, 3]
            s, e, i, r = self.model.downsampling(s.shape[0], leny, s, e, i, r)
            y_model = np.cumsum(xi*i)[y_range[0]-y_range[0]:y_range[1]-y_range[0]]
            eval_list.append(self.fit_eval(y_model, y_obs, metrics=metrics))

        return np.array(eval_list)

    def fit_eval(self, y_model, y_obs, metrics='MSE'):
        """ Returns a fitting score based on observed and model data.

        Parameters
        ----------
        y_model : np.ndarray, shape=(duration, )
        y_obs : np.ndarray, shape=(duration, )

        Return
        ----------
        fitting_evalutation : float
        """
        if metrics=='MSE':
            return mean_squared_error(y_model, y_obs)
        elif metrics=='RMSE':
            return np.sqrt(mean_squared_error(y_model, y_obs))
        elif metrics=='MAE':
            return mean_absolute_error(y_model, y_obs)

    def fit_plot(self, ax, sol, y_obs, xi, x, y_range):
        """
        """
        leny = y_obs.shape[0]
        u_list = sol[:, :, x]
        s, e, i, r = u_list[:, 0], u_list[:, 1], u_list[:, 2], u_list[:, 3]
        s, e, i, r = self.model.downsampling(s.shape[0], leny, s, e, i, r)

        x = np.arange(y_range[0]-y_range[0], y_range[1]-y_range[0])
        y_obs = np.cumsum(y_obs[y_range[0]:y_range[1]])
        y_model = np.cumsum(xi*i)[y_range[0]-y_range[0]:y_range[1]-y_range[0]]
        ax.plot(x, y_obs)
        ax.plot(x, y_model)
        return ax


class PrefectureNetwork:
    """ Graph network of prefectures
    """
    def __init__(self):
        self.g = nx.Graph()

    def build(self, df, c_type='Adjuscent'):
        """

        Parameters
        ----------
        df : pd.DataFrame
            dataframe which contains information for each prefecture and \
            prefectures to be connected to it. \
            index : prefecture index, columns = ['Prefecture', 'Adjuscent', ...]
        c_type : str, default='Adjuscent'
            type of connection between prefectures.
        """
        # create nodes
        for i in df.index:
            self.g.add_node(i, 
                            prefecture=df.loc[i, 'Prefecture'], 
                            connect=df.loc[i, c_type])

        # connect nodes with each other based on the connect information in the df
        for i in df.index:
            connect_list = self.g.node[i]['connect']
            for connect_pref in connect_list:
                if connect_pref in list(nx.get_node_attributes(self.g, 'prefecture').values()):
                    self.g.add_edge(i, df[df['Prefecture']==connect_pref].index[0])
                else:
                    pass
