# utils.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np
import networkx as nx


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


class PositiveRate(Calibration):
    def __init__(self):
        pass

    def linear(self, sol, xi):
        _, _, i, _ = sol
        return xi * i


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
