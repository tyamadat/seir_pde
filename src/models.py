# models.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""


import numpy as np
from scipy.integrate import odeint
from odeintw import odeintw
import networkx as nx

import utils
import params


class SeirOde:
    """ Basic SEIR model described by ordinary differential equations (ODE). 

    Variables: 
    S (susceptible), E (exposed), I (infected), R (recovered). 

    Parameters: 
    beta (transmission rate), epsilon (transfer rate from E to I), 
    rho (recovery rate), xi (coefficient for calculating positive rate), 
    E0 (size of E on the first day), I0 (size of I on the first day), 
    tau (1/tau is the variance of data noise).

    Parameters
    ----------
    calibration : str, default 'PR' (positive rate)

    """
    def __init__(self, prefecture='Tokyo', population=1e7, calibration='PR'):
        self.prior = utils.Prior
        self.likelihood = utils.Likelihood
        if calibration == 'PR':
            self.calib = utils.PositiveRate
        else:
            raise ValueError(f'Calibration "{calibration} is not implemented."')
        self.prefecture=prefecture
        self.params = params.SeirOde(population=population)

    def ode(self, u, t, beta, epsilon, rho):
        """ Returns the list of ODE functions

        Parameters
        ----------
        u : List[np.ndarray], shape=(4, )
        t : np.ndarray

        Return
        ----------
        ODEs : List[np.ndarray], shape=(4, )
        """
        s, e, i, r = u
        dsdt = -beta*i*s
        dedt = beta*i*s - epsilon*e
        didt = epsilon*e - rho*i
        drdt = rho*i
        return [dsdt, dedt, didt, drdt]

    def solve_ode(self, theta, t):
        """ Returns a solution for the ODEs. 

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
        t : np.ndarray

        Return
        ----------
        solution : List[np.ndarray]
        """
        beta, epsilon, rho, _, e0, i0, _ = theta
        args = (beta, epsilon, rho)
        s0 = self.params.population - e0 - i0
        r0 = 0
        u0 = [s0, e0, i0, r0]
        u_list = odeint(self.ode, u0, t, args=args)
        s, e, i, r = u_list[:, 0], u_list[:, 1], u_list[:, 2], u_list[:, 3]
        return [s, e, i, r]

    def downsampling(self, t, y, s, e, i, r):
        """ Returns downsampled ODE results that match observed data. 

        Parameters
        ----------
        t : np.ndarray
        y : np.ndarray

        Return
        ----------
        downsampled solution : List[np.ndarray]
        """
        step = len(t) // len(y)
        if len(t) % len(y) == 0:
            return [s[::step], e[::step], i[::step], r[::step]]
        else:  # x[::step] becomes len(y)+1 -> remove the final element in the array
            return [s[::step][:-1], e[::step][:-1], i[::step][:-1], r[::step][:-1]]

    def log_prior(self, theta, prior_param_list, prior='uniform'):
        """ Returns log prior probability. 

        Parameter
        ----------
        theta : np.ndarray, shape=(nparam)
        prior_param_list : List[Tuple], shape=(nparam)
            each tuple contains prior distribution parameters. 
            e.g., uniform: (min, max).

        Return
        ----------
        log prior probability : float
        """
        lp = 0.0
        for i, param in enumerate(theta):
            lp += getattr(self.prior, prior)(self.prior, param, prior_param_list[i])
        return lp

    def log_likelihood(self, theta, t, y, n_beta=1, model='linear', likelihood='gaussian'):
        """ Returns log likelihood value for an observed value based on the model.

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
        t : 
        y : 
        n_beta : int
            number of parameter beta in the model.  
        model : str
            the name of a model for y.
        likelihood : str
            the name of likelihood function.

        Return
        ----------
        log likelihood : np.float
        """
        s, e, i, r = self.solve_ode(theta, t)
        s, e, i, r = self.downsampling(t, y, s, e, i, r)
        xi = theta[n_beta:][2]
        tau = theta[n_beta:][5]
        y_model = getattr(self.calib, model)(self.calib, [s, e, i, r], xi)
        return getattr(self.likelihood, likelihood)(self.likelihood, y, y_model, tau)

    def log_probability(self, theta, prior_param_list, t, y):
        """ Returns log posterior probability (log prior + log likelihood). 

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
        t : np.ndarray
        y : np.ndarray

        Return
        ----------
        log posterior probability : float
            ∝ log prior probability + log likelihood
        """
        lp = self.log_prior(theta, prior_param_list)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, t, y)


class GraphDiff(SeirOde):
    """ Network SEIR model with diffusion described by ordinary differential equations (ODE).

    Parameters
    ----------
    g : nx.Graph
        networkx Graph object which contains network information \
        about prefectures
    arr_pop : np.ndarray, shape=(n_pref, )
    calibration : str, default 'PR' (positive rate)

    Attributes
    ----------
    L : np.array, shape=(n_pref, n_pref)
        Laplacian matrix of the prefecture network.
    n_pref : int
        number of nodes in the prefecture network = number of prefectures. 
    """
    def __init__(self, g, arr_pop, calibration='PR'):
        super().__init__(calibration=calibration)
        self.params = params.GraphDiff(n_pref=g.number_of_nodes())
        self.g = g
        self.L = nx.linalg.laplacianmatrix.laplacian_matrix(g).toarray()
        self.n_pref = nx.number_of_nodes(g)
        self.arr_pop = arr_pop

    def ode(self, u, t, beta, epsilon, rho, d):
        """

        Parameters
        ----------
        u : 

        Return
        ----------

        """
        s, e, i, r = u
        dsdt = -beta*i*s + self.L.dot(d*beta*s)
        dedt = beta*i*s - epsilon*e + self.L.dot(d*beta*e)
        didt = epsilon*e - rho*i + self.L.dot(d*beta*i)
        drdt = rho*i + self.L.dot(d*beta*r)
        return [dsdt, dedt, didt, drdt]

    def solve_ode(self, theta, t):
        """

        Parameters
        ----------
        theta : np.ndarray, shape=((n_pref + 7), )
            array of parameters. 

        Return
        ----------
        """
        beta = theta[:self.n_pref]
        epsilon, rho, _, e0, i0, _, d = theta[self.n_pref:]

        # convert from scalar to vector (for each prefecture)
        epsilon = epsilon * np.ones(self.n_pref)
        rho = rho * np.ones(self.n_pref)
        d = d * np.ones(self.n_pref)
        args = (beta, epsilon, rho, d)

        # initial values
        e0 = e0 * self.arr_pop
        i0 = i0 * self.arr_pop
        s0 = self.arr_pop - e0 - i0
        r0 = np.zeros(self.n_pref)
        u0 = [s0, e0, i0, r0]

        u_list = odeintw(self.ode, u0, t, args=args)
        s, e, i, r = u_list[:, 0], u_list[:, 1], u_list[:, 2], u_list[:, 3]
        return [s, e, i, r]

    def downsampling(self, t, y, s, e, i, r):
        """
        Parameters
        ----------
        t : np.ndarray
        y : np.ndarray, shape=(n_days, n_pref)

        Return
        ----------
        downsampled solution : List[np.ndarray]
        """
        step = t.shape[0] // y.shape[0]
        if len(t) % len(y[:, 0]) == 0:
            return [s[::step], e[::step], i[::step], r[::step]]
        else:  # x[::step] becomes len(y)+1 -> remove the final element in the array
            return [s[::step][:-1], e[::step][:-1], i[::step][:-1], r[::step][:-1]]
