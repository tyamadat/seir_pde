# models.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""


import numpy as np
from scipy.integrate import solve_ivp

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

    def ode(self, t, u, beta, epsilon, rho):
        """ Returns the list of ODE functions

        Parameters
        ----------
        t : np.ndarray
        u : List[np.ndarray], shape=(4, )

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
        s0 = self.population - e0 - i0
        r0 = 0
        u0 = [s0, e0, i0, r0]
        u_list = solve_ivp(self.ode, t, u0, args=args)
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
        return [s[::step], e[::step], i[::step], r[::step]]

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

    def log_likelihood(self, theta, t, y, model='linear', likelihood='gaussian'):
        """ Returns log likelihood value for an observed value based on the model.

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
        t : 
        y : 
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
        _, _, _, xi, _, _, tau = theta
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
