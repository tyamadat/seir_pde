# mcmc.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np
import emcee

import utils
import params


class MCMC:
    def __init__(self):
        pass

    def param_disperse(self, theta, random_exponent, nwalkers, ndim):
        """ Returns dispersed initial parameter values for MCMC. 

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
            initial parameter values
        random_exponent : np.ndarray, shape=(nparam)
            10^(exponent) is the coefficient of the standard normal distribution 
            added to the initial parameter values in order to disperse them
        nwalkers : int
            number of walkers in MCMC

        Return
        ----------
        dispersed parameters : np.ndarray, shape=(nwalkers, nparam(ndim))
        """
        pos = np.array([])
        for i in range(nwalkers):
            pos = np.append(pos, theta + 10**random_exponent * np.random.randn(ndim))
        pos = pos.reshape(nwalkers, ndim)
        return pos
    
    def run_mcmc(self, log_probability, theta, random_exponent, 
                 nwalkers, ndim, nstep, prior_param_list, t, y, progress=True):
        """ Run MCMC. 

        Parameters
        ----------
        log_probability : func
        theta : np.ndarray, shape=(nparam)
            initial parameter values
        random_exponent : np.ndarray, shape=(nparam)
            10^(exponent) is the coefficient of the standard normal distribution 
            added to the initial parameter values in order to disperse them
        nwalkers : int
        ndim : int
        nstep : int
        t : np.ndarray
        y : np.ndarray
        progress : bool
        """
        pos = self.param_disperse(theta, random_exponent, nwalkers, ndim)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                             args=(prior_param_list, t, y))
        _ = self.sampler.run_mcmc(pos, nstep, progress=progress)
