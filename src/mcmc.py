# mcmc.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np
import emcee
import corner


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
            pos = np.append(pos, theta + random_exponent * np.random.randn(ndim))
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


class Analysis:
    """
    """
    def __init__(self, sampler, discard=0, thin=1):
        self.sampler = sampler
        self.flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        print(len(self.flat_samples))

    def plot_scatter(self, fig, labels):
        corner.corner(self.flat_samples, labels=labels, fig=fig)
        return fig

    def next_pow_two(self, n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def autocorr_func_1d(self, x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = self.next_pow_two(len(x))
        
        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n
        
        # Optionally normalize
        if norm:
            acf /= acf[0]
        
        return acf

    def auto_window(self, taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def autocorr(self, y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += self.autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self.auto_window(taus, c)
        return taus[window]

    
