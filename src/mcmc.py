# mcmc.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


class MCMC:
    def __init__(self):
        pass

    def param_disperse(self, theta, random_exponent, nwalkers, ndim):
        """ Returns dispersed initial parameter values for MCMC. 

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam,)
            initial parameter values
        random_exponent : np.ndarray, shape=(nparam,)
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
        theta : np.ndarray, shape=(nparam,)
            initial parameter values
        random_exponent : np.ndarray, shape=(nparam,)
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

    def plot_convergence(self, ax):
        chain = self.sampler.get_chain()[:, :, 0].T
        
        # Compute the extimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        autocorr = np.empty(len(N))
        for i, n in enumerate(N):
            autocorr[i] = self.autocorr(chain[:, :n])
        
        # Plot the comparisons
        ax.plot(N, autocorr, '-o', label='Integrated autocorrelation')
        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        ylim = plt.gca().get_ylim()
        ax.plot(N, N / 50.0, '--k', label=r'$\tau = N/50$')
        ax.set_ylim(ylim)
        ax.set_xlabel('number of samples, $N$')
        ax.set_ylabel(r'$\tau$ estimates')
        ax.legend(fontsize=14)

        return ax

    def next_pow_two(self, n):
        """ Returns the next power of two greated than or equal to 'n'.

        Parameter
        ----------
        n : int
            length of each chain

        Return
        ----------
        i : int
            the next power of two greated than or equal to 'n'
        """
        i = 1
        while i < n:
            i = i << 1  # a left shift by 1 bit
        return i

    def autocorr_func_1d(self, x, norm=True):
        """ Returns autocorrelation function. 

        Parameters
        ----------
        x : np.ndarray, shape=(N (number of samples),)
            array of the trajectory of a walker
        norm : bool, default=True
            whether to normalize the autocorrelation function

        Return
        ----------
        acf : np.ndarray, shape=(N,)
            autocorrelated function
        """
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
        """ Returns a window size to estimate the integrated autocorrelation time.  

        Parameters
        ----------
        taus : np.ndarray, shape=(2N+1,)
            array of the integrated autocorrelation time
        c : float
            constant for defining window size m

        Return
        ----------
        m : int
            window size: the smallest value of M where M >= C*tau(M) \
            for a constant C~5.
        """
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def autocorr(self, y, c=5.0):
        """ Returns an estimation of the integrated autocorrelation time. 

        Parameters
        ----------
        y : np.ndarray, shape=(nwalker, N (number of samples))
            arrays of each chain for which the integrated autocorrelation time \
            is calculated
        c : float
            constant for defining window size m

        Return
        ----------
        tau : float
            the integrated autocorrelation time 
        """
        f = np.zeros(y.shape[1])
        for yy in y:
            f += self.autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self.auto_window(taus, c)
        return taus[window]
