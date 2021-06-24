# mcmc.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


class MCMC:
    def __init__(self):
        pass

    def param_disperse_simple(self, args):
        """ Returns dispersed initial parameter values for MCMC. 

        Parameters
        ----------
        theta_ini : np.ndarray, shape=(nparam,)
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
        theta_ini, random_exponent, nwalkers, ndim = args

        pos = np.array([])
        for i in range(nwalkers):
            pos = np.append(pos, theta_ini + random_exponent * np.random.randn(ndim))
        pos = pos.reshape(nwalkers, ndim)
        return pos

    def param_disperse_complex(self, args):
        """

        Parameters
        ----------
        theta_ini : np.ndarray
        random_exponent : List[float]
        nwalkers : int
        ndim : int

        Return
        ----------
        dispersed parameters : np.ndarray, shape=(nwalkers, nparam)
        """
        theta_ini_all, random_exponent_all, nwalkers, ndim = args
        pos = np.array([])
        for i in range(nwalkers):
            pos = np.append(pos, theta_ini_all + random_exponent_all * np.random.randn(ndim))
        pos = pos.reshape(nwalkers, ndim)
        return pos
    
    def run_mcmc(self, log_probability, theta_ini, random_exponent, nwalkers, ndim, nstep, 
                 prior_param_list, t, y, param_num=None, disperse_method='simple', 
                 nprocess=1, progress=True):
        """ Run MCMC. 

        Parameters
        ----------
        log_probability : func
        theta_ini : np.ndarray, shape=(nparam,)
            initial parameter values
        random_exponent : np.ndarray, shape=(nparam,)
            10^(exponent) is the coefficient of the standard normal distribution 
            added to the initial parameter values in order to disperse them
        nwalkers : int
        ndim : int
        nstep : int
        t : np.ndarray
        y : np.ndarray
        nprocess : int
            number of CPUs to be used. 
        progress : bool
            whether to show the progress bar of MCMC. 
        """
        if disperse_method == 'simple':
            args = (theta_ini, random_exponent, nwalkers, ndim)
        elif disperse_method == 'complex':
            # expand redundant parameters
            theta_ini_all = []
            random_exponent_all = []
            prior_param_list_all = []
            for i, num in enumerate(param_num):
                theta_ini_all = theta_ini_all + [theta_ini[i]] * num
                random_exponent_all = random_exponent_all + [random_exponent[i]] * num
                prior_param_list_all = prior_param_list_all + [prior_param_list[i]] * num
            theta_ini_all = np.array(theta_ini_all)
            random_exponent_all = np.array(random_exponent_all)
            prior_param_list = prior_param_list_all

            args = (theta_ini_all, random_exponent_all, nwalkers, ndim)

        with Pool() as pool:
            pos = getattr(self, f'param_disperse_{disperse_method}')(args)
            # pos = self.param_disperse(theta_ini, random_exponent, nwalkers, ndim)
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                                 args=(prior_param_list, t, y), 
                                                 pool=pool)
            _ = self.sampler.run_mcmc(pos, nstep, progress=progress)


class Fitting:
    """
    """
    def __init__(self, data, model, sampler, discard=0, thin=1):
        self.data = data
        self.x = np.arange(len(data))
        self.model = model
        self.flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    def plot_data(self, ax):
        ax.plot(self.x, self.data, color='orange')
        return ax

    def estimate(self, theta, t):
        xi = theta[3]
        s, e, i, r = self.model.solve(theta, t)
        s, e, i, r = self.model.downsampling(t.shape[0], self.data.shape[0], s, e, i, r)
        return xi * i
    
    def plot_median(self, ax, t):
        median = np.median(self.flat_samples, axis=0)
        est = self.estimate(median, t)
        ax.plot(self.x, est, color='dodgerblue')
        return ax

    def plot_mean(self, ax, t):
        mean = np.mean(self.flat_samples, axis=0)
        est = self.estimate(mean, t)
        ax.plot(self.x, est, color='dodgerblue')
        return ax

    def plot_mode(self, ax, t, bins=10):
        mode_list = []
        for i in range(self.flat_samples.shape[1]):
            param = self.flat_samples[:, i]
            hist, mode = np.histogram(param, bins=bins)
            mode_list.append(mode[np.argmax(hist)])
        mode_arr = np.array(mode_list)
        est = self.estimate(mode_arr, t)
        ax.plot(self.x, est, color='dodgerblue')
        return ax

    def plot_credible_interval(self, ax, t, ode_step=1000):
        l_lim = np.percentile(self.flat_samples, 2.5, axis=0)
        u_lim = np.percentile(self.flat_samples, 97.5, axis=0)
        l_est = self.estimate(l_lim, t)
        u_est = self.estimate(u_lim, t)
        ax.plot(self.x, l_est, color='orchid')
        ax.plot(self.x, u_est, color='orchid')
        ax.fill_between(self.x, l_est, u_est, color='orchid', alpha=0.5)
        return ax


class Analysis:
    """
    """
    def __init__(self, sampler, discard=0, thin=1):
        self.sampler = sampler
        self.flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    def plot_scatter(self, fig, labels):
        """ See parameter distributions. 
        """
        corner.corner(self.flat_samples, labels=labels, fig=fig)
        return fig

    def plot_convergence(self, ax):
        """ Judge convergence of MCMC based on autocorrelation function. 
        """
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
