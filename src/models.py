# models.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = '10'
os.environ["MKL_NUM_THREADS"] = '10'
os.environ["VECLIB_NUM_THREADS"] = '10'

import numpy as np
from scipy.integrate import odeint
from scipy.sparse import dia_matrix, identity
from odeintw import odeintw
import networkx as nx
from tqdm import tqdm

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
    def __init__(self, prefecture='Tokyo', population=1e7, calibration='NC'):
        self.prior = utils.Prior
        self.likelihood = utils.Likelihood()
        self.params = params.SeirOde(population=population)
        if calibration == 'NC': # New case
            self.calib = utils.NewCase()
            self.model = 'i'
            self.params.set_nc()
        elif calibration == 'PR': # Positive rate
            self.calib = utils.PositiveRate()
            self.model = 'linear'
            self.params.set_pr()
        else:
            raise ValueError(f'Calibration "{calibration} is not implemented."')
        self.prefecture=prefecture

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

    def solve(self, theta, t):
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

    def downsampling(self, lent, leny, s, e, i, r):
        """ Returns downsampled ODE results that match observed data. 

        Parameters
        ----------
        lent : float
            length of array t
        leny : float
            length of array y

        Return
        ----------
        downsampled solution : List[np.ndarray]
        """
        step = lent // leny
        if lent % leny == 0:
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

    def log_likelihood(self, theta, t, y, n_beta=1, likelihood='gaussian'):
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
        s, e, i, r = self.solve(theta, t)
        s, e, i, r = self.downsampling(t.shape[0], y.shape[0], s, e, i, r)
        xi = theta[n_beta:][2]
        tau = theta[n_beta:][5]
        # y_model = getattr(self.calib, self.model)(self.calib, [s, e, i, r], xi)
        y_model = getattr(self.calib, self.model)([s, e, i, r], xi)
        return getattr(self.likelihood, likelihood)(y, y_model, tau)

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


class SeirPde(SeirOde):
    """ Diffusion model from Tokyo (described by ODE) described by PDE.
    """
    def __init__(self, prefecture='Tokyo', population=1e7):
        super().__init__(prefecture=prefecture, population=population)
        self.params = params.SeirPde(population=population)

    def pde(self, u, t, A, dx, beta, epsilon, rho):
        """ Describe PDE as a set of ODEs by a central difference approximation in space. \
        Initial condition: u[0, 0]=u[0], u[x, 0]=0 (x!=0) \
        Boundary condition: Neumann boundary condition

        Parameters
        ----------
        u : List[np.ndarray], shape=(4, )
        t : np.ndarray
            mesh points in time
        A : scipy.sparse
            operator for calculating diffusion term
        dx : float
            spatial resolution. 
        """
        s, e, i, r = u

        dsdt = A.dot(s) / dx**2
        dedt = A.dot(e) / dx**2
        didt = A.dot(i) / dx**2
        drdt = A.dot(r) / dx**2

        # boundary condition at x=0
        dsdt[0] = -beta*i[0]*s[0]
        dedt[0] = beta*i[0]*s[0] - epsilon*e[0]
        didt[0] = epsilon*e[0] - rho*i[0]
        drdt[0] = rho*i[0]
        
        return [dsdt, dedt, didt, drdt]

    def solve(self, theta, t, t2=None, ode_step=None):
        """ Returns PDE solutions. 

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
            n : number of points in space
            D : diffusion coefficient
            L : upper limit of space (lower limit: 0)

        Return
        ----------
        u_arr : np.ndarray, shape=(len(t), 4, len(x))
            solution for the PDEs
        """
        n, D, L, beta, epsilon, rho, _, e0, i0, _ = theta
        n = int(n)
        x = np.linspace(0, L, n+1) # mesh points in space
        dx = x[1] - x[0] # spatial resolution
        A = self.gen_operator(n, D)
        args = (A, dx, beta, epsilon, rho)
        
        # Set initial conditions
        s0 = self.params.population - e0 - i0
        s0 = np.append(s0, np.zeros(n))
        e0 = np.append(e0, np.zeros(n))
        i0 = np.append(i0, np.zeros(n))
        r0 = np.zeros(n+1)

        u0 = [s0, e0, i0, r0]
        u_arr = odeintw(self.pde, u0, t, args=args)
        if t2 is None or t2[0]==0:
            return u_arr
        else:
            s0 = np.append(u_arr[int(t2[0]*ode_step), 0, 0], np.zeros(n))
            e0 = np.append(u_arr[int(t2[0]*ode_step), 1, 0], np.zeros(n))
            i0 = np.append(u_arr[int(t2[0]*ode_step), 2, 0], np.zeros(n))
            r0 = np.append(u_arr[int(t2[0]*ode_step), 3, 0], np.zeros(n))
            u0 = [s0, e0, i0, r0]
            u_arr = odeintw(self.pde, u0, t2-t2[0], args=args)
            return u_arr

    def gen_operator(self, n, D):
        """ Returns operator for calculating diffusion term. 

        Parameters
        ----------
        n : int
        D : float
            diffusion coefficient

        Return
        ----------
        A : scipy.sparse.csr.csr_matrix
        """
        # Using a forward difference in time and a central difference in space
        mx = np.array([np.ones(n+1), -2.0*np.ones(n+1), np.ones(n+1)])
        mx[0, n-1] = 2 # for boundary condition at x=L (Neumann boundary condition)
        offsets = np.array([-1, 0, 1])
        B = dia_matrix((mx, offsets), shape=(n+1, n+1)) # create diagonal matrix
        E = identity(n+1)
        A = E + D*B
        return A


class SeirPdeSimulation(SeirPde):
    def __init__(self, prefecture='Tokyo', population=1e7):
        super().__init__(prefecture=prefecture, population=population)
        self.params = params.SeirPde(population=population)
    
    def pde_euler(self, i0, i_boundary, t, A, dx):
        """ Describe PDE as a set of ODEs by a central difference approximation in space. \
        Initial condition: u[0, 0]=u[0], u[x, 0]=0 (x!=0) \
        Boundary condition: Neumann boundary condition

        Parameters
        ----------
        i0 : np.ndarray, shape=(n,)
        i_boundary : np.ndarray, shape=(lent,)
        A : scipy.sparse
            operator for calculating diffusion term
        dx : float
            spatial resolution. 

        Return
        ----------
        i : np.ndarray, shape=(n, lent)
        """
        dt = t[1] - t[0]
        lent = len(t)
        i = np.array([i0])

        for j in tqdm(range(lent-1)):
            i[-1, 0] = i_boundary[j]
            it = i[-1, :]
            didt = A.dot(it) / dx**2 * dt
            i = np.vstack([i, it+didt])
        return i

    def solve(self, theta, t, t2=None, cf=1., ode_step=None):
        """ Returns PDE solutions. 

        Parameters
        ----------
        theta : np.ndarray, shape=(nparam)
            n : number of points in space
            D : diffusion coefficient
            L : upper limit of space (lower limit: 0)
        cf : float
            coefficient for simulation

        Return
        ----------
        u_arr : np.ndarray, shape=(len(t), 4, len(x))
            solution for the PDEs
        """
        n, D, L, beta, epsilon, rho, _, e0, i0, _ = theta
        n = int(n)
        x = np.linspace(0, L, n+1) # mesh points in space
        dx = x[1] - x[0] # spatial resolution
        A = self.gen_operator(n, D)
        args = (A, dx)

        # Calculate boundary condition (x=0)
        _, _, i_boundary, _ = SeirOde(population=self.params.population).solve(theta[3:], t)
        i_boundary = i_boundary * 10**cf

        # Set initial conditions
        i0 = np.append(i0, np.zeros(n))
        i = self.pde_euler(i0, i_boundary, t, A, dx)

        if t2 is None or t2[0]==0:
            return i
        else:
            i0 = np.append(i[int(t2[0]*ode_step), 0], np.zeros(n))
            i = self.pde_euler(i0, i_boundary[t2[0]:t2[-1]], t2-t2[0], A, dx)
            return i


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

    def solve(self, theta, t):
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
