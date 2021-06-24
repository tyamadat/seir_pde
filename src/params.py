# models.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np


class SeirOde:
    def __init__(self, population=1e7):
        self.population = population
        self.list = [
            'beta', 
            'epsilon', 
            'rho', 
            'xi', 
            'e0', 
            'i0', 
            'tau', 
        ]
        self.param_num = None
        self.ndim = 7
    
    def set_nc(self):
        self.ini = np.array([
            8.0e-8, # beta
            1.0e-1, # epsilon
            1.0e-1, # rho
            1.0,    # xi
            3.8e-5*self.population, # e0
            3.8e-5*self.population, # i0
            2.0e-5, # tau
        ])
        self.uniform_range = [
            (1e-10, 1e-5), # beta
            (1e-2, 1.),    # epsilon
            (1e-2, 1.),    # rho
            (1e-3, 1e1),   # xi
            (1e1, 1e4),    # e0
            (1e1, 1e4),    # i0
            (1e-8, 1e2),   # tau
        ]
        self.random_exponent = np.array([
            1e-12, # beta
            1e-4,  # epsilon
            1e-4,  # rho
            1e-5,  # xi
            1e-1,  # e0
            1e-1,  # i0
            1e-10, # tau
        ])
    
    def set_pr(self):
        self.ini = np.array([
            1.4e-7, # beta
            2.0e-1, # epsilon
            3.8e-1, # rho
            6.4e-7, # xi
            5.5e-2*self.population, # e0
            1.2e-2*self.population, # i0
            2.4e-3, # tau
        ])
        self.uniform_range = [
            (1e-10, 1e-5), # beta
            (1e-2, 1.),    # epsilon
            (1e-2, 1.),    # rho
            (1e-10, 1e-5), # xi
            (1, 1e6),      # e0
            (1, 1e6),      # i0
            (1e-8, 1e2),   # tau
        ]
        self.random_exponent = np.array([
            1e-12, # beta
            1e-4,  # epsilon
            1e-4,  # rho
            1e-12, # xi
            1e-1,  # e0
            1e-2,  # i0
            1e-10, # tau
        ])


class SeirPde:
    def __init__(self, population=1e7):
        self.population = population


class GraphDiff:
    def __init__(self, n_pref=47):
        self.list = [
            'beta', 
            'epsilon', 
            'rho', 
            'xi', 
            'e0', 
            'i0', 
            'tau', 
            'd', 
        ]
        self.param_num = [
            n_pref, # beta: for each prefecture
            1,      # epsilon
            1,      # rho
            1,      # xi
            1,      # e0
            1,      # i0
            1,      # tau
            1,      # d
        ]
        self.ndim = np.sum(self.param_num)
        self.ini = np.array([
            1.4e-7, # beta
            2.0e-1, # epsilon
            3.8e-1, # rho
            1.0e-6, # xi
            1.0e-5, # e0
            1.0e-5, # i0
            1.0e-7, # tau
            1.0e-7, # d
        ])
        self.uniform_range = [
            (1e-10, 1e-5), # beta
            (1e-2, 1.),    # epsilon
            (1e-2, 1.),    # rho
            (1e-10, 1e-5), # xi
            (1e-6, 1e-3),  # e0
            (1e-6, 1e-3),  # i0
            (1e-8, 1e2),   # tau
            (1e-10, 1e-5), # d
        ]
        self.random_exponent = np.array([
            1e-12, # beta
            1e-4,  # epsilon
            1e-4,  # rho
            1e-12, # xi
            1e-8,  # e0
            1e-8,  # i0
            1e-10, # tau
            1e-12, # d
        ])
        