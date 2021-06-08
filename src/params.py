# models.py
# Contact: Tetsuya Yamada <tetsu.steel.iron.1222@gmail.com>

"""

"""

import numpy as np


class SeirOde:
    def __init__(self, population=1e7):
        self.population = population
        self.ndim = 7
        self.list = [
            'beta', 
            'epsilon', 
            'rho', 
            'xi', 
            'e0', 
            'i0', 
            'tau', 
        ]
        self.ini = np.array([
            1.4e-7, # beta
            2.0e-1, # epsilon
            3.8e-1, # rho
            6.4e-7, # xi
            5.5e-2*population, # e0
            1.2e-2*population, # i0
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
        