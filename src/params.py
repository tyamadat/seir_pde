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
        self.ini = {
            'beta': 1.4e-7, 
            'epsilon': 2.0e-1, 
            'rho': 3.8e-1, 
            'xi': 6.4e-7, 
            'e0': 7.0e-3*population, 
            'i0': 1.6e-4*population, 
            'tau': 2.4e-3, 
        }
        self.uniform_range = {
            'beta': (1e-10, 1e-5), 
            'epsilon': (1e-2, 1.), 
            'rho': (1e-2, 1.), 
            'xi': (1e-10, 1e-5), 
            'e0': (1, 1e6), 
            'i0': (1, 1e5), 
            'tau': (1e-8, 1e2), 
        }
        self.random_exponent = np.array([
            1e-12, # beta
            1e-4,  # epsilon
            1e-4,  # rho
            1e-12, # xi
            1e-1,  # e0
            1e-2,  # i0
            1e-10, # tau
        ])
        