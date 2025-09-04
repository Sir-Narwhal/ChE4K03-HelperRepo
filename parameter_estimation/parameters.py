import pandas as pd 
import numpy as np 
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit


def rate_law(C: Tuple[float, float], k1: float, k2: float, k3: float, k4: float, k5: float) -> float: 
    """
    Change this to your Equation for Parameter Estimation. 
    Params: k1, k2, k3, k4, k5 (type: float)
    """
    c1, c2 = C
    r1 = k1*c1 + k2*c2 + k3*c1*c2 + k4*c1**2 + k5*c2**2
    return  r1


def fit_data(raw_data: Tuple[np.ndarray, ...]) -> Tuple[float, ...]: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    c1, c2, r1 = raw_data # => Use the function or import on your own 
    params, _ = curve_fit(rate_law, (c1, c2), r1)     # Fit paramaeters here as an optimization problem 
    return params # (k1, k2, etc.)


if __name__ == "__main__": 
    """p
    PUT CODE HERE
    """
    print(f"blank print")