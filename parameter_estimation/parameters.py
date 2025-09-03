import pandas as pd 
import numpy as np 
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit


def get_data(csv_source:str)->Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    _df = pd.read_csv(csv_source) 
    c1 = _df.loc[:,"C_A1"].to_numpy()
    c2 = _df.loc[:,"C_A2"].to_numpy()
    r1 = _df.loc[:,"r_A1"].to_numpy()
    return c1, c2, r1


def rate_law(C: Tuple[float, float], k1: float, k2: float)->float: 
    c1, c2 = C
    return  (k1*c1)/(1+c1+k2*c2)


def fit_data(csv_source:str) -> Tuple[float, float]: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    c1, c2, r1 = get_data(csv_source) 
    params, _ = curve_fit(rate_law, (c1, c2), r1)     # Fit paramaeters here 

    return params # (k1, k2)

if __name__ == "__main__": 
    """
    PUT CODE HERE
    """
    print(f"blank print")