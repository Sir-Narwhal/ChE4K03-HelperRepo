import os 
import pandas as pd 
import numpy as np 
from typing import Tuple, List
from neural_network.covolution_network import train_ratelaw_estimation, estimate_model
from parameter_estimation.parameters import fit_data, rate_law, get_data
import matplotlib.pyplot as plt

def mean_squared_error(x_estimated: np.ndarray, x_actual: np.ndarray): 
    error = (x_estimated-x_actual).reshape((-1, 1))
    mse = np.matmul(error.T, error)
    return mse


if __name__ == "__main__": 
    """
    PUT CODE HERE 
    """

    for eachfile in os.listdir(r"./data"): 
        
        if eachfile.endswith(".csv"): 
            source_path = os.path.abspath(os.path.join("./data", eachfile)) 
            print(f"source_path:\t{source_path}") 

            c1, c2, r1 = get_data(source_path)

            # 1. Estimate Parameters 
            (k1, k2) = fit_data(source_path) 
            estimated_r1_param = rate_law((c1, c2), k1, k2) 
            mse_param = mean_squared_error(estimated_r1_param, r1) 
            print(f"k1: {k1:4f}\tk2: {k2:4f}")
            print(f"MSE Param: {mse_param[0][0]:8f}") 

            # 2. Estimate with Nonlinear Regression (CNN)
            trained_model = train_ratelaw_estimation(source_path)
            estimated_r1_cnn = estimate_model(trained_model, (c1, c2)) 
            mse_cnn = mean_squared_error(estimated_r1_cnn, r1) 
            print(f"MSE CNN: {mse_cnn[0][0]:8f}") 

    