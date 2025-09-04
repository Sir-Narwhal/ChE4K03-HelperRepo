import os 
import math
import pandas as pd 
import numpy as np 
from typing import Tuple, List
from neural_network.convolution_network import train_ratelaw_estimation, estimate_model
from parameter_estimation.parameters import fit_data, rate_law
import matplotlib.pyplot as plt
import random 


def mean_squared_error(x_estimated: np.ndarray, x_actual: np.ndarray): 
    error = (x_estimated-x_actual).reshape((-1, 1))
    mse = np.matmul(error.T, error)
    return mse


def data_split_function(raw_data: Tuple[np.ndarray, ...], training_split: float = 0.7) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]: 
    """
    raw_data: c1, c2, c3, etc. 
    output -> (training_data, testing_data) 
    training_data: c1, c2, c3, etc. 
    testing_data: c1, c2, c3, etc.
    """
    length_of_data = raw_data[0].shape[0] 
    
    rand_training_idx = random.sample(range(length_of_data), math.ceil(length_of_data*training_split))
    rand_testing_idx = list(set(range(length_of_data)) - set(rand_training_idx))

    training_data = [] 
    testing_data = []

    for data_remenant in raw_data:
        training_data.append(data_remenant[rand_training_idx]) 
        testing_data.append(data_remenant[rand_testing_idx]) 
    
    return tuple(training_data), tuple(testing_data) 


def get_data(csv_source:str) -> Tuple[np.ndarray, ... ]: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    _df = pd.read_csv(csv_source) 
    c1 = _df.loc[:,"C_A1"].to_numpy()
    c2 = _df.loc[:,"C_A2"].to_numpy()
    r1 = _df.loc[:,"r_A1"].to_numpy() 
    return c1, c2, r1


if __name__ == "__main__": 
    """
    PUT YOUR CODE HERE: See example below for help. 
    """

    for eachfile in os.listdir(r"./data"): 
        
        if eachfile.endswith(".csv"): 
            source_path = os.path.abspath(os.path.join("./data", eachfile)) 
            print(f"source_path:\t{source_path}") 

            c1, c2, r1 = get_data(source_path)

            training_set, testing_set = data_split_function((c1, c2, r1), 0.7) 

            (c1_testing, c2_testing, r1_testing) = testing_set
            (c1_training, c2_training, r1_training) = training_set 

            # 1. Estimate Parameters 
            estimated_params = fit_data(training_set) 
            estimated_r1_param = rate_law((c1_testing, c2_testing), *estimated_params) 
            mse_param = mean_squared_error(estimated_r1_param, r1_testing) 
            for idx, each_param in enumerate(estimated_params): 
                print(f"k{idx+1}: {each_param:4f}", end="\t")
            print(f"MSE Param: {mse_param[0][0]:8f}")  

            # 2. Estimate with Nonlinear Regression (CNN)
            trained_model = train_ratelaw_estimation(training_set)
            estimated_r1_cnn = estimate_model(trained_model, (c1_testing, c2_testing)) 
            mse_cnn = mean_squared_error(estimated_r1_cnn, r1_testing) 
            print(f"MSE CNN: {mse_cnn[0][0]:8f}") 

    