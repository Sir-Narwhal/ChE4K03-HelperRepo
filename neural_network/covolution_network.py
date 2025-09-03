
from typing import Dict, Tuple, List
import pandas as pd 
import numpy as np
import torch as t 
import random as r
import logging

class RateLawEstimation(t.nn.Module): 
    def __init__(self): 
        super(RateLawEstimation, self).__init__() 
        self.net = t.nn.Sequential(
            t.nn.Linear(2, 128),   # First Layer
            t.nn.ReLU(),           # First Activation Fn
            t.nn.Linear(128, 128), # Second Layer
            t.nn.ReLU(),           # Second Activation Fn
            t.nn.Linear(128, 1)    # Last Layer
        )
    def forward(self, C: t.Tensor) -> t.Tensor:  
        r1 = self.net(C)
        return r1


def get_data(csv_source:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    _df = pd.read_csv(csv_source) 
    c1 = _df.loc[:,"C_A1"].to_numpy()
    c2 = _df.loc[:,"C_A2"].to_numpy()
    r1 = _df.loc[:,"r_A1"].to_numpy() 
    return c1, c2, r1


def train_ratelaw_estimation(csv_source:str)->RateLawEstimation: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    c1, c2, r1 = get_data(csv_source) 

    r1 = t.tensor(r1, 
                  dtype=t.float32).view(-1, 1) 
    C = t.tensor(np.stack((c1.astype(np.float32), c2.astype(np.float32))).T, 
                 dtype=t.float32)        # Must convert to t.Tensor

    model = RateLawEstimation() 
    criterion = t.nn.MSELoss()      # Mean Squared Error to minimize error and set parameters 
                                            # within CNN 
    optimizer = t.optim.Adam(model.parameters(), lr=0.01) 
    
    epochs = 500 
    for epoch in range(epochs): 
        model.train()
        optimizer.zero_grad()
        output = model(C) 
        loss = criterion(output, r1) 
        loss.backward()
        optimizer.step() 

        if epoch % 100 == 0: 
            logging.info(f"Epoch {epoch}, Loss:\t{loss.item():.4f}") 
    
    return model 

def estimate_model(model: RateLawEstimation, Cnew: Tuple[np.ndarray, np.ndarray]): 
    (c1, c2) = Cnew
    Cnew = t.tensor(np.stack((c1.astype(np.float32), c2.astype(np.float32))).T, 
                 dtype=t.float32)
    model.eval() 
    with t.no_grad(): 
        predictions = model(Cnew) 
    return predictions.numpy().T 


if __name__ == "__main__": 
    csv_source = r'.\data\K_0.2_dataset.csv'
    trained_model = train_ratelaw_estimation(csv_source) 
    
    # estimate_model(trained_model)

