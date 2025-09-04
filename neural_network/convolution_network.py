
from typing import Dict, Tuple, List
import pandas as pd 
import numpy as np
import torch as t 
import random as r
import logging

class RateLawEstimation(t.nn.Module): 
    def __init__(self): 
        super(RateLawEstimation, self).__init__() 

        """
        Neural Net Setup: 
            1. x4 Layers
                a. x1 Input (2 nodes)
                b. x2 Hidden (5 nodes each) 
                c. x1 Output (1 node) 
            2. Activation Function: 
                a. ReLU

        Ideas for Neural Net Design Decisions: 
            - Increase the # of nodes for hidden layers 
            - Increase the # of hidden layers 
            - Change Activation Function
        """

        self.net = t.nn.Sequential(
            t.nn.Linear(2, 5),   # First to Second Layer            2 inputs to 5 nodes 
            t.nn.ReLU(),           # First Activation Fn
            t.nn.Linear(5, 5), # Second to Third Layer              5 nodes to 5 nodes 
            t.nn.ReLU(),           # Second Activation Fn
            t.nn.Linear(5, 1)    # Third to Last Layer              5 nodes to 1 output
        )

    def forward(self, C: t.Tensor) -> t.Tensor:  
        r1 = self.net(C)
        return r1


def train_ratelaw_estimation(raw_data: Tuple[np.ndarray, ...]) -> RateLawEstimation: 
    """
    csv_source: str => location of data containing c1, c2, and r1
    """
    c1, c2, r1 = raw_data  

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


def estimate_model(model: RateLawEstimation, Cnew: Tuple[np.ndarray, np.ndarray]) -> np.ndarray: 
    (c1, c2) = Cnew
    Cnew = t.tensor(np.stack((c1.astype(np.float32), c2.astype(np.float32))).T, 
                 dtype=t.float32)
    model.eval() 
    with t.no_grad(): 
        predictions = model(Cnew) 
    return predictions.numpy().T 



