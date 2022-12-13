#!/usr/bin/env python3

import torch


# Definition of the model. For now a 1 neural network
class Model(torch.nn.Module):
    def __init__(self):
        # Define the neural network
        super().__init__()
        self.layer1 = torch.nn.Linear(1,1)


    def forward(self,xs):
        ys=self.layer1(xs)

        return ys
