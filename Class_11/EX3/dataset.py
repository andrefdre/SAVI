#!/usr/bin/env python3

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self,num_points,f=0.9,a=14,sigma=3):
        self.num_points = num_points
        # Generate xs
        self.xs_np=np.random.rand(num_points,1)*20 -10 # to get values between -10 and 10
        self.xs_np = self.xs_np.astype(np.float32)
        # compute ys
        self.ys_np_labels = np.sin(f*self.xs_np) * a
        self.ys_np_labels += np.random.normal(loc=0.0,scale=sigma,size=(num_points,1))

        # Convert to torch tensor
        self.xs_ten = torch.from_numpy(self.xs_np)
        self.ys_ten_labels = torch.from_numpy(self.ys_np_labels)

    def __getitem__(self,index): # returns a specific x,y of the datasets
        return self.xs_ten[index], self.ys_ten_labels[index]

    def __len__(self):
        return self.num_points