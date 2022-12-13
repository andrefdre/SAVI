#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
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


def main():
    plt.figure()
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.grid()
    plt.xlabel('x')
    plt.xlabel('y')

    file = open('pts.pk1', 'rb')

    # dump information to that file
    pts = pickle.load(file)
    file.close()

    print("Created a figure")
    
    plt.plot(pts['xs'],pts['ys'],'sk',linewidth=2,markersize=12)

    # Convert the pts to np array
    xs_np= np.array(pts['xs'],dtype=np.float32).reshape(-1,1)
    ys_np_labels= np.array(pts['ys'],dtype=np.float32).reshape(-1,1)
  
    # Initialize model
    # Cuda: 0 index of gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model()
    model.to(device)
    print(device)

    # Parameters
    learning_rate=0.01
    maximum_num_epochs = 50
    criterion = torch.nn.MSELoss()
    optimizer= torch.optim.SGD(model.parameters() , lr=learning_rate)

    plt.plot(xs_np,ys_np_labels,'go', label='labels')
    plt.legend(loc='best')
    plt.show()
    ########################
    # Training
    ########################
    idx_epoch = 0

    while True:
        xs_ten = torch.from_numpy(xs_np).to(device)
        ys_ten_labels = torch.from_numpy(ys_np_labels).to(device)
        # Aplicar a rede para obter os predicted
        ys_ten_predicted =model.forward(xs_ten)
        # Calcular o erro
        loss = criterion(ys_ten_predicted,ys_ten_labels)   
        # atualizar o model
        optimizer.zero_grad() # Resets the weights to make sure we are not accumulating
        loss.backward() # propagates the loss error into each neuron
        optimizer.step() # apply the weights

        # Report
        print('Epoch ' +str(idx_epoch) + ' Loss ' +str(loss.item()))

        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print(' Finished training. Reached maximum number of epochs')
            break

        idx_epoch +=1
    
    # Finalization
    ys_ten_predicted =model.forward(xs_ten)
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()
    plt.plot(xs_np,ys_np_labels,'go', label='labels')
    plt.plot(xs_np,ys_np_predicted,'rx', label='predicted')
    plt.show()

if __name__ == '__main__':
    main()

