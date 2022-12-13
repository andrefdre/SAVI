#!/usr/bin/env python3

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from model import Model
from dataset import Dataset
from colorama import Fore, Style
from tqdm import tqdm




def main():

    dataset = Dataset(3000)
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=256,shuffle=True)

    # batch_idx=0
    # for xs,ys_ten_labels in loader:
    #     print('batch ' +str(batch_idx) +' has xs,ys of size ' + str(xs.shape))

    #     batch_idx +=1

    print("Created a figure")

    plt.plot(dataset.xs_np,dataset.ys_np_labels,'go',linewidth=1,markersize=0.9)
    plt.show()
   
    # Initialize model
    # Cuda: 0 index of gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model()
    model.to(device)
    print(device)

    # Parameters
    learning_rate=0.01
    maximum_num_epochs = 500
    termination_loss_threshold = 0.8
    criterion = torch.nn.MSELoss()
    optimizer= torch.optim.Adam(model.parameters() , lr=learning_rate)


    ########################
    # Training
    ########################
    idx_epoch = 0

    while True:
        losses = []
        for batch_idx,(xs_ten,ys_ten_labels) in tqdm(enumerate(loader),total=len(loader),desc='Fore.GREEN' + 'Training batches for epoch' + str(idx_epoch) + Style.RESET_ALL):
            xs_ten = xs_ten.to(device)
            ys_ten_labels = ys_ten_labels.to(device)
            # Aplicar a rede para obter os predicted
            ys_ten_predicted =model.forward(xs_ten)
            # Calcular o erro
            loss = criterion(ys_ten_predicted,ys_ten_labels)   
            losses.append(loss.data.item())
            # atualizar o model
            optimizer.zero_grad() # Resets the weights to make sure we are not accumulating
            loss.backward() # propagates the loss error into each neuron
            optimizer.step() # apply the weights

            # Report
            print('Epoch ' +str(idx_epoch) + ' batch ' +str(batch_idx) + ' Loss ' +str(loss.item()))

        # compute the loss for the epoch
        epoch_loss = mean(losses)
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print(' Finished training. Reached maximum number of epochs')
            break
        if epoch_loss < termination_loss_threshold:
            print(' Finished training. Reached loss threshold')
            break
        

        idx_epoch +=1
    
    # Finalization
    ys_ten_predicted =model.forward(dataset.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()
    plt.plot(dataset.xs_np,dataset.ys_np_labels,'go', label='labels')
    plt.plot(dataset.xs_np,ys_np_predicted,'rx', label='predicted')
    plt.show()

if __name__ == '__main__':
    main()

