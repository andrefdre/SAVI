#!/usr/bin/env python3

import glob
import random

import torch
from dataset import Dataset
from model import Cnn
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style

def main():

    dataset_path= './dataset/train'
    image_filenames = glob.glob(dataset_path + '/*.jpg')
    
    tensor_to_pill_image = transforms.Compose([
            transforms.ToPILImage()
    ])
    # Sample ony a few images for develop
    image_filenames = random.sample(image_filenames,k=700)
    train_image_filenames,test_image_filenames = train_test_split(image_filenames,test_size=0.2)
    #print(len(train_image_filenames),len(test_image_filenames))

    dataset_train = Dataset(train_image_filenames)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=256,shuffle=True)
    for image_t ,label_t in loader_train:
        #print(image_t.shape) 
        num_images = image_t.shape[0]
        image_idxs=random.sample(range(0,num_images),k=25)
        #print(image_idxs)

        fig = plt.figure()
        for plot_idx,image_idx in enumerate(image_idxs,start=1):
            image = tensor_to_pill_image(image_t[image_idx,:,:,:]) # get images idx image_idx
            ax = fig.add_subplot(5,5,plot_idx) ## Create subplot
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            label =label_t[image_idx].data.item()
            class_name = 'dog' if label == 0 else 'cat'
            ax.set_xlabel(class_name)
            plt.imshow(image)

        #plt.show()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Cnn()
    model.to(device)

    # Parameters
    learning_rate=0.001
    maximum_num_epochs = 500
    termination_loss_threshold = 0.05
    criterion = torch.nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters() , lr=learning_rate)

    #TODO ver a loss function
    ########################
    # Training
    ########################
    idx_epoch = 0
    epoch_losses=[] 
    while True:
        losses = []
        for batch_idx,(image_t,label_t) in tqdm(enumerate(loader_train),total=len(loader_train),desc='Fore.GREEN' + 'Training batches for epoch' + str(idx_epoch) + Style.RESET_ALL):
            image_t = image_t.to(device)
            label_t = label_t.to(device)
            # Aplicar a rede para obter os predicted
            label_t_predicted =model.forward(image_t)
            # Calcular o erro
            loss = loss_function(label_t_predicted,label_t)   
            losses.append(loss.data.item())
            # atualizar o model
            optimizer.zero_grad() # Resets the weights to make sure we are not accumulating
            loss.backward() # propagates the loss error into each neuron
            optimizer.step() # apply the weights

            # Report
            print('Epoch ' +str(idx_epoch) + ' batch ' +str(batch_idx) + ' Loss ' +str(loss.item()))

        # compute the loss for the epoch
        epoch_loss = mean(losses)
        epoch_losses.append(epoch_loss)
        # Termination criteria
        if idx_epoch > maximum_num_epochs:
            print(' Finished training. Reached maximum number of epochs')
            break
        if epoch_loss < termination_loss_threshold:
            print(' Finished training. Reached loss threshold')
            break
        

        idx_epoch +=1

if __name__ == '__main__':
    main()