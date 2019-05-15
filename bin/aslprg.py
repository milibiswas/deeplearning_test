# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:36:46 2019

@author: Mili Biswas (MSc - Computer Sc.)

"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import data_loader_massey as dlm
import data_loader_kaggle as dlk

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##############################
# Type of dataset

DATASET_MESSEY=True
DATASET_KAGGLE=False
CHANNEL_SIZE=[8,30,30,64,128]



# preparing the model

class MLPModel(nn.Module):
    def __init__(self,):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(128*128*3,128),
        nn.LeakyReLU(0.2),
        nn.Linear(128,36))

    def forward(self,input):
        input = input.view(input.size(0),-1)
        ret=self.layers(input)
        return ret



class ConvModel(nn.Module):
    def __init__(self,):
        super(ConvModel,self).__init__()
        self.conv_layer=nn.Sequential(
                nn.Conv2d(3,CHANNEL_SIZE[2],kernel_size=3,padding=0),
                nn.ReLU(),
                nn.Conv2d(CHANNEL_SIZE[2],CHANNEL_SIZE[1],kernel_size=3,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
                nn.Conv2d(CHANNEL_SIZE[1],CHANNEL_SIZE[1],kernel_size=3,padding=0),
                nn.ReLU(),
                nn.Conv2d(CHANNEL_SIZE[1],CHANNEL_SIZE[1],kernel_size=3,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
                nn.Conv2d(CHANNEL_SIZE[1],CHANNEL_SIZE[1],kernel_size=3,padding=0),
                nn.ReLU(),
                nn.Conv2d(CHANNEL_SIZE[1],CHANNEL_SIZE[1],kernel_size=3,padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25)
                )
        self.linear_layer=nn.Sequential(
                nn.Linear(CHANNEL_SIZE[1]*21*21,128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128,26),
                nn.ReLU(),
                nn.Dropout(0.5),
                )
    def forward(self,input):
        conv_out = self.conv_layer(input)
        conv_out_flat = conv_out.view(conv_out.size(0),-1)
        out=self.linear_layer(conv_out_flat)
        return out

# train/ validate the model

def train(model,train_dataloader,loss_fn,optimizer):
    train_loss = 0
    n_correct = 0
    model.train()
    for (images,labels) in train_dataloader:
        images=images.to(device)
        labels=labels.to(device)
        out = model(images)
        loss = loss_fn(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_correct += torch.sum(out.argmax(1)==labels).item()
    accuracy = (100*n_correct)/len(train_dataloader.dataset)
    average_loss = train_loss/len(train_dataloader)
    return average_loss,accuracy

def valid(model,valid_dataloader,loss_fn):
    valid_loss = 0
    n_correct = 0
    with torch.no_grad():
        for (images,labels) in valid_dataloader:
            images=images.to(device)
            labels=labels.to(device)
            out = model(images)
            loss = loss_fn(out,labels)
            valid_loss +=loss.item()
            n_correct += torch.sum(out.argmax(1)==labels).item()
        accuracy = 100*n_correct / len(valid_dataloader.dataset)
        average_loss = valid_loss/len(valid_dataloader)
        return average_loss,accuracy

def test(model,test_dataloader):
    n_correct = 0
    with torch.no_grad():
        for (images,labels) in test_dataloader:
            images=images.to(device)
            labels=labels.to(device)
            out = model(images)
            n_correct += torch.sum(out.argmax(1)==labels).item()
        accuracy = 100*n_correct / len(test_dataloader.dataset)
        return accuracy

def fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,n_epoch):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(n_epoch):
        train_loss,train_accuracy = train(model,train_dataloader,loss_fn,optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_loss,valid_accuracy = valid(model,valid_dataloader,loss_fn)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        print("Epoch:{}, train loss:{}, train accuracy:{}, validation loss:{}, validation accuracy:{}, loss diff:{}".format(epoch,train_loss,train_accuracy,valid_loss,valid_accuracy,abs(train_loss-valid_loss)))
        #print("Epoch:{} the validation loss is {} and the validation accuracy is {}".format(epoch,valid_loss,valid_accuracy))
    return train_losses , train_accuracies,valid_losses,valid_accuracies


if __name__=="__main__":
    
    if DATASET_KAGGLE:
        d=dlk.data_loader_kaggle()
    if DATASET_MESSEY:
        d=dlm.data_loader_messey()
        

    train_dataloader=d.train_dataloader
    valid_dataloader=d.valid_dataloader
    
    #........................................
    model = ConvModel()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    #--------------------------------------------------
    
    t_losses , t_accuracies,v_losses,v_accuracies = fit(model,train_dataloader,valid_dataloader,loss_fn,optimizer,100)


    ##########################
    #   plot function
    ##########################
    def plot (x_axis,y_axis,plotName=None,typeName="train"):
        plt.figure(figsize=(50,50))
        if plotName=="scatter":
            plt.scatter(x_axis,y_axis)
            plt.savefig(plotName+"_"+typeName+".png")
        else:
            plt.plot(x_axis,y_axis)
            plt.savefig("_"+typeName+".png")
    
    plot(range(len(t_losses)),t_losses,None,"trainLoss")
    plot(range(len(t_accuracies)),t_accuracies)
    
    plot(range(len(v_losses)),v_losses,None,"validLoss")
    plot(range(len(v_accuracies)),v_accuracies,None,"validAccuracy")
    #######################################
    #     Test phase
    #######################################
    
    #  Data Loader (Test)
    test_dataloader = d.test_dataloader
    
    test_accuracy=test(model,test_dataloader)
    
    print("======================== Test Accurracy Score ========================")
    print("Accuracy Score : ",test_accuracy)
    print("========================        End           ========================")


