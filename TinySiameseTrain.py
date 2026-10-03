import csv
def get_lines(fileName,lines):
    with open(fileName, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            lines.append(row)
    return lines

features_train=list()
features_train=get_lines('csv_train3.csv',features_train)
label_train=list()
label_train=get_lines('csv_trainLabels3.csv',label_train)

featuresTrain=[]
for line in features_train:
  vec=[]
  for val in line:
    vec.append(float(val))
  featuresTrain.append(vec)

labelTrain=[]
for line in label_train:
    val=line[0].replace('[', '')
    val=val.replace(']', '')
    labelTrain.append(int(val))

# importing the required libraries
import os
import glob
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import cv2
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
import os
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
from PIL import Image

# creating the model class
class SiameseSkipAC(nn.Module):

    # initializing the model
    def __init__(self):
        super(SiameseSkipAC, self).__init__()
        self.dimen = 224
        self.linear = nn.Sequential(
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 4096), nn.Sigmoid()
        )
        self.skip_norm = nn.LayerNorm(4096)      # chuẩn hóa raw diff
        self.out = nn.Sequential(
            nn.Linear(12288, 1), nn.Sigmoid()    # 8192 → 12288
        )

    # passing image into the model to get output
    def forward_one(self, x):                    # ← Phương án A
        residual = x.float()
        out = self.linear(x.float())
        return out + residual

    # passing image to get model output
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        out1 = F.normalize(out1, p=2, dim=1)
        out2 = F.normalize(out2, p=2, dim=1)

        raw_diff = self.skip_norm(               # ← Phương án C
            torch.abs(x1.float() - x2.float())
        )

        emb = torch.cat((
            torch.abs(out1 - out2),   # encoded diff
            out1 * out2,              # encoded similarity
            raw_diff                  # raw diff (skip)
        ), dim=1)
        return self.out(emb)
# instantiating the class
net = SiameseSkipAC()
#net.load_state_dict(torch.load("model/torch1DVGG16-120.model"))
# setting optimizer and loss function
optimizer = Adam(net.parameters(), lr=0.00003)
criterion = BCELoss()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print('parameters')
print(count_parameters(net))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
paramNb=count_parameters(net)
print(paramNb)

# if GPU is available, then put the model into the GPU for faster processing
if torch.cuda.is_available():
  net = net.cuda()
  criterion = criterion.cuda()
# function to get unique values
def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    #for x in unique_list:
    #print(len(unique_list))
    return unique_list
unique_list=unique(labelTrain)
print(len(labelTrain))

import random
from random import randrange

# defining the number of epochs
n_epochs = 240
# empty list to store training losses
train_losses = []
# empty list to store validation losses
nbr=len(labelTrain)
labels_train_Index=[]
for i in range(nbr):
  labels_train_Index.append(i)
for epoch in range(n_epochs):
  train_losses = []
  truth_labels=[]
  scores=[]
  net.train()
  tr_loss = 0
  loss_train = 1
  #for NumberImage in enumerate(tqdm(range(18))):
  for user1 in enumerate(tqdm(unique_list)):
     # print(user1[1])
    # clearing the Gradients of the model parameters
      optimizer.zero_grad()
      x=[]
      NotX=[]

      i=-1
      j=0
      for selectUser in labelTrain:
        i=i+1
        if selectUser == user1[1]:
          x.append(featuresTrain[i])
      random.shuffle(x)
      
      i=-1
      for selectUser in labelTrain:
        i=i+1
        if selectUser != user1[1]:
          NotX.append(featuresTrain[i])
      
      numberVectors=len(x)
      number=0
      
      while ( (len(x)-number) > 32 ):
        x1=[]
        x2=[]
        y=[]
        for numV in range(number, number+32):
          x1.append(x[numV])
          x2.append(x[numV])
        
        number = number + 32
        
        numberSamp=len(x2)
        i=0
        for j in range(numberSamp):
          x1.append(x1[i])
          i=i+1
        random.shuffle(x1)
      
      #List4Comp = random.sample(labels_train, numberSamp)
        random_featuresV = random.sample(NotX,32)
        
        for compV in random_featuresV:
          x2.append(compV)

        for i in range(numberSamp):
          y.append(1)
          truth_labels.append(1)
        for i in range(numberSamp):
          y.append(0)
          truth_labels.append(0)

        x1 = np.array(x1)
        x1 = x1.reshape(len(x1), 4096)
        x1 = torch.from_numpy(x1)
    
        x2 = np.array(x2)
        x2 = x2.reshape(len(x2), 4096)
        x2  = torch.from_numpy(x2)

        y = np.array(y)
        y = y.reshape(len(y), 1)
        y  = torch.from_numpy(y)
      
        x1, x2, y  = Variable(x1), Variable(x2), Variable(y)
      
      # if GPU available, put data into GPU for faster processing
      # only adding batches into GPU to save GPU RAM
        if torch.cuda.is_available():
          x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
      
        # get model output
        output = net.forward(x1, x2)

        for j in range(len(output)):
            scores.append((output[j][0]).item())
      # calculate loss
      ##################################################
        loss_train = criterion(output.float(), y.float())
        #print(loss_train)
        train_losses.append(loss_train)
        # backpropagate
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
  # save PyTorch model weights
  torch.save(net.state_dict(), "model/torch1DVGG16-200-2.model")
  
  FinalPredictions=[]
  for i in range(len(scores)):
    if scores[i]>=0.5:
      FinalPredictions.append(1)
    else:
      FinalPredictions.append(0)

  mean = torch.mean(torch.stack(train_losses))
  print('Epoch : ',epoch+1, '\t', 'loss :', mean)
  from sklearn import metrics
  print("Accuracy:",metrics.accuracy_score(truth_labels, FinalPredictions)*100)
