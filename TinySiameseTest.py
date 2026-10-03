import csv
def get_lines(fileName,lines):
    with open(fileName, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            lines.append(row)
    return lines


features_probe=list()
features_probe=get_lines('csv_Test.csv',features_probe)
label_probe=list()
label_probe=get_lines('csv_LabelsTest.csv',label_probe)

    
featuresProbe=[]
for line in features_probe:
  vec=[]
  for val in line:
    vec.append(float(val))
  featuresProbe.append(vec)

labelProbe=[]
for line in label_probe:
    val=line[0].replace('[', '')
    val=val.replace(']', '')
    labelProbe.append(int(val))

# importing the required libraries
import os
import glob
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_curve

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
from sklearn.metrics import confusion_matrix

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
net.load_state_dict(torch.load("model/torch1D_AlexnetPbmltEucHam.model"))
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
unique_list=unique(labelProbe)
print(len(unique_list))

import random
from random import randrange

# defining the number of epochs
n_epochs = 10

# empty list to store validation losses
nbr=len(labelProbe)
labels_train_Index=[]
precisions=[]
recalls=[]
fscores=[]
accuracies=[]
NPVs=0
FPRs=0
FDRs=0
FNRs=0
maxprecision=0
fpr = dict()
tpr = dict()
net.eval()
for i in range(nbr):
  labels_train_Index.append(i)
with torch.no_grad():
  for epoch in range(n_epochs):
    truth_labels=[]
    scores=[]
    times=0
    nbtimes=0
    for user1 in enumerate(tqdm(unique_list)):
      #print(user1[1])
    # clearing the Gradients of the model parameters
      x1=[]
      x2=[]
      y=[]
      i=-1
      j=0
      
      for selectUser in labelProbe:
        i=i+1
        if selectUser == user1[1]:
          j=j+1
          if j<=12:
            x1.append(featuresProbe[i])
      j=0
      i=-1
      kl=0
      for selectUser in labelProbe:
        i=i+1
        if selectUser == user1[1]:
          #kl=kl+1
          #if kl>=3:
            j=j+1
            if j<= len(x1):
              x2.append(featuresProbe[i])
      
      numberSamp=len(x1)

      i=0
      for j in range(numberSamp):
        x1.append(x1[i])
        i=i+1
      random.shuffle(x1)
      
      #List4Comp = random.sample(labels_train, numberSamp)
      random_index = random.sample(labels_train_Index,numberSamp)
      List4Comp=[]
      for i in range(numberSamp):
        List4Comp.append(labelProbe[random_index[i]])
      
      while user1[1] in List4Comp:
        #List4Comp = random.sample(labels_train, numberSamp)
        random_index = random.sample(labels_train_Index,numberSamp)
        List4Comp=[]
        for i in range(numberSamp):
          List4Comp.append(labelProbe[random_index[i]])
      
      for compIndex in random_index:
        
        x2.append(featuresProbe[compIndex])

      for i in range(numberSamp):
        truth_labels.append(1)
      for i in range(numberSamp):
        truth_labels.append(0)
      
      
      x1 = np.array(x1)
      x1 = x1.reshape(len(x1), 4096)
      x1 = torch.from_numpy(x1)
    
      x2 = np.array(x2)
      x2 = x2.reshape(len(x2), 4096)
      x2  = torch.from_numpy(x2)
      
      x1, x2  = Variable(x1), Variable(x2)
      
      x11 = np.array(x1[0])
      x11 = x11.reshape(1, 4096)
      x11 = torch.from_numpy(x11)
    
      x22 = np.array(x2[0])
      x22 = x22.reshape(1, 4096)
      x22  = torch.from_numpy(x22)
      
      x11, x22  = Variable(x11), Variable(x22)
        # get model output
      start_time = time.time()
      output = net.forward(x11, x22)
      times= (time.time() - start_time)+times
      nbtimes=nbtimes+1
      
      output = net.forward(x1, x2)
      for j in range(len(output)):
          scores.append((output[j][0]).item())
      
    FinalPredictions=[]
    for i in range(len(scores)):
      if scores[i]>=0.5:
        FinalPredictions.append(1)
      else:
        FinalPredictions.append(0)
    
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, fscore, support = score(truth_labels, FinalPredictions)
    TN, FP, FN, TP = confusion_matrix(truth_labels, FinalPredictions, labels=[0, 1]).ravel()
    NPV = TN / (TN + FN) 
    FPR = FP / (FP + TN) 
    FDR = FP / (FP + TP)
    FNR = FN / (FN + TP)
    NPVs=NPVs+NPV
    FPRs=FPRs+FPR
    FDRs=FDRs+FDR
    FNRs=FNRs+FNR

    print('precision: {}'.format(np.mean(precision)))
    print('recall: {}'.format(np.mean(recall)))
    print('fscore: {}'.format(np.mean(fscore)))
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(truth_labels, FinalPredictions)*100)
    print("--- %s seconds ---" % (times/nbtimes))
    precisions.append(np.mean(precision))
    recalls.append(np.mean(recall))
    fscores.append(np.mean(fscore))
    accuracies.append(metrics.accuracy_score(truth_labels, FinalPredictions))
    
    if maxprecision<np.mean(precision):
      maxprecision=np.mean(precision)
      # Compute ROC curve and ROC area for each class
      fpr = dict()
      tpr = dict()
      #roc_auc = dict()
      fpr, tpr, _ = roc_curve(truth_labels, scores)

  print('NPV', str(NPVs/10))
  print('FPR', str(FPRs/10))
  print('FDR', str(FDRs/10))
  print('FNR', str(FNRs/10))
  print('precisions: {}'.format(np.mean(precisions)))
  print('recalls: {}'.format(np.mean(recalls)))
  print('fscores: {}'.format(np.mean(fscores)))
  print("Accuracies: {}".format(np.mean(accuracies)*100))
  
  with open('fpr1D.pickle', 'wb') as handle:
    pickle.dump(fpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('tpr1D.pickle', 'wb') as handle:
    pickle.dump(tpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  plt.figure()
  plt.plot(fpr,tpr,color="darkorange",label="ROC curve ")
    #plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC curves")
    #plt.legend(loc="lower right")
  plt.show()
