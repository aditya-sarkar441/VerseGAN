import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import os 
import numpy as np
import pandas as pd

import glob
import random

from torch.autograd import Variable
from torch.autograd import Function
from torch import optim

# calling user defined functions
from discriminator import discriminator
from generator import generator
import data 

# initialize loss functions 
def gen_loss(e1,e2):
    loss_wssl = nn.CosineSimilarity()
    m = nn.Sigmoid()
    l1 = (loss_wssl(e1,e2))
    return(m(l1))
    
def disc_loss(v, o):
    l2 = 
    
    
    
if __name__ == '__main__':
    # initialize training
    model1 = generator()
    model2 = generator()

    model1.cuda()
    model2.cuda()

    model = discriminator(model1,model2)
    model.cuda()
    
    optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum= 0.9)

    # custom losses
    loss_lang = torch.nn.CrossEntropyLoss(reduction='mean')  

    loss_lang.cuda()

    n_epoch = 20
    manual_seed = random.randint(1,10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    files_list=[]
    
    # Train files in csv format
    folders = glob.glob('/u/home/a/asarkar/scratch/VerseGAN/bnf_embeds/*')  
    for folder in folders:
        for f in glob.glob(folder+'/*.csv'):
            files_list.append(f)

    l = len(files_list)
    random.shuffle(files_list)
    print('Total Training files: ',l)
    
    # finished initialization
    print('#'*30)
    
    model1.train()
    model2.train()
    model.train()
    
    for e in range(n_epoch):
        cost = 0.
        random.shuffle(files_list)
        
        # number of files completed in the epoch
        i=0  
        for fn in files_list:    
            #print(fn)
            df = pd.read_csv(fn,encoding='utf-16',usecols=list(range(0,80)))
            data = df.astype(np.float32)
            X = np.array(data) 
            N,D=X.shape

            if N>look_back2:
                model.zero_grad()

                XX1,XX2,YY1,Yint = lstm_data(fn)
                XNP=np.array(XX1)
                if(np.isnan(np.sum(XNP))):
                    continue

                XNP=np.array(XX2)
                if(np.isnan(np.sum(XNP))):
                    continue

                i = i+1
                XX1 = np.swapaxes(XX1,0,1)
                XX2 = np.swapaxes(XX2,0,1)
                X1 = Variable(XX1,requires_grad=False).cuda()
                Y1 = Variable(YY1,requires_grad=False).cuda()
                X2 = Variable(XX2,requires_grad=False).cuda()


                fl,e1,e2,u = model.forward(X1,X2)   
                
                err_l = loss_lang(fl,Y1)
                err_wssl = abs(loss_wssl(e1,e2))               
                T_err = err_l + 0.25*err_wssl   # alpha = 0.25 in this case. Change it to get better output       

                T_err.backward()
                optimizer.step()
                cost = cost + T_err.item()

                print("ZWSSL5:  epoch "+str(e+1)+" completed files  "+str(i)+"/"+str(l)+" Loss= %.3f"%(cost/i))

        # Save model after every epoch
        path = "/home/iit/Muralikrishna/shantanu_2/e1_e2_wssl/e1_e2_wssl_1/models/ZWSSL5_e"+str(e+1)+".pth" 
        torch.save(model.state_dict(),path)