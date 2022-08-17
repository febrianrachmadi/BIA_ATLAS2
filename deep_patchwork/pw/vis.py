import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict

import matplotlib.pyplot as plt
#import torch.optim as optim
import math
import random


def plot3D(imgA,imgB,l1,l2,dim,mode="max"):
    if mode == "max":
        img2D_A,_ = torch.max(imgA[0,l1,:,:,:],dim)
        img2D_B,_ = torch.max(imgB[0,l2,:,:,:],dim)
    if mode == "half":
        if dim == 0:
            img2D_A = imgA[0,l1,imgA.shape[2]//2,:,:]
            img2D_B = imgB[0,l2,imgB.shape[2]//2,:,:]       
        if dim == 1:
            img2D_A = imgA[0,l1,:,imgA.shape[3]//2,:]
            img2D_B = imgB[0,l2,:,imgB.shape[3]//2,:]    
        if dim == 2:
            img2D_A = imgA[0,l1,:,:,imgA.shape[4]//2]
            img2D_B = imgB[0,l2,:,:,imgB.shape[4]//2]    
    imgRGB = np.zeros(tuple(img2D_A.shape)+(3,))
    imgRGB[:,:,0] = img2D_A.numpy()         
    imgRGB[:,:,1] = img2D_B.numpy()      
    return imgRGB

 
def sliceplot(imgA_,imgB,d=10):
    imgA = torch.zeros(imgB.shape)
    imgA[:imgA_.shape[0],:imgA_.shape[1],:imgA_.shape[2]] = imgA_
    print(imgA.shape)
    print(imgB.shape)
    f = plt.figure()
    #f.tight_layout()
    ax = f.gca()
    s = imgA.shape[-1]
    ns = s // d
    nr = 1
    nc = ns
    if ns>4:
        nc = 4
        nr = math.ceil(ns / 4.0)
    for a in range(1,ns-1):
        imgA_2D = imgA[:,:,a*d]
        imgB_2D = imgB[:,:,a*d]

        imgRGB = np.zeros(tuple(imgA_2D.shape)+(3,))
        imgRGB[:,:,0] = imgA_2D.numpy()         
        imgRGB[:,:,1] = imgB_2D.numpy()   
        ax = plt.subplot(nr,nc,a)
        
        ax.imshow(imgRGB)
        ax.axis('off')
        

def plot_patch_batch(x,index,offsets,f=None):
       if f is None:
           f = plt.figure()
       
       ax = f.gca()
       ax.axis('off')
       
       num_scales = len(x.patchlist) 
       for a in range(0,num_scales):
           tensor = x.patchlist[a].tensor.cpu().numpy()
           shape = tensor.shape[2:4]
           ax = plt.subplot(2,num_scales,a+1)
           im = ax.imshow(tensor[index,0,:,:])
           ax = plt.subplot(2,num_scales,num_scales+a+1)
           x0 = offsets[a,0]
           y0 = offsets[a,0]
           x1 = x0 +  shape[0]//2**a
           y1 = y0 +  shape[1]//2**a
           im = ax.imshow(tensor[index,0,x0:x1,y0:y1])






def plot_patch(x,offsets,f=None):
       if f is None:
           f = plt.figure()
       
       ax = f.gca()
       ax.axis('off')
       num_scales = x.shape[1] 
       shape = x.shape[2:4] 
       for a in range(0,num_scales):
           #print(num_scales)
           ax = plt.subplot(2,num_scales,a+1)
           im = ax.imshow(x[0,a,:,:])
           ax = plt.subplot(2,num_scales,num_scales+a+1)
           x0 = offsets[a,0]
           y0 = offsets[a,0]
           x1 = x0 +  shape[0]//2**a
           y1 = y0 +  shape[1]//2**a
           #print([x0,y0,x1,y1])
           im = ax.imshow(x[0,a,x0:x1,y0:y1])
           
           
           
           
           
           