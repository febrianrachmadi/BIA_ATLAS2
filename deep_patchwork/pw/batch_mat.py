#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:29:31 2021

@author: skibbe
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#from . import tools as tls
from pw import tools as tls
import math
#mp.set_start_method('spawn')
#import pw.tools as pw_tools

class batch_mat():
    
    def __init__(self,
                 dim,
                  dtype = torch.float32,
                  device="cpu"):
       self.dtype = dtype
       self.device = device
       if dim == 2:
            self.identity = torch.tensor([[1,0,0],
                             [0,1,0],
                             [0,0,1]],dtype = torch.float32,device=device)
       else:
            self.identity = torch.tensor([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]],dtype = torch.float32,device=device)
            
       self.dim = dim
       
    def batch_identity(self,batch_size):
        return self.identity[None,:,:].repeat([batch_size,1,1])
    
    def ext_vec(self,x): 
        return torch.cat((tls.tt(x),torch.ones(x.shape[0],1,device=x.device,dtype=x.dtype)),dim=1)[:,:,None]
    
    def transl_mat(self,x):
        dim = x.shape[1]
        batch_size = x.shape[0]
        mat = torch.zeros([batch_size,dim+1,dim+1],device=self.device,dtype=self.dtype)
        #print(mat.shape)
        #print(x.shape)
        mat[:,-1,:dim] = x[:,:]
        for d in range(dim+1):
            mat[:,d,d] = 1
        return mat
    
    def batch_diag(self,vec):
        assert(False)
        mat = self.ext_vec(vec)
        return torch.einsum('exy,bxd->bxy',self.identity[None,:,:],mat)
    
    
    #def batch_mm_diag(self,matA,vecB):
    #    matB = self.ext_vec(vecB.to(device=self.device))
    #    return torch.einsum('bxy,bxd->bxy',matA,matB)
    
    #batch_mm = lambda matA,matB : torch.einsum('bzx,byz->bxy',matA,matB)
    def batch_mm(self,matA,matB):
        return torch.matmul(matA,matB)
        #return torch.einsum('bxz,bzy->bxy',matA,matB)
    #batch_mult_diag = lambda matA,matB :  torch.einsum('bxy,bxd->bxy',matA,extvec(matB))
    
    def batch_m_inverse(self,mat):
        return torch.linalg.inv(matA,matB)  
        #mat_ = torch.zeros(mat.shape,dtype=mat.dtype,device=mat.device)
        #for b in mat.shape[0]:
        #    mat_[b,...] = torch.inverse(mat[b,...])
        #return mat_
        
    def diag_mat(self,aug_scales):
        num_batches = aug_scales.shape[0]
        assert(aug_scales.shape[1] == self.dim)
        scale_mat = torch.zeros([num_batches,self.dim+1,self.dim+1],dtype = self.dtype,device=self.device)
        for a in range(aug_scales.shape[1]):
            scale_mat[:,a,a] = aug_scales[:,a]
        scale_mat[:,-1,-1] = 1
        return scale_mat
            
            
        
  
    def quat2rot(self,aug_rotations_):
        aug_rotations = aug_rotations_.clone()
        num_batches = aug_rotations.shape[0]
        if aug_rotations.shape[1] == 4:
            rotate = tls.tt(aug_rotations) 
            #rotate[:,:3] = 0
            #rotate[:,0] = 1
            #rotate[:,3] = math.pi/2.0
            rotate[:,:3]= aug_rotations[:,:3]/torch.norm(aug_rotations[:,:3],dim=1, keepdim=True)
            #print(rotate[0,:3])
            qr = torch.cos(rotate[:,3]/2.0)
            sin_ = torch.sin(rotate[:,3]/2.0)
            qi = rotate[:,0] * sin_
            qj = rotate[:,1] * sin_
            qk = rotate[:,2] * sin_
            rot_mat = torch.zeros([num_batches,4,4],dtype = self.dtype,device=self.device)
            rot_mat[:,0,0] = 1.0-2.0*(qj**2+qk**2)
            rot_mat[:,0,1] = 2*(qi*qj+qk*qr)
            rot_mat[:,0,2] = 2*(qi*qk-qj*qr)
            
            rot_mat[:,1,0] = 2*(qi*qj-qk*qr)
            rot_mat[:,1,1] = 1-2*(qi**2+qk**2)
            rot_mat[:,1,2] = 2*(qj*qk+qi*qr)
            
            rot_mat[:,2,0] = 2*(qi*qk+qj*qr)
            rot_mat[:,2,1] = 2*(qj*qk-qi*qr)
            rot_mat[:,2,2] = 1-2*(qi**2+qj**2)
            rot_mat[:,3,3] = 1
        else:
            rotate = torch.squeeze(tls.tt(aug_rotations))
            rot_mat = torch.zeros([num_batches,3,3],dtype = self.dtype,device=self.device)
            rot_mat[:,0,0] = torch.cos(rotate)
            rot_mat[:,0,1] = torch.sin(rotate)
            rot_mat[:,1,0] = -rot_mat[:,0,1] 
            rot_mat[:,1,1] = rot_mat[:,0,0]
            rot_mat[:,2,2] = 1
            #rot_mat = torch.tensor([[math.cos(rotate),math.sin(rotate),0],[-math.sin(rotate),math.cos(rotate),0],[0,0,1]],dtype = torch.float32,device=aug_device)
    
        return rot_mat
        
