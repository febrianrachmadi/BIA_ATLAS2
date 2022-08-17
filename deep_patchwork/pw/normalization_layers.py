#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 22:32:32 2021

@author: skibbe
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict

import matplotlib.pyplot as plt
#import torch.optim as optim
import math
import random


from . import tools as tls
from . import data as pw_data
from .units import layersND


class hist_layer_marco(nn.Module):
    def __init__(self,settings={}):#,dim,scfac=0.2,init='ct',normalize=True):
        super().__init__()
        defaults = {
            "scfac":0.2,
            "dim":3,
            "scale":1,
            "init":"ct",
            "normalize":False
        }
        for key, arg in settings.items():
            defaults[key] = arg
        self.param = defaults    
        
        self.init = self.param["init"]
        self.scfac =  self.param["scfac"]
        self.dim = self.param["dim"]
        self.normalize = self.param["normalize"]
        
        if isinstance(self.init,str):
            if self.init == 'ct':        
                self.centers = np.array([-1000,-500,-100,-50,-25,0,25,50,100,500,1000])
            else:
                assert False, "init not defined for histolayer"
        else:
            self.centers = np.array(self.init)

        width = np.convolve(self.centers,[-1,0,1],mode='valid')
        self.width = np.abs(np.append(np.append([width[0]],width),[width[-1]]))*self.scfac
        self.centers = -self.centers/self.width
        fout = len(self.centers)
        
        self.conv = layersND.Conv(self.dim,in_channels=1,out_channels=fout,kernel_size=1,bias=True)
        with torch.no_grad():
            wshape = self.conv.weight.shape
            bshape = self.conv.bias.shape
            self.conv.weight.copy_(torch.reshape(torch.from_numpy(1.0/self.width).float(),wshape))
            self.conv.bias.copy_(torch.reshape(torch.from_numpy(self.centers).float(),bshape))
        
    def forward(self,inputs):
        x = self.conv(inputs)
        eps = 0.0000000001
        ch = 1.0/(torch.cosh(x))
        if self.normalize:
            ch = (ch+eps) / (ch.sum(dim=1,keepdims=True)+eps)
            return ch
        else:
            return ch
        
    def tensorboard(self,writer,name,iteration):
        my_plots_w = {}
        my_plots_b = {}

        count = 0
        #for hu in self.param["hu"]:
        for hu in range(self.conv.weight.view(-1).shape[0]):
            my_plots_w[str(hu)] = self.conv.weight.view(-1)[count].detach().cpu().numpy()
            my_plots_b[str(hu)] = self.conv.bias.view(-1)[count].detach().cpu().numpy()
            count += 1
            #hu_range = torch.tensor(self.param["hu"][hu])
        #print(my_plots_min)
        writer.add_scalars(name+"/hu_bias", my_plots_b, iteration)
        writer.add_scalars(name+"/hu_weights", my_plots_w, iteration)



class hist_layer(nn.Module):
    def __init__(self,settings={}):
        super().__init__()
        defaults = {
            "scfac":0.4,
            "dim":3,
            "scale":1,
            "init":"ct",
            "normalize":False #True
        }
        for key, arg in settings.items():
            defaults[key] = arg
        self.param = defaults    
        
        self.init = self.param["init"]
        self.scfac =  self.param["scfac"]
        self.dim = self.param["dim"]
        self.normalize = self.param["normalize"]
        
        if isinstance(self.init,str):
            if self.init == 'ct':        
                self.centers = tls.tt(np.array([-1000,-500,-100,-50,-25,0,25,50,100,500,1000]),dtype=torch.float32)
            else:
                assert False, "init not defined for histolayer"
        else:
            self.centers = tls.tt(np.array(self.init),dtype=torch.float32)

        width = np.convolve(self.centers,[-1,0,1],mode='valid')
        self.width = tls.tt(np.abs(np.append(np.append([width[0]],width),[width[-1]]))*self.scfac,dtype=torch.float32)
        #self.centers = -self.centers/self.width
        #fout = len(self.centers)
        
        
        self.width_t = torch.nn.parameter.Parameter(1.0/self.width.reshape([1,len(self.width),1,1,1]), requires_grad=True)# torch.autograd.Variable(mins, requires_grad=True)
        self.centers_t = torch.nn.parameter.Parameter(self.centers.reshape([1,len(self.centers),1,1,1]), requires_grad=True)# torch.autograd.Variable(maxs, requires_grad=True)
    
    def forward(self,inputs):
       # print("histlayer in {} {}".format(inputs.min(),inputs.max()))
        ch = (inputs - self.centers_t)*self.width_t
        
        #return -(ch**2*self.width_t)
    
        ch = torch.exp(-(ch**2))
        #print("histlayer out {} {}".format(ch.min(),ch.max()))
        
        eps = 0.0000000001
        if self.normalize:
            ch = (ch+eps) / (ch.sum(dim=1,keepdims=True)+eps)
            #ch = (ch) / (ch.sum(dim=1,keepdims=True))
            
            return ch
        else:
            return ch
        
    def tensorboard(self,writer,name,iteration):
        my_plots_w = {}
        my_plots_b = {}

        count = 0
        #for hu in self.param["hu"]:
        for hu in range(self.centers_t.view(-1).shape[0]):
            my_plots_w[str(hu)] = self.width_t.view(-1)[count].detach().cpu().numpy()
            my_plots_b[str(hu)] = self.centers_t.view(-1)[count].detach().cpu().numpy()
            count += 1
            #hu_range = torch.tensor(self.param["hu"][hu])
        #print(my_plots_min)
        writer.add_scalars(name+"/hu_bias", my_plots_b, iteration)
        writer.add_scalars(name+"/hu_weights", my_plots_w, iteration)

class clamp_layer(nn.Module):
    def __init__(self,dim,mins,maxs):
        super().__init__()
        self.start_min = mins
        self.start_max = maxs
        
        #self.norm = layersND.InstanceNorm(dim,len(mins))
        self.relu_min = nn.LeakyReLU()
        self.relu_max = nn.LeakyReLU()        
        self.bias_min = torch.nn.parameter.Parameter(mins.reshape([1,len(mins),1,1,1]), requires_grad=True)# torch.autograd.Variable(mins, requires_grad=True)
        self.bias_max = torch.nn.parameter.Parameter(maxs.reshape([1,len(maxs),1,1,1]), requires_grad=True)# torch.autograd.Variable(maxs, requires_grad=True)
    def forward(self,inputs):
        bmin = self.bias_min
        bmax = self.bias_max
        d = (bmax-bmin)
        
        return (self.relu_max((-(self.relu_min(inputs-bmin)))+d)-d/2)/d
        
    
    #return (self.relu_max((-(self.relu_min(inputs-bmin)))+(bmax-bmin))-(bmax-bmin)/2)/(bmax-bmin)
        #return (F.relu((-(F.relu(t-interv[0])))+(interv[1]-interv[0]))-(interv[1]-interv[0])/2)/(interv[1]-interv[0])
    
    def tensorboard(self,writer,name,iteration):
        my_plots_min = {}
        my_plots_max = {}

        count = 0
        #for hu in self.param["hu"]:
        for hu in range(self.bias_min.shape[1]):
            my_plots_min[str(hu)] = self.bias_min.view(-1)[count].detach().cpu().numpy() - self.start_min[count].cpu().numpy() 
            my_plots_max[str(hu)] = self.bias_max.view(-1)[count].detach().cpu().numpy() - self.start_max[count].cpu().numpy() 
            count += 1
            #hu_range = torch.tensor(self.param["hu"][hu])
        #print(my_plots_min)
        writer.add_scalars(name+"/hu_min", my_plots_min, iteration)
        writer.add_scalars(name+"/hu_max", my_plots_max, iteration)
        



class CT_relu_layer(nn.Module):
    def __init__(self,settings={}):
        super().__init__()
        defaults = {
            "scfac":0.4,
            "dim":3,
            "scale":1,
            "init":"ct",
        }
        for key, arg in settings.items():
            defaults[key] = arg
        self.param = defaults    
        
        self.init = self.param["init"]
        self.scfac =  self.param["scfac"]
        self.dim = self.param["dim"]
        
        if isinstance(self.init,str):
            if self.init == 'ct':        
                self.centers = tls.tt(np.array([-1000,-500,-100,-50,-25,0,25,50,100,500,1000]),dtype=torch.float32)
            else:
                assert False, "init not defined for histolayer"
        else:
            self.centers = tls.tt(np.array(self.init),dtype=torch.float32)

        width = np.convolve(self.centers,[-1,0,1],mode='valid')
        self.width = tls.tt(np.abs(np.append(np.append([width[0]],width),[width[-1]]))*self.scfac,dtype=torch.float32)
        
        
        #t2 = rel(((t-centers)*width)+3)
        self.width_t = torch.nn.parameter.Parameter(3.0/self.width.reshape([1,len(self.width),1,1,1]), requires_grad=True)# torch.autograd.Variable(mins, requires_grad=True)
        self.centers_t = torch.nn.parameter.Parameter(self.centers.reshape([1,len(self.centers),1,1,1]), requires_grad=True)# torch.autograd.Variable(maxs, requires_grad=True)

        self.relu = torch.nn.ReLU6()
    def forward(self,inputs):
       # print("histlayer in {} {}".format(inputs.min(),inputs.max()))
        return self.relu((inputs - self.centers_t)*self.width_t+3)
        
    def tensorboard(self,writer,name,iteration):
        my_plots_w = {}
        my_plots_b = {}

        count = 0
        #for hu in self.param["hu"]:
        for hu in range(self.centers_t.view(-1).shape[0]):
            my_plots_w[str(hu)] = self.width_t.view(-1)[count].detach().cpu().numpy()
            my_plots_b[str(hu)] = self.centers_t.view(-1)[count].detach().cpu().numpy()
            count += 1

        writer.add_scalars(name+"/hu_bias", my_plots_b, iteration)
        writer.add_scalars(name+"/hu_weights", my_plots_w, iteration)




class CT_clamp_layer(nn.Module):
    def __init__(self,settings={}):
        super().__init__()
        defaults = {
            "scfac":2.0,
            "dim":3,
            "scale":1,
            "init":"ct",
            "epsilon":0.001#.0.00001,
        }
        for key, arg in settings.items():
            defaults[key] = arg
        self.param = defaults    
        self.epsilon = self.param["epsilon"]
        self.init = self.param["init"]
        self.scfac =  self.param["scfac"]
        self.dim = self.param["dim"]
        
        if isinstance(self.init,str):
            if self.init == 'ct':        
                self.centers = tls.tt(np.array([-1000,-500,-100,-50,-25,0,25,50,100,500,1000]),dtype=torch.float32)
                #self.centers = tls.tt(np.array([-1000,-500,-250,-100,-50,-25,0,25,50,100,250,500,1000]),dtype=torch.float32)
                
            else:
                assert False, "init not defined for histolayer"
        else:
            self.centers = tls.tt(np.array(self.init),dtype=torch.float32)

        width = np.convolve(self.centers,[-1,0,1],mode='valid')
        self.width = self.scfac/tls.tt(np.abs(np.append(np.append([width[0]],width),[width[-1]])),dtype=torch.float32)
        centers = np.convolve(self.centers,[0.5,0,0.5],mode='valid')
        print(self.width)
        self.centers = tls.tt(np.append(np.append([self.centers[0]],centers),[self.centers[-1]]))
        print(self.centers)
        
        #t2 = rel(((t-centers)*width)+3)
        #self.width_t = 1.0/self.width.reshape([len(self.width),1,1,1,1])# torch.autograd.Variable(mins, requires_grad=True)
        #self.centers_t = self.centers# torch.autograd.Variable(maxs, requires_grad=True)

        self.conv = layersND.Conv(self.dim,in_channels=1,out_channels=self.width.shape[0],kernel_size=1,bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(self.width.reshape(self.conv.weight.shape))
            self.conv.bias.copy_(-self.width*self.centers)
        
        #self.relu = torch.nn.ReLU6()
    def forward(self,inputs):
       # print("histlayer in {} {}".format(inputs.min(),inputs.max()))
        #return torch.clamp((inputs - self.centers_t)*self.width_t,min=-1,max=1)
        #print(inputs.shape)
        
        y = self.conv(inputs)
        
        #if False:
        #    mask = torch.logical_or((y<-1),(y>1))
        #    y[mask] *= self.epsilon
        #mask = torch.logical_or((y<-1),(y>1))        
        #mask_ = torch.logical_or((y<-1),(y>1))      
        #else:
        mask = y<-1
        y[mask] =  (1+y[mask])* self.epsilon - 1
        mask = y>1
        y[mask] = (y[mask]-1)* self.epsilon + 1
        
      
        return y
        #print("yea")
        #return y
        #return torch.clamp(y,min=-1,max=1) #+ torch.logical_or((y<-1),(y>1))*y*self.epsilon
        
    
    def tensorboard(self,writer,name,iteration):
        my_plots_w = {}
        my_plots_b = {}

        count = 0
        #for hu in self.param["hu"]:
        for hu in range(self.centers.view(-1).shape[0]):
            my_plots_w[str(hu)] = self.conv.weight.view(-1)[count].detach().cpu().numpy()
            my_plots_b[str(hu)] = self.conv.bias.view(-1)[count].detach().cpu().numpy()
            count += 1

        writer.add_scalars(name+"/hu_bias", my_plots_b, iteration)
        writer.add_scalars(name+"/hu_weights", my_plots_w, iteration)
        
class CT_net(nn.Module):
    def __init__(self,settings={}):
        super().__init__()
        defaults = {
            "scale":1,
            "dim":3,
            "hu_":{
                "bone":[700,3000],
                "soft_tissue":[100,300],
                "liver":[40,60],
                "wm":[20,30],
                "gm":[37,45],
                "muscle":[10,40],
                "blood":[30,45],
                "kidney":[30,30],
                "csf":[15,15],
                "water":[0,0],
                "fat":[-100,-50],
                "lung":[-500,500],
                "air":[-1000,-1000],            
            },
            "hu__":{
                "bone":[250,3000],
                "soft_tissue":[90,300],
                "organs":[10,100],
                "water":[-50,15],
                "fat":[-200,-50],
                "lung":[-700,-200],
                "air":[-3000,-700],            
            },
            "hu":{
                "4":[250,3000],
                "3":[90,300],
                "2":[10,100],
                "1":[0,50],
                "0":[-25,25],
                "-1":[-50,0],
                "-2":[-100,-10],
                "-3":[-300,-90],
                "-4":[-3000,-250],            
            }

            #https://radiopaedia.org/articles/windowing-ct
        }
        for key, arg in settings.items():
            defaults[key] = arg
        self.param = defaults    
        self.dim = self.param["dim"]
        self.fout = len(self.param["hu"])
        #self.norm_layer = clamp_layer()
        mins = []
        maxs = []
        for hu in self.param["hu"]:
            hu_range = torch.tensor(self.param["hu"][hu])
            # = hu_range,abs().max() * self.param["hu"]["margin"]            
            c = (hu_range[1] - hu_range[0])/2.0
            hu_range = self.param["scale"] * (hu_range - c) + c
            mins += [hu_range[0]]
            maxs += [hu_range[1]]
            #print(hu)
       # print(mins)
       # print(maxs)
        self.main = clamp_layer(self.dim,torch.tensor(mins),torch.tensor(maxs))
    def forward(self,inputs):
        return self.main(inputs)
