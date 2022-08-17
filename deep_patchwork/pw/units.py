import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict

import matplotlib.pyplot as plt
#import torch.optim as optim
import math
import random
import torch.onnx
import pickle
#import onnx
import imageio
import nibabel as nib
import numpy as np
import copy
from . import tools as tls
#from . import data as pw_data


def rebrand(net_,brand):
    net = copy.deepcopy(net_)
    net["out"] = net.pop("OUT")
    keys = list(net.keys())
    for layer in keys:
        net[layer+brand] = net.pop(layer)
        net[layer+brand]["in"] =  [i+brand for i in net[layer+brand]["in"]]
    return net

    
        


class layersND:
    MaxPool = lambda dim_, *args, **kwargs: nn.MaxPool2d(*args, **kwargs) if dim_ == 2 else nn.MaxPool3d(*args, **kwargs)
    AvgPool = lambda dim_, *args, **kwargs: nn.AvgPool2d(*args, **kwargs) if dim_ == 2 else nn.AvgPool3d(*args, **kwargs)
    Conv = lambda dim_, *args, **kwargs: nn.Conv2d(*args, **kwargs) if dim_ == 2 else nn.Conv3d(*args, **kwargs)
    ConvT = lambda dim_, *args, **kwargs: nn.ConvTranspose2d(*args, **kwargs) if dim_ == 2 else nn.ConvTranspose3d(*args, **kwargs)
    InstanceNorm = lambda dim_, *args, **kwargs: nn.InstanceNorm2d(*args, **kwargs) if dim_ == 2 else nn.InstanceNorm3d(*args, **kwargs)
    BatchNorm = lambda dim_, *args, **kwargs: nn.BatchNorm2d(*args, **kwargs) if dim_ == 2 else nn.BatchNorm3d(*args, **kwargs)
    Dropout = lambda dim_, *args, **kwargs: nn.Dropout2d(*args, **kwargs) if dim_ == 2 else nn.Dropout3d(*args, **kwargs)
    InstanceNorm_ = lambda dim_, *args, **kwargs: nn.InstanceNorm2d_(*args, **kwargs) if dim_ == 2 else nn.InstanceNorm3d_(*args, **kwargs)
    BatchNorm_ = lambda dim_, *args, **kwargs: nn.BatchNorm2d_(*args, **kwargs) if dim_ == 2 else nn.BatchNorm3d_(*args, **kwargs)
    Spectral_norm = lambda dim_, *args, **kwargs: torch.nn.utils.spectral_norm(*args, **kwargs) 
    Dropout_ = lambda dim_, *args, **kwargs: nn.Dropout2d_(*args, **kwargs) if dim_ == 2 else nn.Dropout3d_(*args, **kwargs)
    Sigmoid =  lambda dim_, *args, **kwargs: nn.Sigmoid()
    Relu =  lambda dim_, *args, **kwargs: nn.ReLU()
    LeakyReLU =  lambda dim_, *args, **kwargs: nn.LeakyReLU(*args, **kwargs)
    Tanh =  lambda dim_, *args, **kwargs: nn.Tanh()
    Softmax =  lambda dim_, *args, **kwargs: nn.Softmax(*args, **kwargs)
    Identity = lambda dim_, *args, **kwargs: torch.nn.Identity(*args, **kwargs)
    FConv = lambda dim_, *args, **kwargs: torch.nn.functional.conv2d(*args, **kwargs) if dim_ == 2 else torch.nn.functional.conv3d(*args, **kwargs)

    @staticmethod
    def hasop(op):
        return hasattr(layersND, op)
    
    @staticmethod
    def op(op):
        return getattr(layersND, op)
    
    

class nn_op(nn.Module):
    def __init__(self,
                 dim,
                 op,
                 *args, **kwargs):
        super().__init__()
        self.dim = dim
        #print("nn_ops ",kwargs)
        #print("nn_ops ",kwargs)
        self.op = layersND.op(op)(self.dim,*args, **kwargs)

    def forward(self, input):
        #return self.op(input)
        #print("yea")
        return self.op(torch.cat(input,dim=1))
        
        
    
class grid_sample_layer(nn.Module):
        def __init__(self,dim,CT="FC",debug=False,offsets=[]):
            super().__init__()
            self.ref_img = None
            self.dim = dim
            self.CT=CT
            self.scat = None
            self.debug = debug
            self.offsets = offsets
            if len(offsets)>0:
                print("grid sampler with offsets")
                print(offsets)
            
        def scatter(self,inputs,in_im,scale=None):
            if self.scat is None:
                self.scat = tls.scatterer()
            invers_coords_ = self.ctrafo(inputs)
            
            if self.dim == 2:
                invers_coords_ = invers_coords_.permute(dims=[0,2,3,1])[...,[1,0]]
            else:
                invers_coords_ = invers_coords_.permute(dims=[0,2,3,4,1])[...,[2,1,0]]
            
            if self.debug:
                self.invers_coords_ = invers_coords_
            
            #if (len(self.offsets)==0):
            out_shape =  self.ref_img.shape
            if scale is not None:
                out_shape = tls.tt(out_shape,dtype=torch.float)
                out_shape[2:]*=tls.tt(scale,dtype=torch.float)
                out_shape = out_shape.round()
                #print(out_shape)
                #print(out_shape.dtype)
                out_shape = list(out_shape.to(dtype=torch.long).numpy())

                #patch = torch.zeros(self.ref_img.shape)
            patch = torch.zeros(out_shape)
                 
                #invers_coords__ = torch.nn.functional.interpolate(
                #    invers_coords_, size=shape_out_vox[2:].tolist(), 
                #    mode='bilinear' if self.dim == 2 else "trilinear")
            #print("wtf")
            #print(invers_coords__.shape)
            #print(in_im.shape)
            #print(patch.shape)
            
            self.scat.scatter_2_img(
                                invers_coords_,
                                in_im,
                                patch,
                                result_counts=None,
                                votes=None)
            
            return patch
            
        def ctrafo(self,inputs):
            if self.CT == "FC":
                if self.dim == 2:
                    ff = -1
                    invers_coords = torch.cat(
                        (torch.atan2(ff*(inputs[:,1,...]),ff*(inputs[:,0,...]))[:,None,...],
                         torch.atan2(ff*(inputs[:,3,...]),ff*(inputs[:,2,...]))[:,None,...],
                         ),dim=1)
                if self.dim == 3:
                    ff = -1
                    invers_coords = torch.cat(
                        (torch.atan2(ff*(inputs[:,1,...]),ff*(inputs[:,0,...]))[:,None,...],
                         torch.atan2(ff*(inputs[:,3,...]),ff*(inputs[:,2,...]))[:,None,...],
                         torch.atan2(ff*(inputs[:,5,...]),ff*(inputs[:,4,...]))[:,None,...],
                         ),dim=1)
                    
                    
                invers_coords = (invers_coords+math.pi)/(2*math.pi)
                invers_coords_ = (-1.0+2.0*invers_coords)
                    
                return invers_coords_
            if self.CT == "CAR":
                return (-1.0+2.0*inputs)
                        
        def forward(self,inputs):
              invers_coords_ = self.ctrafo(inputs)
             # print(invers_coords_.shape)
              if self.dim == 2:
                  invers_coords_ = invers_coords_.permute(dims=[0,2,3,1])[...,[1,0]]
              else:
                  invers_coords_ = invers_coords_.permute(dims=[0,2,3,4,1])[...,[2,1,0]]
              
           #   print(invers_coords_.shape)
              batch_size = invers_coords_.shape[0]
              shape = inputs.shape[2:]
              modalities = self.ref_img.shape[1] 
              
              num_offsets = max(len(self.offsets),1)
              patch = torch.zeros((batch_size,)+(modalities*num_offsets,)+tuple(shape),device=invers_coords_.device)
              
              
              #print("################################")
              #print(patch.shape)
              
              in_shape = (1,)+(batch_size*invers_coords_.shape[1],) + tuple(invers_coords_.shape[2:-1])+(self.dim,)
              out_shape = (batch_size,)+ tuple(invers_coords_.shape[1:-1])
              
              if self.ref_img.device != invers_coords_.device:
                  self.ref_img = self.ref_img.to(invers_coords_.device)
              
              if len(self.offsets) > 0:
                #print("###########")
                count = 0
                for o in range(num_offsets):
                    ref_shape = self.ref_img.shape[2:] 
                    #print(torch.tensor(ref_shape ))
                  #  print(self.offsets[o])
                    oset = self.offsets[o] / torch.tensor(ref_shape )[None,None,None,None,:]
                   # print(oset)
                    coords = invers_coords_.reshape(in_shape)+oset.to(device=invers_coords_.device)
                    for l in range(modalities):
                        patch[:,count,...] = torch.nn.functional.grid_sample( 
                            self.ref_img[:,l,None,...],
                            coords,
                            align_corners=True).reshape(out_shape)    
                        count += 1

              else:
                  for l in range(modalities):
                #   print(self.ref_img[:,l,None,...].shape)
                    #  print(invers_coords_[...,:self.dim].reshape(in_shape).shape)
                    if True:
                        patch[:,l,...] = torch.nn.functional.grid_sample( 
                            self.ref_img[:,l,None,...],
                            invers_coords_.reshape(in_shape),
                            align_corners=True).reshape(out_shape)    
                    else:
                        tmp = torch.nn.functional.grid_sample( 
                            self.ref_img[:,l,None,...],
                            invers_coords_.reshape(in_shape),
                            align_corners=True)#.reshape(out_shape)    
                    #   print(tmp.shape)
                        #print(out_shape)
                        # print(patch[:,l,...].shape)
                    
              
            #  print(patch.shape)
              return patch
          
        def set_ref_img(self,img):
            self.ref_img = img
            
class ifft_layer(nn.Module):
    def __init__(self,
                 dim,
                 K,
                 pfield,
                 #in_coeffs,
                 ):
        super().__init__()
        self.K = K
        self.pfield = pfield if len(pfield)> 1 else pfield*dim
        self.dim = dim
        
        indx = []
        self.pf = 2
        #self.pf = 1
        #coeff = []
        
        #pfield_ = list(np.array(self.pfield)*2)
        pfield_ = list(np.array(self.pfield)*self.pf)
        
        #pfield_ = list(np.array(self.pfield))
        pfield_[-1] = pfield_[-1] // 2 + 1
        self.pfield_ = pfield_
        n_coeffs = 0
        # print(pfield_)
        #print(K)
        for k1 in range(min(pfield_[0],K+1)):
            for k2 in range(min(pfield_[1],K+1)):
                if dim == 2:
                    if k1+k2 <= K:
                        #print(k1," ",k2)
                        indx += [k1*pfield_[1]+k2]
                        n_coeffs += 1
                else:
                    for k3 in range(min(pfield_[2],K+1)):
                        if k1+k2+k3 <= K:
                            #print(k1," ",k2," ",k3)
                            indx += [(k1*pfield_[1]+k2)*pfield_[2]+k3]
                            n_coeffs += 1
            
      
        self.indx = torch.tensor(indx)
        self.indx_ = self.indx.clone()
        
        n_coeff_t = ifft_layer.get_input_length(K,dim,pfield,self.pf)
        assert(n_coeff_t == n_coeffs)
        self.n_coeffs = n_coeffs
        
        #self.conv_r = layersND.Conv(dim,in_channels=in_channels,out_channels=n_coeffs,kernel_size=1,bias=True)
        #self.conv_i = layersND.Conv(dim,in_channels=in_channels,out_channels=n_coeffs,kernel_size=1,bias=True)
        #self.conv_r = layersND.Conv(dim,in_channels=in_coeffs,out_channels=n_coeffs,kernel_size=1,bias=True)
        #self.conv_i = layersND.Conv(dim,in_channels=in_coeffs,out_channels=n_coeffs,kernel_size=1,bias=True)
        
   
    
    @staticmethod
    def get_input_length(K,dim,pfield,pf=2):
        #return (((K)+1)*((K)+2))//2
        dim = len(pfield) if type(pfield) in [list,tuple] else len(pfield.shape)
        n_coeffs = 0
        pfield_ = list(np.array(pfield)*pf)
        
        #pfield_ = list(np.array(self.pfield))
        pfield_[-1] = pfield_[-1] // 2 + 1
       # print(pfield_)
        print(K)
        for k1 in range(min(pfield_[0],K+1)):
            for k2 in range(min(pfield_[1],K+1)):
                if dim == 3:
                    for k3 in range(min(pfield_[2],K+1)):
                        if k1+k2+k3 <= K:
                            #print(k1," ",k2," ",k3)
                            n_coeffs += 1
                else:
                    if k1+k2 <= K:
                       # print(k1," ",k2)
                        n_coeffs += 1
        return n_coeffs
    
    def forward(self, input):
            shape = torch.tensor(input.shape)
            
            shape[1] = shape[1:].prod()//(2*self.n_coeffs)
            shape[2] = 2*self.n_coeffs
            #shape = [shape[0],shape[1],shape[2:].prod()]
            shape = [shape[0],shape[1],shape[2]]
            input_ = input.reshape(shape)
            Ti = torch.zeros([shape[0],shape[1]]+self.pfield_,dtype=torch.complex64,device=input.device)
            #coeff_r = self.conv_r(input)
            #coeff_i = self.conv_i(input)
            #n_ind = len(self.indx)
            #ind_ = self.indx.clone()
            #offset = shape[2:].prod()
            #channels = shape[1]
            
            input_ /= math.sqrt(self.n_coeffs)
            coeff_r = input_[:,:,:self.n_coeffs]
            coeff_i = input_[:,:,self.n_coeffs:]
            coeff = torch.complex(coeff_r,coeff_i) 
            
            offset_c = np.array(self.pfield_).prod()
            offset_b = offset_c*shape[1]
            
            if (len(self.indx_.shape)!=len(coeff.shape)) or \
                (torch.tensor(self.indx_.shape) != torch.tensor(coeff.shape)).any() or \
                    self.indx_.device!=input.device:
                print("updaing scatter indx")
                self.indx_ = torch.zeros(coeff.shape,dtype=torch.int64,device=input.device)#self.indx.clone()
                for b in range(shape[0]):
                    for c in range(shape[1]):
                        self.indx_[b,c,...] = self.indx+c*offset_c+b*offset_b
            
            fft_n = "forward"
            #fft_n = "backward"
            #coeff[:] = 1 + 1j
            
            Ti.view(-1).scatter_(0,self.indx_.view(-1),coeff.view(-1))
            #Ti[0,0,0,0] =  1 + 1j
            #Ti[0,0,0,1] =  1 + 1j
            #Ti[0,0,1,0] =  1 + 1j
            
            if self.dim == 2:
               # plt.imshow(Ti[0,0,...].real)
            #    plt.pause(0.1)
             #   plt.imshow(Ti[0,0,...].imag)
                #return torch.fft.irfft2(Ti,norm=fft_n)[:,:,:self.pfield[0],:self.pfield[1]]
                y = torch.fft.irfft2(Ti,norm=fft_n)[:,:,:self.pfield[0],:self.pfield[1]]
               # print("fft minmax: {} {}".format(y.amin(),y.amax()))
               # print("coeff_r minmax: {} {}".format(coeff_r.amin(),coeff_r.amax()))
               # print("coeff_i minmax: {} {}".format(coeff_i.amin(),coeff_i.amax()))
               # print("coeff shape {}".format(coeff.shape))
                return y
            return torch.fft.irfftn(Ti,norm=fft_n,dim=(2,3,4))[:,:,:self.pfield[0],:self.pfield[1],:self.pfield[2]]
            

class basic(nn.Module):
      def __init__(self,
                   dim,
                   btype="add",
                   params=None):
          super().__init__()
          self.dim = dim
          self.btype = btype
          self.params = params
         
      def forward(self, input):
          if self.btype == "add":
              #print("add:")
              for x,indx in zip(input,range(len(input))):
               #   print("add {}".format(x.shape))
                  if indx == 0:
                      y = x#.clone()
                  else:
                      y += x
          if self.btype == "mult":
              for x,indx in zip(input,range(len(input))):
                  if indx == 0:
                      y = x#.clone()
                  else:
                      y *= x
                      
          if self.btype == "expand":
              y = input.expand(**self.params)
              
          #if self.btype == "eval":
          #    y = eval(self.params)
          #if self.btype == "fun":
          #    y = eval(self.params)
              
          return y
                  


#test_im_hist.py for ISH init
class Hist_layer(nn.Module):
    def __init__(self,settings={}):
        super().__init__()
        defaults = {
            "fout":4,
            "initfile":None,
        }
        for key, arg in settings.items():
            defaults[key] = arg
        self.param = defaults    
        
        self.fout = self.param["fout"]
        
        self.pre_hist = False
        #if type(self.fout)==str:
        if True:
            if self.param["initfile"] is not None:
                print("init histlayer with {}".format(self.param["initfile"]))
                dbfile = open(self.param["initfile"], 'rb')     
                hmap = torch.tensor(pickle.load(dbfile)).transpose(0,1)
                dbfile.close()
                #self.fout = hmap.shape[0]
                self.pre_hist = True
        
        #self.init = True
        self.norm = layersND.BatchNorm(2,self.fout)
        
        
        self.conv = layersND.Conv(2,in_channels=256,out_channels=self.fout,kernel_size=1,bias=True)
        
        with torch.no_grad():
            #print(self.conv.weight.shape)
            #print(self.conv.bias.shape)
            #print(torch.arange(256).shape)
            if self.pre_hist:
                self.conv.weight.copy_(hmap[:,:,None,None])
            else:
                self.conv.weight.copy_(torch.arange(256)[None,:,None,None])
            self.conv.bias.copy_(torch.zeros(self.fout))
        
    def forward(self,inputs):
        
        if True:
                bins = 256
                indeces = inputs.ceil().flatten(start_dim=1).to(torch.int64)
                
                indeces += bins*torch.arange(indeces.shape[1],device=inputs.device)[None,...]
                indeces += bins*inputs.shape[2]*inputs.shape[3]*torch.arange(inputs.shape[0],device=inputs.device)[...,None]
                
                t = torch.zeros([inputs.shape[0]*bins*inputs.shape[2]*inputs.shape[3]],requires_grad=False,device=inputs.device)
                t[indeces] = 1
                t = t.reshape([inputs.shape[0],inputs.shape[2],inputs.shape[3],bins])
                t = t.permute(dims=[0,3,1,2])
                t = t.clone()
            
        y = self.norm(self.conv(t))
        
        return y
    
class CT_clamp_layer(nn.Module):
    def __init__(self,settings={}):
        super().__init__()
        defaults = {
            "scfac":2.0,
            "dim":3,
            "scale":1,
            "init":"ct",
            "epsilon":0.00001#.0.00001,
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
        #print(self.width)
        self.centers = tls.tt(np.append(np.append([self.centers[0]],centers),[self.centers[-1]]))
        #print(self.centers)
        
        self.conv = layersND.Conv(self.dim,in_channels=1,out_channels=self.width.shape[0],kernel_size=1,bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(self.width.reshape(self.conv.weight.shape))
            self.conv.bias.copy_(-self.width*self.centers)
        
    def forward(self,inputs):
        y = self.conv(inputs)
        #mask = y<-1
        #y[mask] =  (1+y[mask])* self.epsilon - 1
        #mask = y>1
        #y[mask] = (y[mask]-1)* self.epsilon + 1
      
        return torch.clamp(y,min=-1,max=1) + self.epsilon*y
       
        
    
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
        


class CC_layer(nn.Module):
        def __init__(self,dim,d=[5,5,5],pfield=None,normalized=False,eps=0.0001,eps_p="futsu"):
            super().__init__()
            
            self.eps=eps
            self.eps_p=eps_p
            self.normalized = normalized
            self.pfield = pfield
            self.dim = dim#len(d)
            #self.d = np.minimum(np.array(d),np.array(pfield))
            self.d = np.minimum(np.array(d),pfield)
            self.k = None

        def build_kernel(self):
            t = 1.25
            w_size = self.d*2+1
            if self.dim==1:
                dx = torch.arange(0,w_size[0])
                XX = torch.meshgrid(dx, indexing="ij")
                XX = (XX[0] - (w_size[0]//2)).float()
                k = (((XX/self.d[0])**2)<((t))).float()
            if self.dim==2:
                dx = torch.arange(0,w_size[0])
                dy = torch.arange(0,w_size[1])
                YY, XX = torch.meshgrid(dx, dy, indexing="ij")
                XX = (XX - (w_size[1]//2)).float()
                YY = (YY - (w_size[0]//2)).float()
                #k = ((XX**2+YY**2)<((w_size+0.5)**2)).float()
                k = (((XX/self.d[1])**2+(YY/self.d[0])**2)<(t)).float()
            if self.dim==3:
                dx = torch.arange(0,w_size[0])
                dy = torch.arange(0,w_size[1])
                dz = torch.arange(0,w_size[2])
                ZZ, YY, XX = torch.meshgrid(dx, dy, dz, indexing="ij")
                XX = (XX - (w_size[2]//2)).float()
                YY = (YY - (w_size[1]//2)).float()
                ZZ = (ZZ - (w_size[0]//2)).float()
                
                k = (((XX/self.d[2])**2+(YY/self.d[1])**2+(ZZ/self.d[0])**2)<((t))).float()
                
            
            return k
        
        def forward(self,inputs):
            #print(inputs[0].shape)
           # print(inputs[1].shape)
            #print(aXb.shape)
            
            aXb = inputs[0]*inputs[1]
            if self.normalized:
                aXa = inputs[0]**2
                bXb = inputs[1]**2
            
            if self.k is None or self.k.device != aXb.device or self.k.shape[1] != aXb.shape[1]:
                self.k = self.build_kernel().to(aXb.device)
                if self.dim == 2:
                    self.k = self.k[None,None,...].expand([-1,aXb.shape[1],-1,-1])
                else:
                    self.k = self.k[None,None,...].expand([-1,aXb.shape[1],-1,-1,-1])
                #self.k /= self.k.sum()
            
            
            y = layersND.FConv(self.dim,
                               aXb,
                               self.k,
                               padding='same')
            
            if self.normalized:
                na = layersND.FConv(self.dim,
                                   aXa,
                                   self.k,
                                   padding='same')
                nb = layersND.FConv(self.dim,
                                   bXb,
                                   self.k,
                                   padding='same')
                
                naXnb = torch.sqrt(torch.clamp(na*nb,min=0.0))
                if self.eps_p == "both":
                    
                    
                    
                                            
                    if False:
                        print("cc inputs[0]: ",inputs[0].amin()," ",inputs[0].amax()," ",inputs[0].sum())
                        print("cc inputs[1]: ",inputs[1].amin()," ",inputs[1].amax()," ",inputs[1].sum())
                        print("cc aXb: ",aXb.amin()," ",aXb.amax()," ",aXb.sum())
                        print("cc wtf 1: ",y.amin()," ",y.amax()," ",y.sum())
                        print("cc wtf 2: ",na.amin()," ",na.amax()," ",na.sum())
                        print("cc wtf 3: ",nb.amin()," ",nb.amax()," ",nb.sum())
                        print("cc wtf 4: ",naXnb.amin()," ",naXnb.amax()," ",naXnb.sum())
                    
                    y = (y+self.eps)/(naXnb+self.eps)
                else:
                    y /= naXnb+self.eps
            else:
                y /= aXb.shape[1]
                
            return y
        
        
        
        
class patch_block(nn.Module):
    def __init__(self,
                 dim,
                 pfield=32,
                 pfield_out=32,
                 input_channels=3,
                 output_channels=3,
                 upsample_features = -1,
                 out_channels_v = 6,
                 out_channels_k = 5,
                 mt = 2,
                 size = 4,
                 act = nn.LeakyReLU,
                 upscale_out=True,
                 res="cat",
                 mlp=False):
        super().__init__()
        self.dim = dim
        self.res = res
       
        self.pfield = pfield
        
        #ps = pfield
        #self.ts = ps//size
        #self.otiles = self.ts**self.dim #s.shape[-1]
        out_channels_q = out_channels_k
        self.mt = mt
        
        assert(pfield%size == 0)
        self.ts = size
        size = pfield//size
        #out_channels_v = self.ts**self.dim
        
        #print("value dim: {}".format(out_channels_v))
        
        self.conv_k = layersND.Conv(self.dim,in_channels=input_channels,out_channels=out_channels_k*mt,kernel_size=size,stride=size)
        self.conv_v = layersND.Conv(self.dim,in_channels=input_channels,out_channels=out_channels_v*mt,kernel_size=size,stride=size)
        self.conv_q = layersND.Conv(self.dim,in_channels=input_channels,out_channels=out_channels_q*mt,kernel_size=size,stride=size)
        
        #self.conv_t = layersND.ConvT(self.dim,in_channels = mt*out_channels_v,out_channels = output_channels,kernel_size = size,stride = size)
        #self.conv_t = layersND.ConvT(self.dim,in_channels = (mt+1)*out_channels_v,out_channels = output_channels,kernel_size = size,stride = size)
        #self.conv_t = layersND.ConvT(self.dim,in_channels = (mt+1)*out_channels_v,out_channels = out_channels_v,kernel_size = size,stride = size)
        
        #print("conv_s {} {}".format(self.otiles*self.mt*self.otiles,self.otiles))
        #self.conv_s = torch.nn.Conv1d(in_channels=1,out_channels=self.otiles*self.mt*self.otiles,kernel_size=self.otiles,stride=1)
        #self.conv_s = torch.nn.Conv1d(in_channels=1,out_channels=self.otiles*self.mt*self.otiles,kernel_size=mt,stride=1)
        
        #self.inorm_k = layersND.InstanceNorm(self.dim,out_channels_k)
        #self.inorm_v = layersND.InstanceNorm(self.dim,out_channels_v)
        #self.inorm_q = layersND.InstanceNorm(self.dim,out_channels_q)
        
        self.inorm_k = layersND.InstanceNorm(self.dim,self.mt)
        self.inorm_v = layersND.InstanceNorm(self.dim,self.mt)
        self.inorm_q = layersND.InstanceNorm(self.dim,self.mt)
        
        
        
        
        
        norm = "InstanceNorm"
        #norm = None
        if self.res == "cat":
            #f_channels_in = 2*self.mt * self.ts**self.dim
            #f_channels_in = self.mt * (self.ts**self.dim + 1)
            f_channels_in = self.mt * (out_channels_v + 1)
            #f_channels = (mt+1)*out_channels_v
        elif self.res == "+":
            #f_channels_in = self.mt * self.ts**self.dim
            f_channels_in = self.mt * out_channels_v
        else:
            f_channels_in = self.mt 
            
        self.inorm_f = layersND.InstanceNorm(self.dim,f_channels_in)

        self.mlp = mlp
        if mlp:
            mlp_c = self.ts**self.dim
            #self.mlp = torch.nn.Linear(in_features=f_channels_in,out_features=f_channels_in)
            self.mlp = torch.nn.Conv1d(in_channels=mlp_c,out_channels=mlp_c,kernel_size=1,stride=1)
            self.mlp2 = torch.nn.Conv1d(in_channels=f_channels_in,out_channels=f_channels_in,kernel_size=1,stride=1)
            self.mlp_act = nn.ReLU()
            self.mlp_norm = nn.InstanceNorm1d(f_channels_in)
        

        f_channels_out = f_channels_in if upsample_features<1 else upsample_features
            
        self.ups = nn.ModuleDict()
        if upscale_out:
            nups = int(math.log2(pfield_out//self.ts))
            #nups = int(math.log2(ps//self.ts))
            #print(" self.ts {}".format(self.ts))
            #print(" size {}".format(size))
            #print(" nups {}".format(nups))
            for n in range(nups):
                self.ups[str(n)] = conv_block(dim=self.dim,
                                      input_channels=f_channels_in,
                                      output_channels=f_channels_out if n<nups-1 else output_channels,
                                      btype="up",
                                      norm=norm,
                                      kernel_size=2)
                f_channels_in = f_channels_out
                
        else:
            self.ups["0"] = conv_block(dim=self.dim,
                                  input_channels=f_channels_in,
                                  output_channels=output_channels,
                                  btype="same",
                                  norm=norm,
                                  kernel_size=1,
                                  padding=0)

    def forward(self, input):
        
        assert(self.pfield == input.shape[-1])
        t_k = self.conv_k(input)
        t_v = self.conv_v(input)
        t_q = self.conv_q(input)
        
        
        shape_v = t_v.shape
        t_k = t_k.reshape([t_k.shape[0],self.mt,t_k.shape[1]//self.mt,np.prod(t_k.shape[2:])])
        t_v = t_v.reshape([t_v.shape[0],self.mt,t_v.shape[1]//self.mt,np.prod(t_v.shape[2:])])
        t_q = t_q.reshape([t_q.shape[0],self.mt,t_q.shape[1]//self.mt,np.prod(t_q.shape[2:])])
        

        t_k =  self.inorm_k(t_k)
        t_v =  self.inorm_v(t_v)
        t_q =  self.inorm_q(t_q)


        #print(input.shape)
        #print(t_k.shape)
      #  print(t_v.shape)
        #print(t_q.shape)
        #%%
        cmat = torch.matmul(t_k.permute([0,1,3,2]),t_q)
        sm = torch.nn.Softmax(dim=-2)
        cmat = sm(cmat)
        if False:
            #%%
            cmat_I = torch.zeros(cmat.shape)
            #for b in range(cmat_I.shape[-1]):
                #cmat_I[:,:,b,b] = 1
            #cmat_I[:,:,0,1] = 1
            #cmat_I[:,:,0,0] = 1
            #cmat_I[0,0,1,0] = 1
            #cmat_I[0,0,0,0] = 1
            
            cmat_I[:,:,1,0] = 1
            cmat_I[:,:,0,0] = 1
                #cmat_I[b,b,:,:] = 1
            
            t_v_t = torch.matmul(t_v[:,:,:,None,:],cmat_I[:,:,None,:,:])
            #(t_v_t[:,:,:,0,:]-t_v).abs().sum()
            (t_v_t[0,4,:,0,0] - t_v[0,4,:,:2].sum(-1)).abs().sum()
            #%%
        out = torch.matmul(t_v[:,:,:,None,:],cmat[:,:,None,:,:])[:,:,:,0,:]
        #%%
        if False:
            #s = (t_q*t_k).sum(axis=1,keepdim=True)
            s = (t_q*t_k).sum(axis=2,keepdim=True)
            sm = torch.nn.Softmax(dim=-1)
            s = sm(s).permute([0,1,3,2])
            
            out = (s*t_v).sum(dim=-1)
            out = out[...,None,:]
        #print(s.shape)
        #print(self.conv_s.weight.shape)
        #a = self.conv_s(s)
        #a = a.reshape([s.shape[0],self.mt*self.otiles,1,s.shape[2]])
        #sm = torch.nn.Softmax(dim=-1)
        #a = sm(a)
        #t_v = t_v.reshape([t_v.shape[0],1,t_v.shape[1],t_v.shape[2]])
        
        
        #out = (a*t_v).sum(dim=-1).permute([0,2,1])
        #out = out.reshape((out.shape[0],out.shape[1]*self.mt,)+(self.ts,)*self.dim)
        
        #if not hasattr(self,"res") or self.res == "cat":
        if self.res == "cat":
            #y = torch.cat([out,t_v.reshape(shape_v)],dim=1)
            #y = torch.cat([out.repeat(1,1,1,t_v.shape[-1]),t_v],dim=1)
            y = torch.cat([out,t_v],dim=2)
        elif self.res == "+":
            y = (out+t_v)
        else:
            y = out
            #if self.dim == 3:
            #    y = out+t_v.reshape(shape_v).repeat([1,self.mt,1,1,1])
            #else:
            #    y = out+t_v.reshape(shape_v).repeat([1,self.mt,1,1])
        #y = y.reshape((y.shape[0],y.shape[1]*y.shape[2])+(self.gen_tile_size,)*self.dim)
       # print(y.shape)
     #   print((y.shape[0],y.shape[1]*y.shape[2])+(self.ts,)*self.dim)
        y = y.reshape((y.shape[0],y.shape[1]*y.shape[2])+(self.ts,)*self.dim)
        
        y = self.inorm_f(y)
        
        
        if self.mlp:
            y_ = self.mlp(y.reshape([y.shape[0],y.shape[1],y.shape[2]*y.shape[3]]).permute([0,2,1])).permute([0,2,1])
            y_ = self.mlp2(y_)
            y_ = self.mlp_norm(self.mlp_act(y_))
            #y_ = self.mlp_norm(self.mlp_act(self.mlp(y.reshape([y.shape[0],y.shape[1]*y.shape[2]*y.shape[3],1]))))
           
            y += y_.reshape(y.shape)
        
        
        #print("wtf: {}".format(y.shape))
        for n in range(len(self.ups)):
            y = self.ups[str(n)](y)
            #print("wtf: {}".format(y.shape))
        return y
        #return self.conv_t() 
        #return self.conv_t(out) + self.conv_skip(t_v.reshape(shape_v))
        
class conv_block(nn.Module):
    def __init__(self,
                 dim,
                 input_channels=3,
                 output_channels=3,
                 btype='same',
                 norm=None,
                 act = nn.LeakyReLU,
                 kernel_size = None,
                 stride = None,
                 padding = None,
                 dropouts=0,
                 probabilistic=False,
                 groups = 1):
        super().__init__()
        self.dim = dim
        self.probabilistic = probabilistic
        pdf_param_fact = 2 if self.probabilistic else 1
        
        self.norm = None if norm is None else getattr(layersND,norm)(self.dim,output_channels*pdf_param_fact)
        #self.norm = None if norm is None else getattr(layersND,norm)(self.dim,output_channels)
        self.dropout = None if dropouts==0 else layersND.Dropout(self.dim,dropouts)   
        self.out = None if act is None else act()

        #print("probabilistic {} {}".format(probabilistic,btype))
        
        
        bias = True
        self.btype = btype 
        
        #padding = tls.to(padding)
     #   kernel_size = tls.to(kernel_size)
       # stride = tls.to(stride)
        
            
     #   print("padding type ",type(padding))
       # print("padding type ",type(kernel_size))

        if btype == 'same':
            kernel_size = 3 if kernel_size is None else kernel_size
            stride = 1 if stride is None else stride
            padding = 1 if padding is None else padding
           # padding = 0
            #print(padding)
            self.main = layersND.Conv(self.dim,input_channels,output_channels*pdf_param_fact , kernel_size, stride, padding, bias=bias,padding_mode='replicate',groups=groups)
            #self.main = layersND.Conv(self.dim,input_channels,output_channels , kernel_size, stride, padding, bias=bias,padding_mode='zeros')

        if btype == 'down':
            kernel_size = 2 if kernel_size is None else kernel_size
            stride = 2 if stride is None else stride
            padding = 0 if padding is None else padding
            #self.main = layersND.Conv(self.dim,input_channels,output_channels , kernel_size, stride, padding, bias=bias,padding_mode='replicate')
            self.main = layersND.Conv(self.dim,input_channels,output_channels*pdf_param_fact , kernel_size, stride, padding, bias=bias,padding_mode='zeros',groups=groups)

        if btype == 'up':
            kernel_size = 2 if kernel_size is None else kernel_size
            stride = 2 if stride is None else stride
            padding = 0 if padding is None else padding
            self.main = layersND.ConvT(self.dim,input_channels,output_channels*pdf_param_fact , kernel_size, stride, padding, bias=bias,groups=groups)
            
        if btype == 'upsample':
            kernel_size = 2 if kernel_size is None else kernel_size
            self.main = torch.nn.Upsample(scale_factor=kernel_size,align_corners=True)
        
        
            
        

    def forward(self, input):
        y =  self.main(input)
        #if self.probabilistic:
            #print("probabilistic")
        #    m = y[:,:y.shape[1]//2,...]
        #    s = y[:,y.shape[1]//2:,...]
        #    shape = tls.tt(y.shape)
            
        #    shape[1] //=2
        #    y = torch.randn(shape.tolist(),device=y.device)*s+m
            
        if self.norm is not None:
            y =  self.norm(y)
        if self.out is not None:        
            y=  self.out(y)
        if self.dropout is not None:
            y =  self.dropout(y)
        if self.probabilistic:
            #print("probabilistic")
            m = y[:,:y.shape[1]//2,...]
            s = y[:,y.shape[1]//2:,...]
            shape = tls.tt(y.shape)
            shape[1] //=2
            #torch.div(a, b, rounding_mode='floor')
            y = torch.randn(shape.tolist(),device=y.device)*s+m
            
            
        return y
    
    


class model_factory(nn.Module):
    
    def __init__(self, model,verbose=0):
        super().__init__()
       
        self.mops = nn.ModuleDict()
        self.ops = {}
        self.model = model["net"]
        self.params = model["params"]
        self.dim = self.params["dim"]
        self.order = []
        
        
        
        
        dependent = [key for key in self.model]
        
        #sort layers based in their dependencies
        in_vects = []
        for key in dependent:
           op = self.model[key]
           assert(type(op["in"])==list)
           in_vects += [k for k in op["in"] if (k not in in_vects and "IN" in k)]
        
        old_len = -1
        while len(dependent) > 0:
            if verbose>0:
                print(dependent)
           # if debug:
           #     print(dependent)
           #print(dependent)
            for key in dependent:
               op = self.model[key]           
               #s = all([k in self.order or k == "IN" for k in  op["in"]])
               s = all([k in self.order or "IN" in k for k in  op["in"]])
               
               
               if s:
                   self.order += [key]
                   dependent.remove(key)
            new_len = len(dependent)
            if new_len==old_len:
                print([k in self.order or "IN" in k for k in  op["in"]])
                print("self.order :",self.order)
                print("op[in] :",op["in"])
                assert(new_len!=old_len)
                
            old_len = new_len
        
        
        #guessing input channles when undefined 
        self.noutput = {}
        for key in self.order:
            op = self.model[key]
            if type(op["op"]) == str:
                if "in_channels" in op["params"] or "input_channels" in op["params"]:
                    in_key = "input_channels" if "input_channels" in op["params"] else "in_channels"
                    #print("in channels {}".format(op["params"][in_key]))
                    if op["params"][in_key] is None:
                        input_channels = 0
                        for key_in in  op["in"]:
                            op_in = self.model[key_in]
                            if "out_channels" in op_in["params"] or "output_channels" in op_in["params"]:
                                out_key = "output_channels" if "output_channels" in op["params"] else "out_channels"
                                input_channels += op_in["params"][out_key]
                        print("number of input channels for {} not given. Guessing: {}".format(key,input_channels))
                        op["params"][in_key] = input_channels
                   
            
        #init all objects
        for key,value in self.model.items():
            op = self.model[key]
            if type(op) == nn.Module:
                self.mops[key] = op["op"]
            elif type(op["op"]) == str:
                #if "in_channels" or "input_channels" in op["params"]:
                #   in_key = "input_channels" if "input_channels" in op["params"] else "in_channels"
               # if verbose>0:
               #     print("op :",op["op"])
                       
                #if op["op"] in layersND.ops:
                if layersND.hasop(op["op"]):
                    self.mops[key] =  layersND.op(op["op"])(self.dim,**op["params"])                
                if op["op"] == "conv_block":
                    self.mops[key] =  conv_block(self.dim,**op["params"])
                if op["op"] == "patch_block":
                    self.mops[key] =  patch_block(self.dim,**op["params"])
                if op["op"] == "basic":
                    self.mops[key] =  basic(self.dim,**op["params"])
                if op["op"] == "ifft_layer":
                    self.mops[key] =  ifft_layer(self.dim,**op["params"])   
                if op["op"] == "grid_sample_layer":
                    self.mops[key] =  grid_sample_layer(self.dim,**op["params"])   
                if op["op"] == "CC_layer":
                    self.mops[key] =  CC_layer(self.dim,**op["params"])   
                if op["op"] == "lambda":
                    #print(op["op"])
                    print(op["params"])
                    self.ops[key] =  eval(op["params"])   
                if op["op"] == "nn_op":
                    self.mops[key] =  nn_op(self.dim,**op["params"])   
                #if op["op"] == "detach":
                #    self.mops[key] =  lambda x:x.detach()
                #if op["op"] == "fun":
                    #self.mops[key] =  op["params"]
                    
            else:
                self.ops[key] = op["op"]
                
    
      
            
        if verbose >0:
            print("inputs : {}".format(in_vects))
            print("IN->"+"->".join(self.order))
        self.init = True
               
               
    def reset_parameters(self,
                         init_fun_conv=nn.init.xavier_uniform_,
                         init_batchnorm={"w":1,"b":0},
                         verbose=0):
            def reset_weights(m):
                    if  isinstance(m, nn.Conv2d) or \
                        isinstance(m, nn.Conv3d) or \
                        isinstance(m, nn.ConvTranspose2d) or \
                        isinstance(m, nn.ConvTranspose3d):
                        init_fun_conv(m.weight.data)
                        nn.init.constant_(m.bias.data,0) 
                        if verbose:
                            print("resetting conv weight")
                    elif isinstance(m, nn.BatchNorm2d) or \
                         isinstance(m, nn.BatchNorm3d):
                        nn.init.constant_(m.weight,init_batchnorm["w"])
                        nn.init.constant_(m.bias, init_batchnorm["b"])
                        if verbose>0:
                            print("resetting batchnorm weight")
            self.apply(reset_weights)   
                
                     
    def export_onnx(self,fname,
                    dummy_input):
            # Export the model
            torch.onnx.export(self,                      # model being run
                              dummy_input,               # model input (or a tuple for multiple inputs)
                              fname,                     # where to save the model (can be a file or file-like object)
                              export_params=True,        # store the trained parameter weights inside the model file
                              opset_version=11,          # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names = ['input'],   # the model's input names
                              output_names = ['output'], # the model's output names
                              dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                            'output' : {0 : 'batch_size'}})
            #onnx_model = onnx.load(fname)
            #onnx.checker.check_model(onnx_model,full_check=True)

        
    def forward(self, inputs,verbose=0):
        #print(type(inputs))
        #print(len(inputs))
        #print(type(verbose))
        #print(len(verbose))
        #verbose = 2
      #  verbose = 3
        #if dummy_tracing_input is not None:
            
        
        inputs = (inputs,) if type(inputs)!=tuple else inputs
        
        
       # print("input minmax : {}".format([(t.amin(dim=(0,2,3)),t.amax(dim=(0,2,3))) for t in inputs]))
      #  print("input shape : {}".format([t.shape for t in inputs]))
        
        
        if verbose>0:
            print("#################################################")
            print("input shape: {}".format([a.shape for a in inputs]))
        #if inputs.shape[0] == 1:
        #    print("Warning: batch contains only one sample")
        outputs = {}
        for key in self.order:
            if verbose>1:
                print("generating output {}".format(key))
                #if key == "OUT":
                #    print("")
            op = self.model[key]
            merge_op = op["args"] if "args" in op else "cat" 
            x_in = ()
            for key_in in  op["in"]:
                rng = op["range"] if "range" in op else None 
               # if rng is not None:
                #    print("")
                
                if key_in in self.model:
                    x_in += (outputs[key_in] if rng is None else outputs[key_in][:,rng[0]:rng[1],...],)
                
                elif "IN" in key_in:
                    if key_in == "IN":
                        assert(rng==None)
                        x_in += inputs 
                    else:             
                        key_in_ = int(key_in.replace("IN",""))
                        #x_in += (inputs[key_in_] if rng is None else inputs[key_in][:,rng[0]:rng[1],...] ,)
                        x_in += (inputs[key_in_] if rng is None else inputs[key_in_][:,rng[0]:rng[1],...] ,)
                    #x_in += (inputs,)
                else:
                    assert(False)
            if len(x_in) == 1:
                if merge_op == "cat":
                    x_in = x_in[0]
               
            else:
                if merge_op == "cat":
                    if verbose>1:
                        shapes_ = [str(tuple(k.shape)) for k in x_in] if  type(x_in)==tuple else tuple(x_in.shape)
                        #.join(" - ")
                        print("cat shapes: "+" ".join(shapes_))
                    x_in = torch.cat(x_in,dim=1)
            
            if verbose>0:
                shapes_ = [tuple(k.shape) for k in x_in] if  type(x_in)==tuple else tuple(x_in.shape)
                print("{} -> {} | {}".format(",".join(op["in"]),key,shapes_))
                
            if verbose > 2:
                if type(x_in) == tuple:
                    for xi in x_in:
                        assert(not xi.isnan().any())
                else:
                    assert(not x_in.isnan().any())
            outputs[key] = self.ops[key](x_in) if key in self.ops else  self.mops[key](x_in)    
            
            
                    
            if verbose>1:                      
                #if key == "OUT_":
                #    print("sdsd")
                shapes_ = [tuple(k.shape) for k in x_in] if  type(x_in)==tuple else tuple(x_in.shape)
                print("{} -> {} [{} -> {}]".format(",".join(op["in"]),
                                                            key,
                                                            shapes_,
                                                            tuple(outputs[key].shape)))
            
            if verbose > 2:
                assert(not outputs[key].isnan().any())
            #elif verbose>0:
            #    print("{} -> {}".format(",".join(op["in"]),key))
    
        if verbose>0:
            print("#################################################")
        return outputs["OUT"]
    
    
def DummyNet(dim=2,
        pfield=(32,),
        name=0,
        fin=0,
        fout=0,
        scale=None):
        net = {}
        net['OUT'] =  {"op":"Identity","params":{},
                                             "in": ["IN"],
                                            }
        
        model = {}
        model["net"] = net
        model["params"] = {"dim":dim}
        return model    

def Patchnet_factory(dim=2,
        fin=1,
        fout=1,
        features = 5,
        depth=3,
        pfield=(32,),
        size=8,
        out_channels_v = 6,
        out_channels_k = 5,
        upsample_features = -1,
        mt = 2,
        name=None,
        res="cat",
        mlp=False,
        scale=None
        ):
    net = {}
    
    
    if type(fin)!=list:
        fin=[fin,0]
    
    print("patchblock pfield {}".format(pfield[0]))
    for d in range(depth):
        ln = 'l'+str(d)# if d<depth-1 else "OUT"
        
        net[ln] =  {"op":"patch_block",
                              "params":{#"input_channels":fin[0]+fin[1] if d==0 else None,
                                        "input_channels":fin[0]+fin[1] if d==0 else features,
                                        #"output_channels":features,# if d<depth-1 else fout, 
                                        "output_channels":features,# if d<depth-1 else fout, 
                                        "pfield":pfield[0] if d==0 else size,
                                        "pfield_out":pfield[0] if d == depth-1 else size,
                                        "out_channels_v":out_channels_v,
                                        "out_channels_k":out_channels_k,
                                        "upsample_features":upsample_features,
                                        "mt":mt,
                                        "size":size,# if d==0 else 1,
                                        "upscale_out":False if d<depth-1 else True,
                                        "res":res,
                                        "mlp":mlp,
                                        },  
                              "in":["IN"] if d == 0 else ['l'+str(d-1)],
                              }
        
    net['OUT'] =  {"op":"Conv","params":{"in_channels":features,
                                         "out_channels":fout,
                                         "kernel_size":1,
                                         "bias":True},
                                         "in": ['l'+str(depth-1)],
                                        }
    model = {}
    model["net"] = net
    model["params"] = {"dim":dim}
    return model




def CC_Feature_net(
        dim=2,
        fin=1,
       # fout=1,
        depth=3,
        norm='BatchNorm',
        conv_block="conv_block",
        dropouts=0.0,
        features=16,
        pfield=(32,),
        name=None,
        probabilistic_enc=False,
        propagation_act = None,
        d=[5,5,5],
        normalized=False,
        eps=0.0001,
        scale=None,
        ):

    

    
    pow_of_2 = lambda n: (n & (n-1) == 0) and n != 0
    
    pfield = pfield if len(pfield)==dim else pfield*dim
    for p in pfield:
        if not pow_of_2(p):
            print("warning: pfield not power of 2")
    
    imsize = np.array(pfield)
    net = {}
    
    
    cc_in = ["IN0","IN1"]
    for d in range(depth):            
        enc_dec_kernel_size = np.array((3,)*dim)
        enc_dec_kernel_size[imsize<3] = 1
        enc_dec_padding = np.array((1,)*dim)
        enc_dec_padding[imsize<3] = 0
        cc_in =  []
        for in_keys,pf in zip(["IN0","IN1"],["c0","c1"]):
            cc_in += ['enc_'+pf+"_"+str(d)] 
            in_channels = fin
            net['enc_'+pf+"_"+str(d)] =  {"op":conv_block,
                                      "params":{"input_channels":in_channels if d == 0 else features,
                                                "output_channels":features,
                                                "norm":norm,
                                                "dropouts":dropouts,
                                                "kernel_size":tuple(enc_dec_kernel_size),
                                                "padding":tuple(enc_dec_padding),
                                                #"probabilistic":False if d<depth else probabilistic_enc,
                                                "probabilistic":probabilistic_enc,
                                                },
                                                "in":[in_keys] if d == 0 else ['enc_'+pf+"_"+str(d-1)],
                                                }
            
    
    
    net["CC"] = {'op':"CC_layer",
                 "params":{"d":d,
                           "pfield":pfield,
                           "normalized":normalized,
                           "eps":eps
                     },
                 "args":"none",
                 "in":cc_in
                 }
    
    net['OUT'] =  {"op":"Identity","params":{},
                        "in": ["CC"]+cc_in,
                        }

     
    
    model = {}
    model["net"] = net
    model["params"] = {"dim":dim}
    return model
        
def FFT_Net_factory(
        dim=2,
        fin=1,
        fout=1,
        depth=3,
        norm='BatchNorm',
        K=5,
        conv_block="conv_block",
        dropouts=0.0,
        feat_fun=lambda d:8*2**d,
        pfield=(32,),
        name=None,
        probabilistic_enc=False,
        propagation_act = None,
        n_labels = None,
        scale=None,
        ):

    #print("mode: {}".format(mode))
    if type(fin)!=list:
        fin=[fin,0]
    print("fin {}".format(fin))
    
    feat_fun = tls.string2fun(feat_fun)
    
    pow_of_2 = lambda n: (n & (n-1) == 0) and n != 0
    
    pfield = pfield if len(pfield)==dim else pfield*dim
    for p in pfield:
        if not pow_of_2(p):
            print("warning: pfield not power of 2")
    
    imsize = np.array(pfield)
    net = {}
    
    prop_in = "IN1"
    if propagation_act is not None and fin[1]>0:
        net['prop_in'] = {"op":propagation_act,
                       "params":{},
                        "in":["IN1"]}
        prop_in = "prop_in"
    
    for d in range(depth):            
        enc_dec_kernel_size = np.array((3,)*dim)
        enc_dec_kernel_size[imsize<3] = 1
        enc_dec_padding = np.array((1,)*dim)
        enc_dec_padding[imsize<3] = 0
        
        in_keys = ["IN0",prop_in] if fin[1]>0 else ["IN0"]
        in_channels = fin[0]+fin[1] 
        net['enc'+str(d)] =  {"op":conv_block,
                                  "params":{"input_channels":in_channels if d == 0 else feat_fun(d),
                                            "output_channels":feat_fun(d),
                                            "norm":norm,
                                            "dropouts":dropouts,
                                            "kernel_size":tuple(enc_dec_kernel_size),
                                            "padding":tuple(enc_dec_padding),
                                            #"probabilistic":False if d<depth else probabilistic_enc,
                                            "probabilistic":probabilistic_enc,
                                            },
                                            "in":in_keys if d == 0 else ['down'+str(d-1)],
                                            }
        
        
        #if d < depth:
        down_kernel_size = np.minimum(2,imsize)
        down_kernel_stride = np.array((2,)*dim)
        down_kernel_stride[down_kernel_size<2] = 1
            
        net['down'+str(d)] =  {"op":conv_block,
                               "params":{"input_channels":feat_fun(d),
                                        "output_channels":feat_fun(d+1),
                                        "btype":'down',
                                        "norm":norm,
                                        "kernel_size":tuple(down_kernel_size),
                                        "stride":tuple(down_kernel_stride)},
                                        "in":['enc'+str(d)],
                                        } 
            
        imsize = np.maximum(imsize // 2,1)
     
    
    n_coeff =  ifft_layer.get_input_length(K,dim,pfield)
    d = depth
    net['down_2_coeff'] =  {"op":conv_block,
                           "params":{"input_channels":feat_fun(d),
                                    "output_channels":fout*n_coeff*2,
                                    "btype":'down',
                                    "norm":norm,
                                    "act":nn.Identity,
                                    "kernel_size":tuple(imsize),
                                    "stride":tuple(imsize)},
                                    "in":['down'+str(d-1)],
                                    } 
    
     
    net["OUT"] = {"op":"ifft_layer",
                   "params":{"K":K,
                            "pfield":pfield},
                   "in":['down_2_coeff'],
                   }

    model = {}
    model["net"] = net
    model["params"] = {"dim":dim}
    return model
        
def UNet_factory(
        dim=2,
        fin=1,
        fout=1,
        depth=3,
        noskip=[],
        norm='BatchNorm',
        conv_block="conv_block",
        upsampling = "conv",
        dropouts=0.0,
        skip_type="3x3",
        feat_fun=lambda d:8*2**d,
        skip_feat_fun=None,
        up_feat_fun=None,
        pfield=(32,),
        name=None,
        probabilistic_enc=False,
        mode="standard",
        propagation_act = None,
        n_labels = None,
        scale=None,
        ):

    #print("mode: {}".format(mode))
    if type(fin)!=list:
        fin=[fin,0]
    print("fin {}".format(fin))
    
    groups = 1
    if n_labels is not None and fout!=n_labels:
        groups = 2    
    
    
    
    skip_feat_fun = feat_fun if skip_feat_fun is None else skip_feat_fun
    up_feat_fun = feat_fun if up_feat_fun is None else up_feat_fun
    
    skip_feat_fun = tls.string2fun(skip_feat_fun)
    up_feat_fun = tls.string2fun(up_feat_fun)
    feat_fun = tls.string2fun(feat_fun)
    
    pow_of_2 = lambda n: (n & (n-1) == 0) and n != 0
    
    pfield = pfield if len(pfield)==dim else pfield*dim
    for p in pfield:
        #assert(pow_of_2(p))
        if not pow_of_2(p):
            print("warning: pfield not power of 2")
    

    #main = nn.ModuleDict()
    #init = False
    
    #ksize = {}
    imsize = np.array(pfield)
    
    net = {}
    
    prop_in = "IN1"
    if propagation_act is not None and fin[1]>0:
        net['prop_in'] = {"op":propagation_act,
                       "params":{},
                        "in":["IN1"]}
        prop_in = "prop_in"
    
    for d in range(depth+1):            
        enc_dec_kernel_size = np.array((3,)*dim)
        enc_dec_kernel_size[imsize<3] = 1
        enc_dec_padding = np.array((1,)*dim)
        enc_dec_padding[imsize<3] = 0
        
        in_keys = ["IN0",prop_in] if mode in ["standard","diff"] and fin[1]>0 else ["IN0"]
        in_channels = fin[0]+fin[1] if mode in ["standard","diff"] else fin[0]
        net['enc'+str(d)] =  {"op":conv_block,
                                  "params":{"input_channels":in_channels if d == 0 else feat_fun(d),
                                            "output_channels":feat_fun(d),
                                            "norm":norm,
                                            "dropouts":dropouts,
                                            "kernel_size":tuple(enc_dec_kernel_size),
                                            "padding":tuple(enc_dec_padding),
                                            #"probabilistic":False if d<depth else probabilistic_enc,
                                            "probabilistic":probabilistic_enc,
                                            },
                                            "in":in_keys if d == 0 else ['down'+str(d-1)],
                                            }
        if d < depth:
            
            fin_ = up_feat_fun(d) + (0 if d in noskip else skip_feat_fun(d))
            fin_ += fin[1] if (mode == "forward" and d==0) else 0                
            #fin_ += 0 if type(fin) != tuple or len(fin) == 1 else fin[1]
            net['dec'+str(d)] =  {"op":conv_block,
                                  "params":{"input_channels":fin_,
                                            "output_channels":feat_fun(d),#//groups, 
                                            "norm":norm,
                                            "dropouts":0,
                                            "kernel_size":tuple(enc_dec_kernel_size),
                                            "padding":tuple(enc_dec_padding),
                                            "groups":groups},
                                            "in":['up'+str(d+1)] + ([prop_in] if  (mode == "forward" and d==0 and  fin[1]>0 )  else []),
                                            }                

        #use the previous (d-1) down_kernel_size for the up convolutions
        if d > 0:
            net['up'+str(d)] =  {"op":conv_block,
                                  "params":{"input_channels": feat_fun(d),
                                            "output_channels":up_feat_fun(d-1),
                                            "btype":"up" if upsampling == "conv" else "upsample",
                                            "norm":norm,
                                            "kernel_size":tuple(down_kernel_size),
                                            "stride":tuple(down_kernel_stride),
                                            "groups":groups},
                                            "in":['enc'+str(d)] if d ==  depth else ['dec'+str(d)],
                                            } 

                                           
        if d < depth:
            down_kernel_size = np.minimum(2,imsize)
            down_kernel_stride = np.array((2,)*dim)
            down_kernel_stride[down_kernel_size<2] = 1
                
            net['down'+str(d)] =  {"op":conv_block,
                                   "params":{"input_channels":feat_fun(d),
                                            "output_channels":feat_fun(d+1),
                                            "btype":'down',
                                            "norm":norm,
                                            "kernel_size":tuple(down_kernel_size),
                                            "stride":tuple(down_kernel_stride)},
                                            "in":['enc'+str(d)],
                                            } 
            
            
            if skip_type in ["3x3","1x1"] and (d not in noskip):
              #  print("NOSKIP : {}".format(noskip))
                if skip_type == "3x3":
                    net['skip'+str(d)] =  {"op":"conv_block",
                                  "params":{"input_channels":feat_fun(d),
                                            "output_channels":skip_feat_fun(d),
                                            "norm":norm,
                                            "dropouts":dropouts,
                                            "kernel_size":tuple(enc_dec_kernel_size),
                                            "padding":tuple(enc_dec_padding),
                                           # "probabilistic":probabilistic_enc,
                                            },
                                            "in":['enc'+str(d)],
                                            }
                    
                if skip_type == "1x1":
                    
                    net['skip'+str(d)] =  {"op":"conv_block",
                                  "params":{"input_channels":feat_fun(d),
                                            "output_channels":skip_feat_fun(d),
                                            "norm":norm,
                                            "dropouts":dropouts,
                                            "kernel_size":1,
                                            "padding":0,
                                          #  "probabilistic":probabilistic_enc,
                                          },
                                            "in":['enc'+str(d)],
                                            }
               
                net['dec'+str(d)]["in"] += ['skip'+str(d)]
            
            #else:
            #    self.main['skip'+str(d)] = torch.nn.Indentity() 
                
            
        imsize = np.maximum(imsize // 2,1)
            
    #self.main['out'] = layersND.Conv(self.dim, self.param["feat_fun"](0),self.param["fout"], 1, 1, 0, bias=True)
    
    if mode == "diff" and fin[1]>0:
        assert(groups==1)
        net['OUT_'] =  {"op":"Conv","params":{"in_channels":feat_fun(0),
                                             "out_channels":fout,
                                             "kernel_size":1,
                                             "bias":True},
                                             "in": ['dec'+str(0)],
                                            }
        net['OUT'] = {"op":"basic",
                      "params":{"btype":"add"},
                      "in":['OUT_',"IN1"],
                      "args":"none"}
    else:
        if groups == 1:

            net['OUT'] =  {"op":"Conv","params":{"in_channels":feat_fun(0),
                                                 "out_channels":fout,
                                                 "kernel_size":1,
                                                 "bias":True},
                                                 "in": ['dec'+str(0)],
                                                }
        else:
            net['OUT_label'] =  {"op":"Conv","params":{"in_channels":feat_fun(0)//2,
                                                 "out_channels":n_labels,
                                                 "kernel_size":1,
                                                 "bias":True},
                                                 "range":[0,feat_fun(0)//2],
                                                 "in": ['dec'+str(0)],
                                                }
            net['OUT_propagation'] =  {"op":"Conv","params":{"in_channels":feat_fun(0)//2,
                                             "out_channels":fout-n_labels,
                                             "kernel_size":1,
                                             "bias":True},
                                              "range":[feat_fun(0)//2,feat_fun(0)],
                                             "in": ['dec'+str(0)],
                                             #"in": ['dec'+str(0)+":"+str(feat_fun(0)//2)+":"+str(feat_fun(0))],
                                            }
   
            net['OUT'] =  {"op":"Identity","params":{},
                                                 "in": ["OUT_label","OUT_propagation"],
                                                }
    
    model = {}
    model["net"] = net
    model["params"] = {"dim":dim}
    return model


if False:
        
    from pw.units import model_factory, UNet_factory, layersND
   # import json
    model = {};
    model["net"] = {}
    model["net"]["l0"] = {"op":"Conv","params":{"in_channels":2,"out_channels":2,"kernel_size":3,"padding":1},"in":["IN"],"init":"default"}
    model["net"]["l1"] = {"op":"Conv","params":{"in_channels":2,"out_channels":1,"kernel_size":3,"padding":1},"in":["l0"],"init":"xaviar"}
    #model["net"]["OUT"] = {"op":"Conv","params":{"in_channels":3,"out_channels":1,"kernel_size":3,"padding":1},"in":["l0","l1"]}
    model["net"]["OUT"] = {"op":"basic","params":{"btype":"mult"},"in":["l0","l1"],"args":"args"}
    
    model["params"] = {"dim":2}
    #json_write(model,'tmp.json')
    
    
    
    if True:        
        mymodel = model_factory(model)
       
        mymodel.export_onnx("test.onnx",torch.ones([1,2,32,32]))
        y = mymodel(torch.ones([1,2,32,32]),verbose=True)        
               
        plt.imshow(y[0,0,:,:].detach()) 
        #torch.nn.Conv3d(**model["y"]["params"])
        #mymodel.reset_parameters()
    
    #%%
    import dill
    import pickle
    
    model = {};
    model["net"] = {}
    model["net"]["l0"] = {"op":layersND.Conv(2,in_channels=2,out_channels=2,kernel_size=3,padding=1),"in":["IN"]}
    model["net"]["l1"] = {"op":layersND.Conv(2,in_channels=2,out_channels=1,kernel_size=3,padding=1),"in":["l0"]}
    model["net"]["OUT"] = {"op":lambda x:x[0]+x[1],"in":["l0","l1"],"args":"args"}
    model["params"] = {"dim":2}
    
    
    dill.dump( model, open( "model.p", "wb" ) )
    test = dill.load( open( "model.p", "rb" ) )
    
    
    
    
    mymodel = model_factory(test)
    y = mymodel(torch.ones([1,2,32,32]),verbose=True)        
    plt.imshow(y[0,0,:,:].detach()) 
    #torch.nn.Conv3d(**model["y"]["params"])
    mymodel.reset_parameters()         
    #mymodel.export_onnx("test.onnx",torch.ones([1,2,32,32]))
    #%%
    defaults = {
                "dim":2,
                "fin":1,
                "fout":1,
                "depth":5,
                "noskip":[],
                "norm":'BatchNorm',
                "conv_block":"conv_block",
                "dropouts":0.0,
                "skip_type":"3x3",
                "feat_fun":lambda d:8*2**d,
                "skip_feat_fun":None,            
                "up_feat_fun":None,                        
                "pfield":(32,2),
                }    
            
    mynet = UNet_factory(**defaults)
    
    mymodel = model_factory(mynet,verbose=1)
    dummy_input = torch.randn([2,1,32,2])
    y = mymodel(dummy_input,verbose=2)        
    
    plt.imshow(y[0,0,:,:].detach()) 
    mymodel.reset_parameters()
    mymodel.export_onnx("test.onnx",dummy_input)
