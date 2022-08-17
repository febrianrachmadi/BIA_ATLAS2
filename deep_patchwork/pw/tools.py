import time
from numba import njit
import numpy as np
import random
import math
import json
import torch
from datetime import datetime
import os
import matplotlib.pyplot as plt
from . import batch_mat as bmat

PW_DEBUG = None
PW_DEBUG_LIST_AVAIL = [ "DEFAULT",
                        "ITENSITY_AUG",
                        "SAMPLING",
                        ]
PW_DEBUG_LIST = ["DEFAULT"]

def pw_debug(log,ptype="DEFAULT"):
    if (PW_DEBUG is not None) and (ptype in PW_DEBUG_LIST):
        now = datetime.now()
        date_time = now.strftime("%Y.%m.%d|%H:%M:%S") 
        hs = open(PW_DEBUG,"a")
        hs.write("{}#{}\n".format(date_time,log))
        hs.close() 

class cnt():
    pass

def rgb2gray(img):
       return np.minimum((0.2989 * img[...,0] + 0.5870 * img[...,1] + 0.1140 * img[...,2]).round(),255)[:,:,np.newaxis]
       

def to(param,dtype=tuple):
    if param is not None:
        if dtype == tuple:
            if type(param).__name__ in ["ndarray"]:
                param_ = [a.item() for a in param]
                param_ = dtype(param_)
                return param_
            elif type(param) in [tuple,list]:
                param_ = [(a.item() if type(a).__module__ == 'numpy' else a)  for a in param]
                param_ = dtype(param_)
                return param_
                

    return param

def set_default(var,value):
    return value if var is None else var

def string2fun(f):
    return eval(f) if type(f) == str else f

def get_n_parameters(model,trainable=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable))



def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        elif shell == 'SpyderShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
#%%
def gauss(w_size,sigma=None,norm="sum"):
    dim = len(w_size)
    if sigma is None:
        sigma = np.array(w_size)/2.0
    if dim == 2:
        dx = torch.arange(0,w_size[0])
        dy = torch.arange(0,w_size[1])
        XX,YY = torch.meshgrid(dx, dy, indexing="ij")
        XX = (XX - (w_size[0]//2)).float()
        YY = (YY - (w_size[1]//2)).float()        
        g_filt  = torch.exp(-(XX**2/sigma[0]**2+YY**2/sigma[1]**2))
    if dim == 3:
        dx = torch.arange(0,w_size[0])
        dy = torch.arange(0,w_size[1])
        dz = torch.arange(0,w_size[2])
        XX , YY, ZZ= torch.meshgrid(dx, dy , dz, indexing="ij")
        XX = (XX - (w_size[0]//2)).float()
        YY = (YY - (w_size[1]//2)).float()        
        ZZ = (YY - (w_size[2]//2)).float()        
        g_filt  = torch.exp(-(XX**2/sigma[0]**2+YY**2/sigma[1]**2+ZZ**2/sigma[2]**2))
    
    if norm == "sum":
        g_filt  /= torch.sum(g_filt)
    if norm == "max":
        g_filt  /= torch.max(g_filt)
    
    return g_filt

#%%
def im_grad(t):
    dim = len(t.shape)-2
    k = torch.tensor([-1,1],device=t.device)
    if dim == 3:
        g = torch.zeros([3,1,2,2,2],device=t.device)
        #print(g[0,0,:,0,0].shape)
        #print(k.shape)
        g[0,0,:,0,0] = k
        g[1,0,0,:,0] = k
        g[2,0,0,0,:] = k
        return torch.nn.functional.conv3d(t,g,padding='valid')
    else:
        g = torch.zeros([2,1,2,2],device=t.device)
        g[0,0,:,0] = k
        g[1,0,0,:] = k
        return torch.nn.functional.conv2d(t,g,padding='valid')
        
#%%
#https://github.com/pytorch/pytorch/issues/35674
def unravel_index(
    indices,#: torch.LongTensor,
    shape,#: Tuple[int, ...],
):# -> torch.LongTensor:
  

    shape = torch.tensor(shape)
    assert(torch.all(indices<shape.prod()))
    #indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        # indices = indices // dim      # CHANGED BY FEBRIAN
        indices = torch.div(indices, dim, rounding_mode='floor')

    return coord.flip(-1)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf





def create_rot_augmentation(scale_facts,
                            n_samples,
                            def_interv = [45,15],mode = "xyz",element_size=None):
    
    aug_rotations = {}
    n_scales = scale_facts.shape[1]
    dim = scale_facts.shape[0]
    for sidx in range(n_scales):
        #w = (n_scales - sidx - 1)/(n_scales-1)
        if n_scales == 1:
            w = 1
        else:
            w = 1-(sidx)/(n_scales-1)
        
        max_angle = (w*def_interv[0] + (1-w)*def_interv[1])/360
      
     #   print("w: {} {}".format(w,max_angle ))
        if dim == 3:
            aug_rotation = torch.zeros([n_samples,4])
            
            aug_rotation[:,:3] = (1-2*torch.rand(n_samples,3))#*max_angle
            #aug_rotation[:,:3] = max_angle
            
            if mode=="xy":
                aug_rotation[:,:3] = 0
                aug_rotation[:,2] = 1
            if mode=="xz":
                aug_rotation[:,:3] = 0
                aug_rotation[:,1] = 1    
            if mode=="yz":
                aug_rotation[:,:3] = 0
                aug_rotation[:,0] = 1    
            #elif mode=="xyz":
            #    aug_rotation[:,:3] = (1-2*torch.rand(n_samples,3))*max_angle
            elif mode=="x-y-z":
               # sign = lambda a: (a>0) - (a<0)
                assert(element_size!=None)
                assert(element_size[2]>=element_size[1])
                assert(element_size[2]>=element_size[0])
                bias = element_size[2]/element_size[0] - 1
                aug_rotation[:,2] += torch.sign(aug_rotation[:,2])*bias
                
            #aug_rotation[:,:3] = 0
            #aug_rotation[:,2] = 1
            #aug_rotation[:,1] = 0
            #aug_rotation[:,0] = 0
                
            aug_rotation[:,3] = max_angle*math.pi*2*(1-2*torch.rand(n_samples))
            
           # print("ACHTUNG, DEBUG SET iT TO given deg, without noise")
           # aug_rotation[:,3] = max_angle*math.pi*2
            #print("WARNING !!! ROT FIX!")
            
            aug_rotations[str(sidx)] = aug_rotation#[ux,uy,uz + ( -0.1 if uz < 0 else 0.1),max_angle*math.pi*2*(1-2*random.random())]
         
        else:
            aug_rotations[str(sidx)] = max_angle*math.pi*2*(1-2*torch.rand(n_samples,1))
            
    return aug_rotations




def create_scale_augmentation(scale_facts,
                              n_samples,
                              def_interv = [[0.5,2],[1,1]]):
    
    aug_scales = {}
    n_scales = scale_facts.shape[1]
    dim = scale_facts.shape[0]
    for sidx in range(n_scales):
        #w = (n_scales - sidx - 1)/(n_scales-1)
        #w = 1-(sidx)/(n_scales-1)
        if n_scales == 1:
            w = 1
        else:
            w = 1-(sidx)/(n_scales-1)
            
        min_scale = (w*def_interv[0][0] + (1-w)*def_interv[1][0])
        max_scale = (w*def_interv[0][1] + (1-w)*def_interv[1][1])
        
        aug_scale = min_scale +  torch.rand([n_samples,dim])*(max_scale-min_scale)
        aug_scales[str(sidx)] = aug_scale
        
    return aug_scales


def create_flip_augmentation(scale_facts,n_samples,dims=[0,1,2]):
     #print("dims {}".format(dims))
     n_scales = scale_facts.shape[1]
     dim = scale_facts.shape[0]
     aug_flips = {}
     for sidx in range(n_scales):
         flips = torch.rand([n_samples,dim])
         aug_flips[str(sidx)] = 1.0*(flips > 0.5) - 1.0*(flips < 0.5)
         for d in range(dim):
             if d not in dims:
                 aug_flips[str(sidx)][:,d] = 1
     return aug_flips
 


def create_random_rot_params(
                        n_samples,
                        dim,
                        max_angle=45,
                        mode = "xyz",
                        element_size=None):
   if dim == 3:
       aug_rotation = torch.zeros([n_samples,4])
       #aug_rotation[:,:3] = (1-2*torch.rand(n_samples,3))#*max_angle
       aug_rotation[:,:3] = torch.randn(n_samples,3)#*max_angle
       
       if mode=="xy":
           aug_rotation[:,:3] = 0
           aug_rotation[:,2] = 1
       if mode=="xz":
           aug_rotation[:,:3] = 0
           aug_rotation[:,1] = 1    
       if mode=="yz":
           aug_rotation[:,:3] = 0
           aug_rotation[:,0] = 1    
       if mode=="not_x":
           aug_rotation[:,0] = 0
       if mode=="not_y":
           aug_rotation[:,1] = 0
           #aug_rotation[:,2] = 0
       if mode=="not_z":
           aug_rotation[:,2] = 0
       if "x/y/z" in mode:
           amount = eval(mode.replace("x/y/z",""))
           aug_rotation[:,:3] *= amount
           aug_rotation[:,random.randint(0,2)] = 1               

       if "x#y#z" in mode:
           amount = eval(mode.replace("x#y#z",""))
           aug_rotation[:,:3] *= amount
               
       if mode=="x-y-z":
          # sign = lambda a: (a>0) - (a<0)
           assert(element_size!=None)
           assert(element_size[2]>=element_size[1])
           assert(element_size[2]>=element_size[0])
           bias = element_size[2]/element_size[0] - 1
           aug_rotation[:,2] += torch.sign(aug_rotation[:,2])*bias
       aug_rotation[:,3] = max_angle/360.0*math.pi*2*(1-2*torch.rand(n_samples))
       #aug_rotation[:,:] =  0
       #aug_rotation[:,0] = 1
       #aug_rotation[:,2] = 1
       #aug_rotation[:,3] = math.pi/2.0
      # print(aug_rotation[0,...])
    
   else:
       aug_rotation = max_angle/360.0*math.pi*2*(1-2*torch.rand(n_samples,1))
       
   return aug_rotation
  # bm = bmat.batch_mat(dim,dtype = torch.float32,device=aug_device)    
   #return  bm.quat2rot(aug_rotation)



def create_random_scale_params(
                              n_samples,
                              dim,
                              scale_interv = [0.5,2]):
    
    min_scale = scale_interv[0]
    max_scale = scale_interv[1]
    
    aug_scale = min_scale +  torch.rand([n_samples,dim])*(max_scale-min_scale)        
    return aug_scale


def create_random_flip_params(n_samples,
                              dim,
                              flip_dims=[0,1,2]):
    flips = torch.rand([n_samples,dim])
    aug_flips = 1.0*(flips > 0.5) - 1.0*(flips < 0.5)
    for d in range(dim):
        if d not in flip_dims:
            aug_flips[:,d] = 1
    return aug_flips


def create_random_shifts(
                              n_samples,
                              dim,
                              shifts):
    
    #shift_mat = bm.identity(n_samples)
    
    #shifts =  1.0-2*torch.rand([n_samples,dim])*(tt(shifts).flip(dims=(0,)))
    shifts =  1.0-2*torch.rand([n_samples,dim])*tt(shifts)
    return shifts



def draw_augmentation_mat(augmentation,
                          n_samples,
                          dim,
                          device="cpu",
                          element_size=None,
                          aug_state={}):
     
     if "shared_aug" in augmentation and augmentation["shared_aug"] is not None and augmentation["shared_aug"] and "aug_mat" in aug_state:
        aug_mat = aug_state["aug_mat"].clone()
       # print("aug_mat")
        return aug_mat,aug_state    
         
    
     #dim = scale_facts.shape[0]
     bm = bmat.batch_mat(dim,dtype = torch.float32,device=device)
     #aug_state = {}
     
     aug_mat = bm.batch_identity(n_samples)
     
     if "user_mat_0" in augmentation and augmentation["user_mat_0"] is not None:
         aug_mat = torch.matmul(aug_mat,augmentation["user_mat_0"].to(device)) 
     
     
     if "aug_scale" in augmentation and augmentation["aug_scale"] is not None:
        aug_scales = create_random_scale_params(
                                           n_samples=n_samples,
                                           dim=dim,
                                           scale_interv=augmentation["aug_scale"])
        
        
        if "aug_flipdims" in augmentation  and augmentation["aug_flipdims"] is not None:
            flips = create_random_flip_params(
                                            n_samples=n_samples,
                                            dim=dim,
                                            flip_dims=augmentation["aug_flipdims"])
            
           #print(flips)   
            aug_scales = aug_scales * flips

                
        scale_mat = bm.diag_mat(aug_scales)
        aug_mat = torch.matmul(aug_mat,scale_mat)
        
     elif "aug_flipdims" in augmentation  and augmentation["aug_flipdims"] is not None:
           flips = create_random_flip_params(
                                           n_samples=n_samples,
                                           dim=dim,
                                           flip_dims=augmentation["aug_flipdims"])
              
          # print(flips)
           aug_scales = flips
           scale_mat = bm.diag_mat(aug_scales)
           aug_mat = torch.matmul(aug_mat,scale_mat)
    
        
     if "aug_rot" in augmentation and augmentation["aug_rot"] is not None:
        rot_mode = "xyz" if not "aug_rot_mode" in augmentation else augmentation["aug_rot_mode"]
        
        
        #rot_mode = rot_mode if not ":" in rot_mode else (rot_mode.split(":")[1] if "aug_rot" in aug_state else rot_mode.split(":")[0])
        if rot_mode == "XYZ":
            aug_rotations_0 = create_random_rot_params(
                                               n_samples=n_samples,
                                               dim=dim,
                                               max_angle=augmentation["aug_rot"],
                                               mode = "yz",
                                               element_size=element_size)
            aug_rotations_1 = create_random_rot_params(
                                   n_samples=n_samples,
                                   dim=dim,
                                   max_angle=augmentation["aug_rot"],
                                   mode = "not_x",
                                   element_size=element_size)
        else:
            
            aug_rotations = create_random_rot_params(
                                               n_samples=n_samples,
                                               dim=dim,
                                               max_angle=augmentation["aug_rot"],
                                               mode = rot_mode,
                                               element_size=element_size)
        
        #aug_rotations[:,:3] = 0
        #aug_rotations[:,0] = 1
        #aug_rotations[:,2] = 1
        if "aug_rot_keep_axis" in augmentation and augmentation["aug_rot_keep_axis"] and  "aug_rot1_0" in aug_state:
                aug_rotations_0[:,:3] = aug_state["aug_rot1_0"][:,:3]
                aug_rotations_1[:,:3] = aug_state["aug_rot1_1"][:,:3]
        if "aug_rot_keep_axis" in augmentation and augmentation["aug_rot_keep_axis"] and  "aug_rot" in aug_state:
                aug_rotations[:,:3] = aug_state["aug_rot"][:,:3]
            
        if "aug_rot_keep_mat" in augmentation and augmentation["aug_rot_keep_mat"] and  "aug_rot1_0" in aug_state:
                aug_rotations_0 = aug_state["aug_rot1_0"].clone()
                aug_rotations_1 = aug_state["aug_rot1_1"].clone()
        if "aug_rot_keep_mat" in augmentation and augmentation["aug_rot_keep_mat"] and  "aug_rot" in aug_state:
                aug_rotations = aug_state["aug_rot"].clone()
            
        
        if rot_mode == "XYZ":
            aug_state["aug_rot1_0"] = aug_rotations_0.clone()
            aug_state["aug_rot1_1"] = aug_rotations_1.clone()
            rot_mat = bm.quat2rot(aug_rotations_0)
          #  print(rot_mat[0,...])
            aug_mat = torch.matmul(aug_mat,rot_mat)
            rot_mat = bm.quat2rot(aug_rotations_1)
            #print(rot_mat[0,...])
            aug_mat = torch.matmul(aug_mat,rot_mat)
       
        else:
            aug_state["aug_rot"] = aug_rotations.clone()
            rot_mat = bm.quat2rot(aug_rotations)
            #print(rot_mat[0,...])
            aug_mat = torch.matmul(aug_mat,rot_mat)
        
        
        
     if "aug_scale2" in augmentation and augmentation["aug_scale2"] is not None:
        aug_scales = create_random_scale_params(
                                           n_samples=n_samples,
                                           dim=dim,
                                           scale_interv=augmentation["aug_scale2"])
        scale_mat = bm.diag_mat(aug_scales)
        aug_mat = torch.matmul(aug_mat,scale_mat) 
        
     if "aug_rot_inv_after_scale" in augmentation and augmentation["aug_rot_inv_after_scale"] is not None and augmentation["aug_rot_inv_after_scale"]:
         if rot_mode == "XYZ":
             aug_rotations_0[:,3] *= -1
             aug_rotations_1[:,3] *= -1
             rot_mat = bm.quat2rot(aug_rotations_1)
             aug_mat = torch.matmul(aug_mat,rot_mat)
             rot_mat = bm.quat2rot(aug_rotations_0)
             aug_mat = torch.matmul(aug_mat,rot_mat)
         else:
             aug_rotations[:,3] *= -1
             rot_mat = bm.quat2rot(aug_rotations)
             aug_mat = torch.matmul(aug_mat,rot_mat)
        
        
     if "aug_rot2" in augmentation and augmentation["aug_rot2"] is not None:
        #rot_mode = "xyz" if not "aug_rot_mode" in augmentation else augmentation["aug_rot_mode"]
        rot_mode = rot_mode if not "aug_rot_mode2" in augmentation else augmentation["aug_rot_mode2"]
        
        #rot_mode = rot_mode if not ":" in rot_mode else (rot_mode.split(":")[1] if "aug_rot" in aug_state else rot_mode.split(":")[0])
        
        aug_rotations = create_random_rot_params(
                                           n_samples=n_samples,
                                           dim=dim,
                                           max_angle=augmentation["aug_rot2"],
                                           mode = rot_mode,
                                           element_size=element_size)
        
        if "aug_rot_keep_axis2" in augmentation and augmentation["aug_rot_keep_axis2"] and  "aug_rot2" in aug_state:
            aug_rotations[:,:3] = aug_state["aug_rot2"][:,:3]
            
        if "aug_rot_keep_mat2" in augmentation and augmentation["aug_rot_keep_mat2"] and  "aug_rot2" in aug_state:
            aug_rotations = aug_state["aug_rot2"].clone()
            
            
        aug_state["aug_rot2"] = aug_rotations.clone()
        rot_mat = bm.quat2rot(aug_rotations)
        aug_mat = torch.matmul(aug_mat,rot_mat)
        

     if "aug_rot3" in augmentation and augmentation["aug_rot3"] is not None:
        #rot_mode = "xyz" if not "aug_rot_mode" in augmentation else augmentation["aug_rot_mode"]
        rot_mode = rot_mode if not "aug_rot_mode3" in augmentation else augmentation["aug_rot_mode3"]
        
        #rot_mode = rot_mode if not ":" in rot_mode else (rot_mode.split(":")[1] if "aug_rot" in aug_state else rot_mode.split(":")[0])
        
        aug_rotations = create_random_rot_params(
                                           n_samples=n_samples,
                                           dim=dim,
                                           max_angle=augmentation["aug_rot3"],
                                           mode = rot_mode,
                                           element_size=element_size)
        
        if "aug_rot_keep_axis3" in augmentation and augmentation["aug_rot_keep_axis3"] and  "aug_rot3" in aug_state:
            aug_rotations[:,:3] = aug_state["aug_rot3"][:,:3]
            
        if "aug_rot_keep_mat3" in augmentation and augmentation["aug_rot_keep_mat3"] and  "aug_rot3" in aug_state:
            aug_rotations = aug_state["aug_rot3"].clone()
            
            
        aug_state["aug_rot3"] = aug_rotations.clone()
        rot_mat = bm.quat2rot(aug_rotations)
        aug_mat = torch.matmul(aug_mat,rot_mat)        
    
        
     if "aug_rot3D_z" in augmentation and augmentation["aug_rot3D_z"] is not None:
        rot_mode = "xy" 
        
        aug_rotations = create_random_rot_params(
                                           n_samples=n_samples,
                                           dim=dim,
                                           max_angle=augmentation["aug_rot3D_z"],
                                           mode = rot_mode,
                                           element_size=element_size)
        
        rot_mat = bm.quat2rot(aug_rotations)
        #aug_mat = torch.matmul(rot_mat,aug_mat)    
        aug_mat = torch.matmul(aug_mat,rot_mat)    
        
     if "shifts" in augmentation and augmentation["shifts"] is not None:
         shift_mat = bm.batch_identity(n_samples)
         shifts = create_random_shifts(
                                       n_samples,
                                       dim,
                                       augmentation["shifts"])
         #print("scale_mat ",scale_mat.shape)
         #print("shifts ",shifts.shape)
         shift_mat[:,-1,:dim] = shifts
         #shift_mat[:,:dim,-1] = shifts
        
         aug_mat = torch.matmul(aug_mat,shift_mat)
        # print(shift_mat)
         
     if "user_mat_1" in augmentation and augmentation["user_mat_1"] is not None:
         aug_mat = torch.matmul(aug_mat,augmentation["user_mat_1"].to(device)) 
         
     aug_state["aug_mat"] = aug_mat
    # print(aug_mat[0,...])
     return aug_mat,aug_state
         
         
#tt = lambda x,d: x if type(x) == torch.Tensor else torch.tensor(x,dtype=d)
def tt(tensor,dtype=None,device=None,requires_grad=False):
    if (type(tensor) in [int,float]):
        tensor = [tensor]
    #assert(type(tensor)!=float)
    
    if type(tensor) == torch.Tensor:
        #print("A")
        dtype = dtype if dtype is not None else tensor.dtype
        device = device if device is not None else tensor.device
        
        if not requires_grad and tensor.requires_grad:
            return tensor.detach().to(device=device,dtype=dtype)
        return tensor.to(device=device,dtype=dtype)
    device = device if device is not None else "cpu"
    if dtype == None:
        #print("B")
        return torch.tensor(tensor,requires_grad=requires_grad).to(device=device)
    #print("C")
    return torch.tensor(tensor,requires_grad=requires_grad).to(device=device,dtype=dtype)

def different_shape(t1,t2):
    return (len(t1.shape)!=len(t2.shape)) or torch.any(tt(t1.shape)!=tt(t2.shape))

def different(t1,t2):
    if different_shape(t1,t2):
        return True
    if torch.any(t1!=t2):
        return True
    return False


def get_bb_numpy(img,dim,leading_axis=1):
    bb = []
    for d in range(dim):
        DIMS=np.arange(dim+leading_axis)
        DIMS = DIMS[DIMS!=d+leading_axis]
        #print(tdata[key]["img"].shape)
        #print(np.abs(tdata[key]["img"]).max(axis=tuple(DIMS)).shape)
        indeces = np.where(np.abs(img).max(axis=tuple(DIMS))>0)[0]
        bb0 = indeces[0]
        bb1 = indeces[-1]
        #print("{} {}".format(bb0,bb1))
        bb += [[bb0,bb1]]
    return bb


def augs2mats(augmentation,scale_facts,num_samples,enable="r1r2s1s2f1"):
        augmentations = {}
        if "aug_rot" in augmentation and "r1" in enable:
            rot_mode = "xyz" if not "aug_rot_mode" in augmentation else augmentation["aug_rot_mode"]
            augmentations["aug_rotations"] = create_rot_augmentation(scale_facts,num_samples,def_interv=augmentation["aug_rot"],mode = rot_mode)
        if "aug_rot2" in augmentation and "r2" in enable:
            rot_mode = "xyz" if not "aug_rot_mode" in augmentation else augmentation["aug_rot_mode"]
            augmentations["aug_rotations2"] = create_rot_augmentation(scale_facts,num_samples,def_interv=augmentation["aug_rot2"],mode = rot_mode)
            
        if "aug_scale" in augmentation and "s1" in enable:    
            augmentations["aug_scales"] = create_scale_augmentation(scale_facts,num_samples,def_interv=augmentation["aug_scale"])       
        if "aug_scale2" in augmentation and "s2" in enable:    
            augmentations["aug_scales2"] = create_scale_augmentation(scale_facts,num_samples,def_interv=augmentation["aug_scale2"])      
            
        if "aug_flipdims" in augmentation and "f1" in enable:    
            if not "aug_scale" in augmentation:
                augmentations["aug_scales"] = create_scale_augmentation(scale_facts,num_samples,def_interv=[[1.0,1.0],[1.0,1.0]])       
            flips = create_flip_augmentation(scale_facts,num_samples,dims=augmentation["aug_flipdims"])
            
            for key in flips:
               # assert(False)
                augmentations["aug_scales"][key] = augmentations["aug_scales"][key] * flips[key]

        return augmentations
    
def estimate_scale_params(
                    tdata,
                    #element_sizes,
                    #img_size_voxels,
                    pfield,
                    cover = 0.5,
                    depth = 3,
                    coarse2fine_min_scale=1.3,
                    target_element_size_scale = 1.0,
                    img_size_reduction=torch.median,
                    target_size_reduction=torch.median,
                    ):
    if type(pfield) == tuple:
        pfield = torch.tensor(pfield)
    dim = len(pfield)
    n_data = len(tdata)
    
    target_element_size = torch.zeros(n_data,3)
    #target_element_size[:] = math.inf
    #for key in tdata:
    for key,idx in zip(tdata,range(n_data)):
        #target_element_size = torch.minimum(target_element_size,torch.tensor(tdata[key]["element_size"]))
        target_element_size[idx,:] = torch.tensor(tdata[key]["element_size"])
    #print("best element size {}".format(target_element_size))
    #print(target_element_size.shape)
    #print(target_size_reduction(target_element_size,dim=0).shape)
    
    print(target_element_size)
    target_element_size,_ = target_size_reduction(target_element_size,dim=0)
    target_element_size = target_element_size*tt(target_element_size_scale)
    
    
    
    smallest_img_size_mu = torch.zeros(n_data,3)
    #smallest_img_size_mu[:] = math.inf
    for key,idx in zip(tdata,range(n_data)):
    #for indx in range(len(element_sizes)):
        
        bb = get_bb_numpy(tdata[key]["img"],len(pfield))
        #print(bb)
        valid_img_shape = torch.tensor([bb[0][1]-bb[0][0]+1,bb[1][1]-bb[1][0]+1,bb[2][1]-bb[2][0]+1])
        img_size_mu = torch.tensor(tdata[key]["element_size"])*valid_img_shape           
        #img_size_mu = torch.tensor(tdata[key]["element_size"])*torch.tensor(tdata[key]["img"].shape[1::])
        
        #img_size_mu = element_sizes[indx]*img_size_voxels[indx]
        
        #smallest_img_size_mu = torch.minimum(smallest_img_size_mu,img_size_mu)
        smallest_img_size_mu[idx,:] = img_size_mu
       # print("{} {} {}".format(img_size_mu, element_sizes[indx],img_size_voxels[indx]))
    smallest_img_size_mu,_ = img_size_reduction(smallest_img_size_mu,dim=0)
    print(smallest_img_size_mu)
    
    coarsest_patch_size_mu = smallest_img_size_mu * cover#0.5
    detailed_patch_size_mu = target_element_size *  pfield
    
    print("coarsest patch size: {} ".format(coarsest_patch_size_mu))
    print("detailed patch size: {} ".format(detailed_patch_size_mu)) 
    
    if torch.any(detailed_patch_size_mu*coarse2fine_min_scale>coarsest_patch_size_mu):
        which_dim_is_small = detailed_patch_size_mu*coarse2fine_min_scale>coarsest_patch_size_mu
        print("####################################################")
        print("image is small, adjusting detailed_patch_size_mu :")
        print("before: {}".format(detailed_patch_size_mu))
        detailed_patch_size_mu[which_dim_is_small] = coarsest_patch_size_mu[which_dim_is_small]/coarse2fine_min_scale
        print("after: {}".format(detailed_patch_size_mu))
        print("####################################################")
        #print("image is small: {} ".format(coarsest_patch_size_mu))
        
    detailed_patch_size_mu = torch.minimum(coarsest_patch_size_mu,detailed_patch_size_mu)
    
    print("target_element_size before adjustment: {}".format(target_element_size))
    target_element_size = detailed_patch_size_mu / pfield
    print("target_element_size after adjustment: {}".format(target_element_size))
    #coarsest_patch_size_mu = torch.maximum(smallest_img_size_mu,detailed_patch_size_mu)
    s = 1
    if depth > 1:
        s = (coarsest_patch_size_mu/detailed_patch_size_mu).pow(1.0/(depth-1))
    print("coarsest patch size: {} ".format(coarsest_patch_size_mu))
    print("detailed patch size: {} ".format(detailed_patch_size_mu))
    print("s factor: {} ".format(s))
    
    
    scale_facts = torch.zeros([dim,depth],dtype=torch.float32)
    for d in range(depth):
        print("patch size for scale {} is {}".format(d,detailed_patch_size_mu*s**d))
        scale_facts[:,d] = s**d
    print("scale facts:\n {}".format(scale_facts))
    
    params = {}
    params["target_element_size"] = target_element_size
    params["scale_facts"] = scale_facts
    params["depth"] = depth
    params["smallest_img_size_mu"] = smallest_img_size_mu
    
    
    
    return params

def compute_patch_size_in_voxel(patch_element_size,
                                img_element_size,
                                pfield=(32,32,32)):
    

        #voxel_size = torch.tensor(pfield).type(torch.float)*torch.tensor(patch_element_size)/torch.tensor(img_element_size)
 
        return torch.tensor(pfield)*patch_element_size/img_element_size
 
 
def compute_scale_fact(patch_element_size,
                       shapes_mu,
                       depth,
                       pfield=32,
                       max_cover=0.5):
    
    virtual_shapes,_ = torch.min(shapes_mu/patch_element_size,dim=0)
    
    
    shape = max_cover*virtual_shapes/(32.0)
    fact = torch.pow(shape,torch.tensor(1.0/(depth-1.0)))
    scale_facts =  fact[:,None]**torch.arange(depth)
    

    return scale_facts

    

def arange_closed(start,end,steps,device="cpu"):
   # print("{} {} {}".format(start,end,steps))
    if (end-start)<0.000000000000001 and steps == 1:
        return torch.arange(start,start+1.0,steps,device=device)

    if steps == 1:
        return torch.full((steps,),(start+end)/2.0,device=device)
    #assert(steps > 1)
    
    step = (end-start)/(steps-1)
    #print("step {}".format(step))
    #print("steps {}".format(steps))
    return torch.arange(start,end+step/2.0,step,device=device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def json_read(filename):
        if (os.path.isfile(filename)):
            with open(filename) as data_file:
                data = json.load(data_file)
            isfile=True;
        else:
            data={}
            isfile=False;
        return data, isfile       
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def crop2shape_old(x,new_shape):
        new_shape_len = len(new_shape)
        old_shape_len = len(x.shape)
        crop_w = (x.shape[old_shape_len-2] - new_shape[new_shape_len-2])//2
        crop_h = (x.shape[old_shape_len-1] - new_shape[new_shape_len-1])//2
        
        if crop_w>0 or crop_h>0:
            return x[...,crop_w:x.shape[old_shape_len-1]-crop_w,crop_h:x.shape[old_shape_len-2]-crop_h]
        return x
    
    
def crop2shape(x,new_shape,offset=None,debug=False):   
    dim = len(x.shape) - 2      
    assert(dim>1)
    new_shape = np.array(new_shape[-dim:])
    old_shape = np.array(x.shape[-dim:])
    if offset is None:
        offset = (old_shape - new_shape) // 2 #+ np.array(offset[:dim])
    if debug:
        #print("crop before: {}".format((old_shape - new_shape) // 2))
        print("offset: {}".format(offset))
        #print("offset: {}".format(np.array(offset[:dim])))
    #if np.any(crop>0):
    if new_shape.size==2:
        return x[...,offset[0]:offset[0]+new_shape[0],offset[1]:offset[1]+new_shape[1]]
    return x[...,offset[0]:offset[0]+new_shape[0],offset[1]:offset[1]+new_shape[1],offset[2]:offset[2]+new_shape[2]]            
    #return x    
    
@njit
def rand_pic_array(pdf, maxvalue = 1):
    choice = random.uniform(0, maxvalue)
    left_pos = 0
    right_pos = pdf.size
    center = right_pos
    while (left_pos+1<right_pos):
        center = math.ceil((right_pos+left_pos)/2.0) 
        if (pdf[center-1]>=choice):
            right_pos=center
        else:
            left_pos=center
    return right_pos-1

#@njit
def pdf_2_cumsum(pdf):
       # print("yea wtf")
        pdf_64 = pdf.astype(dtype=np.float64)
        cumsum = np.cumsum(pdf_64.reshape(-1,))
        #cumsum = np.cumsum(pdf.reshape(-1))
        return (cumsum / cumsum[cumsum.size-1]).astype(dtype=np.float32)

def ind2sub(array_shape, ind):
    if len(array_shape) == 2:
        rows = (ind // array_shape[1])
        cols = (ind % array_shape[1])
        #return (rows, cols)
        #print(rows,cols)
        return (rows,cols)
    rows = ind // (array_shape[1]*array_shape[2])
    ind_ = ind - rows*(array_shape[1]*array_shape[2])
    cols = (ind_ // array_shape[2])
    level = (ind_ % array_shape[2])
    return (rows, cols, level)  


def vox_pos2physical_pos(pos,element_size,dtype=None,device=None):
    dtype = pos.dtype if dtype == None else dtype
    device = pos.device if device == None else device
#    return tt(vec)*tt(element_size)
    #return (tt(vec,dtype=dtype,device=device)-1)*tt(element_size,dtype=dtype,device=device)
    return (tt(pos,dtype=dtype,device=device))*tt(element_size,dtype=dtype,device=device)


def vox_shape2physical_shape(shape,element_size,dtype=None,device=None):
    dtype = shape.dtype if dtype == None else dtype
    device = shape.device if device == None else device
    return (tt(shape,dtype=dtype,device=device)-1)*tt(element_size,dtype=dtype,device=device)


#def physical2rel(vec,shape,dtype=None,device=None):
#    dtype = vec.dtype if dtype == None else dtype
#    device = vec.device if device == None else device
#    return (tt(vec,dtype=dtype,device=device)+1)/2.0*tt(shape-1,dtype=dtype,device=device)




class scatterer():
    grid_weights = None
    g_weight = False
    def scatter_2_img(self,grid,patch,
                      result,
                      result_counts=None,
                    #  batch_id=0,
                      votes=None):
            shape = result.shape[2:]
            nlabels = patch.shape[0]

            if False:
                sub_x = ((grid[...,0].view(-1)+1)/2.0*(shape[2]-1)).round()
                sub_y = ((grid[...,1].view(-1)+1)/2.0*(shape[1]-1)).round()
                sub_z = ((grid[...,2].view(-1)+1)/2.0*(shape[0]-1)).round()
                valid = torch.logical_and(sub_x >-1,  sub_x<shape[2])
                valid.logical_and_(sub_y > -1)
                valid.logical_and_(sub_y<shape[1])
                valid.logical_and_(sub_z > -1)
                valid.logical_and_(sub_z<shape[0])

            
            if len(shape) == 3:
                #sub_x = ((grid[batch_id,:,:,:,0].view(-1)+1)/2.0*(shape[2]-1)).round()
                #sub_y = ((grid[batch_id,:,:,:,1].view(-1)+1)/2.0*(shape[1]-1)).round()
                #sub_z = ((grid[batch_id,:,:,:,2].view(-1)+1)/2.0*(shape[0]-1)).round()
                sub_x = ((grid[...,0].view(-1)+1)/2.0*(shape[2]-1)).round().type(torch.LongTensor)
                sub_y = ((grid[...,1].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                sub_z = ((grid[...,2].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
                valid = torch.logical_and(sub_x >-1,  sub_x<shape[2])
                valid.logical_and_(sub_y > -1)
                valid.logical_and_(sub_y<shape[1])
                valid.logical_and_(sub_z > -1)
                valid.logical_and_(sub_z<shape[0])
                
                indx = ((sub_z*shape[1] + sub_y)*shape[2] + sub_x)#.type(torch.LongTensor)
                indx = indx[valid]
                


            else:
                #sub_x = ((grid[batch_id,:,:,0].view(-1)+1)/2.0*(shape[1]-1)).round()
                #sub_y = ((grid[batch_id,:,:,1].view(-1)+1)/2.0*(shape[0]-1)).round()
                sub_x = ((grid[...,0].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                sub_y = ((grid[...,1].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
            
                valid = torch.logical_and(sub_x >-1,  sub_x<shape[1])
                valid.logical_and_(sub_y > -1)
                valid.logical_and_(sub_y<shape[0])
            
                indx = (sub_y*shape[1] + sub_x)#.type(torch.LongTensor)
                indx = indx[valid]
                #print("vaild {}".format(torch.logical_not(valid).sum()))
            result_ = result.view(-1)
            for l in range(nlabels):
                #src  = patch[l,...].view(-1) 
                src  = patch[:,l,...]
                src = src.reshape(torch.prod(tt(src.shape)))
                if result_counts is not None:
                     result_.scatter_(0, indx+l*torch.prod(torch.tensor(shape)), src[valid].to(device=result_.device))
                     result_counts_ = result_counts.view(-1)
                     result_counts_[indx] = 1
                else:
                     #print(result_.device)
                     #print(src.device)
                     #print(valid.device)
                     result_.scatter_(0, indx+l*torch.prod(torch.tensor(shape)), src[valid].to(device=result_.device))
                if votes is not None:
                    votes.view(-1)[indx] += 1
                    
    def scatter_2_img_add(self,grid,patch,
                      result,
                      #grid_weights,
                      result_counts,
                      batch_id=0,
                      batch_size=None,
                      votes=None,
                      weighted = False):
            #weighted = True
            #weighted = False
            
            shape = result.shape[2:]
            #print("wtf 1")
           # print(shape)
            nlabels = patch.shape[1]
            
           

            #print(patch.shape)            
            if False:
                print(patch.shape)
                for b in range(patch.shape[0]):
                    bla2 = patch[b,0,:,:,:].amax(dim=2)
                    plt.imshow(bla2)
                    plt.pause(0.1)
            
                
            total_num_voxel = batch_size*torch.prod(tt(grid.shape[1:-1]))
            if self.grid_weights is None or self.grid_weights.shape[0] != total_num_voxel or self.grid_weights.device != result_counts.device or weighted != self.g_weight:
            
                self.g_weight = weighted
                if weighted:
                    p_shape = patch.shape[2:]
                    wg = gauss(p_shape,norm="max")[None,...]
                    ee = (batch_size,)+(-1,)*len(list(p_shape))
                    wg = wg.expand(ee).clone()  
                    self.grid_weights  = wg.view(-1).to(device=result.device)
                    print("rezising grid weights (gausian weighted)")
                else:
                    self.grid_weights = torch.ones(total_num_voxel).view(-1).to(device=result.device)            
                    #print(tmp.shape)
                    print("rezising grid weights")
            #grid_weights_ = grid_weights[:batch_size,...]

            if len(shape) == 3:
                #sub_x = ((grid[batch_id,:,:,:,0].view(-1)+1)/2.0*(shape[2]-1)).round().type(torch.LongTensor)
                #sub_y = ((grid[batch_id,:,:,:,1].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                #sub_z = ((grid[batch_id,:,:,:,2].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
                #if batch_size != -1:
                if True:
                    if False:    
                        sub_x = ((grid[batch_id:batch_size,...,0]+1.0)/2.0*(shape[2]-1.0))#.round().type(torch.LongTensor)
                        sub_y = ((grid[batch_id:batch_size,...,1]+1.0)/2.0*(shape[1]-1.0))#.round().type(torch.LongTensor)
                        sub_z = ((grid[batch_id:batch_size,...,2]+1.0)/2.0*(shape[0]-1.0))#.round().type(torch.LongTensor)
                        #spacing =   torch.sqrt(
                        #            (sub_x[:,0,0,0] - sub_x[:,1,1,1])**2 +
                        #            (sub_y[:,0,0,0] - sub_y[:,1,1,1])**2 + 
                        #            (sub_z[:,0,0,0] - sub_z[:,1,1,1])**2 )

                        r = 1
                        spacing = torch.abs(sub_x[:,0,0,0] - sub_x[:,1,1,1])
                        sub_x += (torch.randn(spacing.shape)*spacing)[:,None,None,None]  
                        spacing = torch.abs(sub_y[:,0,0,0] - sub_y[:,1,1,1])
                        sub_y += (torch.randn(spacing.shape)*spacing)[:,None,None,None]
                        spacing = torch.abs(sub_z[:,0,0,0] - sub_z[:,1,1,1])
                        sub_z += (torch.randn(spacing.shape)*spacing)[:,None,None,None]
                        
                        sub_x = sub_x.view(-1).round().type(torch.LongTensor)
                        sub_y = sub_y.view(-1).round().type(torch.LongTensor)
                        sub_z = sub_z.view(-1).round().type(torch.LongTensor)
                        
                        
                    else:
                        sub_x = ((grid[batch_id:batch_size,...,0].view(-1)+1.0)/2.0*(shape[2]-1.0)).round().type(torch.LongTensor)
                        sub_y = ((grid[batch_id:batch_size,...,1].view(-1)+1.0)/2.0*(shape[1]-1.0)).round().type(torch.LongTensor)
                        sub_z = ((grid[batch_id:batch_size,...,2].view(-1)+1.0)/2.0*(shape[0]-1.0)).round().type(torch.LongTensor)
                        
                        
                        
                    #sub_x = ((grid[:batch_size,...,0].view(-1)+1)/2.0*(shape[2]-1)).ceil().type(torch.LongTensor)
                    #sub_y = ((grid[:batch_size,...,1].view(-1)+1)/2.0*(shape[1]-1)).ceil().type(torch.LongTensor)
                    #sub_z = ((grid[:batch_size,...,2].view(-1)+1)/2.0*(shape[0]-1)).ceil().type(torch.LongTensor)
                    #sub_x = ((grid[:batch_size,...,0].view(-1)+1)/2.0*(shape[2]-1)).ceil().type(torch.LongTensor)
                    #sub_y = ((grid[:batch_size,...,1].view(-1)+1)/2.0*(shape[1]-1)).ceil().type(torch.LongTensor)
                    #sub_z = ((grid[:batch_size,...,2].view(-1)+1)/2.0*(shape[0]-1)).ceil().type(torch.LongTensor)
                else:
                    sub_x = ((grid[...,0].view(-1)+1)/2.0*(shape[2]-1)).round().type(torch.LongTensor)
                    sub_y = ((grid[...,1].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                    sub_z = ((grid[...,2].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
                valid = torch.logical_and(sub_x >-1,  sub_x<shape[2])
                valid.logical_and_(sub_y > -1)
                valid.logical_and_(sub_y<shape[1])
                valid.logical_and_(sub_z > -1)
                valid.logical_and_(sub_z<shape[0])
                
                indx = ((sub_z*shape[1] + sub_y)*shape[2] + sub_x)#.type(torch.LongTensor)
                indx = indx[valid]
               # if torch.any(indx>)                
            else:
                #sub_x = ((grid[batch_id,:,:,0].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                #sub_y = ((grid[batch_id,:,:,1].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
                #sub_x = ((grid[:batch_size,...,0].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                #sub_y = ((grid[:batch_size,...,1].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
                #sub_x = ((grid[:batch_size,...,0].view(-1)+1)/2.0*(shape[1]-1)).round().type(torch.LongTensor)
                #sub_y = ((grid[:batch_size,...,1].view(-1)+1)/2.0*(shape[0]-1)).round().type(torch.LongTensor)
                sub_x = ((grid[batch_id:batch_size,...,0].view(-1)+1)/2.0*(shape[1]-1)).ceil().type(torch.LongTensor)
                sub_y = ((grid[batch_id:batch_size,...,1].view(-1)+1)/2.0*(shape[0]-1)).ceil().type(torch.LongTensor)
                
            
                valid = torch.logical_and(sub_x >-1,  sub_x<shape[1])
                valid.logical_and_(sub_y > -1)
                valid.logical_and_(sub_y<shape[0])
            
                indx = (sub_y*shape[1] + sub_x)#.type(torch.LongTensor)
                indx = indx[valid]
             
            result_ = result.view(-1)
            for l in range(nlabels):
                #src  = patch[:,l,...].view(-1)
                #src  = patch[:,l,...]#.reshape(-1)
                #else:
                src  = patch[:batch_size,l,...]#.reshape(-1)                    
                src = src.reshape(torch.prod(tt(src.shape)))
                if result_counts is not None:
                    try:
                        if weighted:
                             tmp = self.grid_weights[valid]*src[valid]
                             result_.scatter_add_(0, indx.to(device=result_.device)+l*torch.prod(torch.tensor(shape)),tmp.to(device=result_.device))
                        else: 
                            result_.scatter_add_(0, indx.to(device=result_.device)+l*torch.prod(torch.tensor(shape)), src[valid].to(device=result_.device))
                    except:
                        print("min max sub_x {} {}".format(sub_x.min(),sub_x.max()))
                        print("min max sub_y {} {}".format(sub_y.min(),sub_y.max()))
                        print("min max sub_z {} {}".format(sub_z.min(),sub_z.max()))
                        print("shape {}".format(shape))

                    #result_counts_ = result_counts.view(-1)
                     #result_counts_[indx] = 1
                     #tmp = torch.tensor(1.0).to(device=result_.device)
                     #result_counts.view(-1).scatter_add_(0, indx, grid_weights[valid])

                else:
                     result_.scatter_(0, indx.to(device=result_.device)+l*torch.prod(torch.tensor(shape)), src[valid].to(device=result_.device))


            if result_counts is not None:
                
                result_counts.view(-1).scatter_add_(0, indx.to(device=result_counts.device), self.grid_weights[valid])

                
            if votes is not None:
                votes.view(-1)[indx] += 1
                
                
def aug_patchbatch(patchlist,dim,aug_int={}):
    if len(aug_int)==0:
        return
    #dim = 3
    if False:
        aug_int = {}
        aug_int["randn"] = (0.5,0.0,0.15)
        aug_int["brightness"] = (0.5,0.5,0,2)
        aug_int["contrast"] = (0.5,1,2)
        aug_int["gamma"] = (0.5,0.25,2)
        aug_int["poisson"] = (0.5,10,100)
        aug_int["invert"] = 0.5
        aug_int["clamp"] = (0.5,0.1,0.1)
        
    
    #aug_int["treshold"] = (0.5,0.5,0)
    
    
    #patchlist = img = dataset_train.img_patches.patchlist
    
    nb = patchlist[0].tensor.shape[0]
    device = patchlist[0].tensor.device
    dim = len(patchlist[0].tensor.shape)-2
    

    if "rand_hist" in aug_int:
#           f = lambda x,o,s,a: ((a*((x-o)/s))**torch.tensor([0,1,2,3])[None,None,None,:]).sum(dim=3)
        #shape_ = torch.tensor(img_.shape)
        hist_f = lambda x,o,s,a: ((a*((x-o)/s))**torch.tensor([0,1,2,3],device=device)[None,None,None,:]).sum(dim=3)
        hist_o = 0.5+torch.randn([nb,1,1,4],device=device)
        #hist_s = 1+torch.randn([nb,1,1,4],device=device)
        hist_s = 2.5*torch.rand([nb,1,1,4],device=device)+0.01
        hist_a = torch.randn([nb,1,1,4],device=device)
      #  hist_params = None
        

#    return f(x.reshape([shape_[0],shape_[1],shape_[2:].prod(),1]),o,s,a).reshape(tuple(shape_.numpy()))


   # print("bung")
    if "invert_(Imax)" in aug_int:
        naug = aug_int["invert_(Imax)"]
        shape = (nb,1,) + (1,) * dim
        invert_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])
        #print(invert_bs_on)
        #print(naug[0])


    if "invert_(mirrow)" in aug_int:
        naug = aug_int["x(mirrow)"]
        shape = (nb,1,) + (1,) * dim
        invert_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])

    if "clamp" in aug_int:
        naug = aug_int["clamp"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        clamp_low = torch.rand(shape,device=device)*r[0]
        clamp_high = torch.rand(shape,device=device)*r[1]
        clamp_bs_on = 1.0*(torch.rand(shape,device=device) <  naug["rate"])

    if "randn_abs" in aug_int:
        naug = aug_int["randn_abs"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        randn_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        randn_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])

    if "randn_rel" in aug_int:
        naug = aug_int["randn_rel"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        randn_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        randn_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])


    if "randn" in aug_int:
        naug = aug_int["randn"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        randn_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        randn_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])

        
    if "contrast" in aug_int:
        naug = aug_int["contrast"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        contrast_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        contrast_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])
        
    if "gamma" in aug_int:
        naug = aug_int["gamma"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        gamma_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        gamma_bs_on = 1.0*(torch.rand(shape,device=device) <naug["rate"])

    if "brightness" in aug_int:
        naug = aug_int["brightness"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        brightness_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        brightness_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])
        
    if "poisson" in aug_int:
        naug = aug_int["poisson"]
        shape = (nb,1,) + (1,) * dim
        #poisson_bs = torch.rand(shape,device=device) * (naug[2]-naug[1])+naug[1]
        #poisson_bs_on = 1.0*(torch.rand(shape,device=device) < naug[0])
        poisson_bs = torch.rand(shape,device=device) * (naug["range"][1]-naug["range"][0])+naug["range"][0]
        poisson_rand = torch.rand(shape,device=device) * (naug["add_uniform"][1]-naug["add_uniform"][0])+naug["add_uniform"][0]
        poisson_bs_on = 1.0*(torch.rand(shape,device=device) < naug["rate"])
        
    if "post_contrast" in aug_int:
        naug = aug_int["post_contrast"]
        shape = (nb,1,) + (1,) * dim
        r = naug["range"]
        post_contrast_bs = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        post_contrast_bs_on = 1.0*(torch.rand(shape,device=device) <  naug["rate"])
        
        
    if "rand_intensities" in aug_int:
        naug = aug_int["rand_intensities"]
        r = naug["range"]
        shape = (nb,256)
        #rand_intensities_on = 1.0*(torch.rand(shape,device=device) <  naug["rate"])
        #rand_intensities = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        
        kw = naug["kw"]
        rep = naug["rep"]

        rand_intensities = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
        kernel = torch.ones([1,1,kw],device=device)/(1.0*kw)
        for a in range(rep):
            rand_intensities = torch.nn.functional.conv1d(rand_intensities[:,None,:],kernel,padding='same')[:,0,:]
           # print(rand_intensities[0,0])
        #print(rand_intensities[0,-1])

    if "random_wipe" in aug_int:
        naug = aug_int["random_wipe"]
        t_ = patchlist[-1].tensor
        wipe_s = naug["max_wipe_size"]
        mask_augmentation = naug["mask_aug"]
        
        
        mask = torch.ones(t_.shape,device=t_.device,dtype=t_.dtype)
        #mask = torch.zeros(t_.shape,device=t_.device,dtype=t_.dtype)
        p_shape = tuple(t_.shape[2:])
        random_wipe_on = 1.0*(torch.rand(nb) < naug["rate"])
        
        
        if not "simple" in naug:
            #print("aug wipe")
            mask_center = tt(p_shape)//2
            mask_w = tt(p_shape)//4
            mask_start = mask_center - mask_w
            mask_end = torch.clamp(mask_center + mask_w,min=1)
            #print("mask ",mask_start," ",mask_end)
            
            grid = torch.ones((nb,)+p_shape+(dim+1,),dtype = torch.float32,device=t_.device)
            if dim == 3:       
                vox2rel_shift = torch.tensor([[0.0,0,1.0,0],
                            [0,1.0,0,0],
                            [1.0,0,0,0],
                            [0,0,0,1]],dtype = torch.float32,device=t_.device)
                
                mask[:,:,mask_start[0]:mask_end[0],mask_start[1]:mask_end[1],mask_start[2]:mask_end[2]] = 0
            
              #  print("p_shape ",p_shape)
                dx = arange_closed(-1,1,p_shape[0],device=t_.device)
                dy = arange_closed(-1,1,p_shape[1],device=t_.device)
                dz = arange_closed(-1,1,p_shape[2],device=t_.device)

                grid_x, grid_y , grid_z = torch.meshgrid(dx, dy, dz, indexing="ij")
                
                grid[:,...,0] = grid_x
                grid[:,...,1] = grid_y
                grid[:,...,2] = grid_z
                
                grid = torch.matmul(grid,vox2rel_shift)
                
                
                aug_mat,aug_state = draw_augmentation_mat(augmentation=mask_augmentation,
                                                    n_samples=nb,
                                                    dim=dim,
                                                    device=t_.device)
                
                grid = torch.einsum('b...x,bxy->b...y',grid ,aug_mat)
                
                mask = torch.nn.functional.grid_sample( 
                    mask,
                    grid[...,:dim], 
                    mode='nearest', 
                    padding_mode="border", 
                    align_corners=True)
            
            
        else:
            #print("simple wipe")
            which_dim = torch.randint(dim,(nb,))
            for b in range(nb):
                if random_wipe_on[b]:
                    mask_start = torch.zeros(dim,dtype=torch.int32)
                    mask_end = torch.zeros(dim,dtype=torch.int32)
                    for d in range(dim):
                        mask_start[d] = 0
                        mask_end[d] = p_shape[d]
                       # print(which_dim[b]," ",d)
                        if p_shape[d] > 1:
                            if which_dim[b]==d:
                                w_size = round(wipe_s*p_shape[d])
                                mask_size = torch.randint(w_size,(1,))
                                mask_start[d] = 0
                                mask_end[d] = torch.clamp(mask_size,max=p_shape[d])
                                #if True:
                                if torch.rand([1]).item() > 0.5:
                                    #tmp = mask_start[d]
                                    mask_start[d] = p_shape[d]-mask_end[d]-1
                                    mask_end[d] = p_shape[d]
                        else:
                            mask_start[d] = 0
                            mask_end[d] = 1
                    
                     
                    if (mask_start<mask_end).all():
                        #print(mask_start," ",mask_end)
                        if dim == 3:
                            mask[b,:,mask_start[0]:mask_end[0],mask_start[1]:mask_end[1],mask_start[2]:mask_end[2]] = 0
                        if dim == 2:
                            mask[b,:,mask_start[0]:mask_end[0],mask_start[1]:mask_end[1]] = 0
                    #else:
                     #   assert(False)
            if False:
                for b in range(nb):
                    if random_wipe_on[b]:
                            
                        mask_start = torch.zeros(dim,dtype=torch.int32)
                        mask_end = torch.zeros(dim,dtype=torch.int32)
                        for d in range(dim):
                            if p_shape[d] > 1:
                                mask_center = torch.randint(p_shape[d],(1,))
                                w_size = round(wipe_s*p_shape[d])
                                mask_size = torch.randint(w_size,(1,))//2
                                mask_start[d] = torch.clamp(mask_center-mask_size,min=0)
                                mask_end[d] = torch.clamp(mask_center+mask_size,max=p_shape[d])
                            else:
                                mask_start[d] = 0
                                mask_end[d] = 1
                        if (mask_start<mask_end).all():
                            if dim == 3:
                                mask[b,:,mask_start[0]:mask_end[0],mask_start[1]:mask_end[1],mask_start[2]:mask_end[2]] = 0
                            if dim == 2:
                                mask[b,:,mask_start[0]:mask_end[0],mask_start[1]:mask_end[1]] = 0
                  
                
        
 #   print(aug_int)
        
    pw_debug("begin augmentation of {} patch scales".format(len(patchlist)),"ITENSITY_AUG")    
    #for sc in range(len(patchlist)):
    
    coarsest_scale = len(patchlist)-1
    for sc in range(coarsest_scale,-1,-1):
        if len(aug_int)>0:
            pw_debug("begin augmentation of patch scale {}".format(sc),"ITENSITY_AUG")    
    
            img = patchlist[sc].tensor[:,...].clone()

            img_dims = (2,3) if dim==2 else (2,3,4)
            reshape = (img.shape[0],img.shape[1],) + (1,)*dim
            
            if "random_wipe" in aug_int:
                #print(patchlist[sc].grid.shape)
                #if True:
                
                img *= mask
                #img = mask.clone()
                #if sc < coarsest_scale:
                if sc > 0:
                    mask = torch.nn.functional.grid_sample( 
                        mask,
                        patchlist[sc-1].grid.to(device=mask.device), 
                        mode='nearest', 
                        padding_mode="zeros", 
                        align_corners=True)
                
            if "rand_intensities3" in aug_int:
               pw_debug("rand_intensities3","ITENSITY_AUG") 
               naug = aug_int["rand_intensities3"]
               r = naug["range"]
               nbins = naug["bins"]
               
               kw = naug["kw"]
               rep = naug["rep"]
               
               gr = naug["grange"]
               
               if "poisson" in naug and  naug["poisson"]:
                   p_shape = (nb,1,) + (1,) * dim
                   p_poisson_bs = torch.rand(p_shape,device=device) * (naug["prange"][1]-naug["prange"][0])+naug["prange"][0]
                   p_poisson_rand = torch.rand(p_shape,device=device) * (naug["add_uniform"][1]-naug["add_uniform"][0])+naug["add_uniform"][0]
             
               shape = (nb,nbins)
               rand_intensities = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
               kernel = torch.ones([1,1,kw],device=device)/(1.0*kw)
               for a in range(rep):
                   rand_intensities = torch.nn.functional.conv1d(rand_intensities[:,None,:],kernel,padding='same')[:,0,:]
               
               shape_g = (nb,1,) + (1,) * dim 
               gamma_ri3 = torch.rand(shape_g,device=device) * (gr[1]-gr[0])+gr[0]
               img = torch.sgn(img)*(torch.abs(img)**(gamma_ri3))
               
               if "poisson" in naug and  naug["poisson"]:
                   pnoise = torch.randn(img.shape,device=device)*p_poisson_rand
                   img = torch.poisson(torch.clamp(torch.trunc((img+pnoise)*p_poisson_bs),min=0)) 
                   
                   img /= img.reshape([img.shape[0],img.shape[1],-1]).amax(dim=(2,)).reshape(p_shape) + 0.01
                   
               for b in range(nb):
                   org_shape = img[b,...].shape
                   indeces = ((nbins-1)*img[b,...]).round().to(dtype=torch.long).reshape([-1])                   
                   imap = rand_intensities[b,:]
                   values = torch.gather(imap, 0,indeces) 
                   img[b,...] = values.reshape(org_shape)
                   
            if "rand_intensities2" in aug_int:
               pw_debug("rand_intensities2","ITENSITY_AUG") 
               naug = aug_int["rand_intensities2"]
               r = naug["range"]
               nbins = naug["bins"]
               shape = (nb,nbins)
               kw = naug["kw"]
               rep = naug["rep"]
               
               rand_intensities = torch.rand(shape,device=device) * (r[1]-r[0])+r[0]
               kernel = torch.ones([1,1,kw],device=device)/(1.0*kw)
               for a in range(rep):
                   rand_intensities = torch.nn.functional.conv1d(rand_intensities[:,None,:],kernel,padding='same')[:,0,:]
               
               for b in range(nb):
                   org_shape = img[b,...].shape
                   indeces = ((nbins-1)*img[b,...]).round().to(dtype=torch.long).reshape([-1])                   
                   imap = rand_intensities[b,:]
                   values = torch.gather(imap, 0,indeces) 
                   img[b,...] = values.reshape(org_shape)
                
            
            if "rand_intensities" in aug_int:
               pw_debug("rand_intensities","ITENSITY_AUG") 
              # print("nb ",nb)
               for b in range(nb):
                   org_shape = img[b,...].shape
                  # print(org_shape)
                   indeces = img[b,...].to(dtype=torch.long).reshape([-1])                   
                   imap = rand_intensities[b,:]
                   #print(imap.shape)
                   #assert(indeces.amin() >= 0)
                   #assert(indeces.amax() <= 255)
                   #print("indeces min max ",indeces.amin()," ",indeces.amax())
                 #  print(imap.shape)
                   values = torch.gather(imap, 0,indeces) 
                 #  print(values.shape)
                   #tmp = imap[indeces]
                   img[b,...] = values.reshape(org_shape)
                
        

            if "rand_hist" in aug_int:
               pw_debug("rand_hist","ITENSITY_AUG") 
               shape_ = torch.tensor(img.shape)

               img = img.reshape([shape_[0],shape_[1],shape_[2:].prod()])
               if True:
                   
                   if sc == coarsest_scale:
                       hist_pre_min = img.amin(dim=-1)[...,None]
                       hist_pre_max = (img.amax(dim=-1)[...,None] -hist_pre_min) + 0.00001 
                   
                   img -=  hist_pre_min
                   img /=  hist_pre_max
                   
               img =  hist_f(img[...,None],hist_o,hist_s,hist_a)
               
               if sc == coarsest_scale:
                   hist_dmin = img.amin(dim=-1)[...,None]
                   hist_dmax = (img.amax(dim=-1)[...,None]-hist_dmin) + 0.00001 
                   img -=  hist_dmin
                   hist_mean = img.mean(dim=-1)[...,None]+0.00000001
                   #hist_mean = (hist_dmax-hist_dmin)+0.00000001
               else:
                   img -=  hist_dmin
                   #hist_dmax = (img.amax(dim=-1)[...,None]-hist_dmin) + 0.00001 
                   
               img /=  hist_mean
                   
               img = torch.clamp(img,min=0)
               
               img =  img.reshape(tuple(shape_.numpy()))
               
              # assert(not img.isnan().any())

            if "invert" in aug_int:
                pw_debug("invert","ITENSITY_AUG") 
                if sc == coarsest_scale:
                    invert_max = img.amax(dim=img_dims).reshape(reshape)
                img = (invert_max - img) * (invert_bs_on) + img * (1-invert_bs_on)
                #assert(not img.isnan().any())
                
            if "clamp" in aug_int:
                pw_debug("clamp","ITENSITY_AUG") 
                if sc == coarsest_scale:
                    img_max = img.amax(dim=img_dims).reshape(reshape)
                img_ = torch.clamp(img,min=clamp_low*img_max,max=(1-clamp_high)*img_max) 
                img_ -= clamp_low*img_max
                
                img = img_ * clamp_bs_on + img * (1-clamp_bs_on)

            
            if "poisson" in aug_int:
                pw_debug("poisson","ITENSITY_AUG") 
               
                noise = torch.randn(img.shape,device=device)*poisson_rand
                img_ = torch.poisson(torch.clamp(torch.trunc((img+noise)*poisson_bs),min=0)) / poisson_bs
                #img_ = torch.poisson(torch.clamp(torch.trunc(img*poisson_bs),min=0)) / poisson_bs
                img = img_ * (poisson_bs_on) + img * (1-poisson_bs_on)
                
                
              #  assert(not img.isnan().any())
    
            if "brightness" in aug_int:
                pw_debug("brightness","ITENSITY_AUG")
                img += brightness_bs*brightness_bs_on
                
               # assert(not img.isnan().any())
                
            if "contrast" in aug_int:
                pw_debug("contrast","ITENSITY_AUG")
                #print(img.amax())
                img *= contrast_bs*contrast_bs_on + 1.0*(1-contrast_bs_on)
                
              #  assert(not img.isnan().any())
                
            if "gamma" in aug_int:
                pw_debug("gamma","ITENSITY_AUG")
                #img = img**(gamma_bs*gamma_bs_on + 1.0*(1-gamma_bs_on))
                img = torch.sgn(img)*(torch.abs(img)**(gamma_bs))*gamma_bs_on + img*(1-gamma_bs_on)
                
             #   assert(not img.isnan().any())
                
            if "randn_abs" in aug_int:
                pw_debug("randn_abs","ITENSITY_AUG")
                noise = torch.randn(img.shape,device=device)*randn_bs
                img += noise*randn_bs_on
 
            if "randn_rel" in aug_int:
                pw_debug("randn_rel","ITENSITY_AUG")
                if sc == coarsest_scale:
                    img_max_abs_randn_rel = img,abs().amax(dim=img_dims).reshape(reshape)
                noise = img_max_abs_randn_rel*torch.randn(img.shape,device=device)*randn_bs
                img += noise*randn_bs_on
            
 
    
            if "post_norm_/mean" in aug_int:
                shape_ = torch.tensor(img.shape)
                img =  img.reshape([shape_[0],shape_[1],shape_[2:].prod()])
                
                if sc == coarsest_scale:
                    norm_mean = img.mean(dim=-1)[...,None]+aug_int["post_norm_/mean"]["eps"]
                    #print(img.shape)
                    
                img /= norm_mean
                     
                img =  img.reshape(tuple(shape_.numpy()))
                
            if "post_contrast" in aug_int:
                pw_debug("post_contrast","ITENSITY_AUG")
                img *= post_contrast_bs*post_contrast_bs_on + 1.0*(1-post_contrast_bs_on )
                
            
            pw_debug("done","ITENSITY_AUG")
            patchlist[sc].tensor[:,...] = img
    pw_debug("done with intensity augmentations","ITENSITY_AUG")
                        
