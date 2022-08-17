import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#mp.set_start_method('spawn')
#from collections import OrderedDict

import random
import matplotlib.pyplot as plt
#import torch.optim as optim
import math
from . import tools as tls
#from .tools import tls

from .patch import  patchwriter

import scipy
import pw
from typing import Dict, Tuple, Sequence, List
import time
from .tools import pw_debug
from .settings import pw_settings 

class PWData():
    def __init__(self, 
                 n_modalities,
                 n_labels,
                 #pfield,
                 dim,
                 name="unnamed",
                 ):
      
        self.n_modalities = n_modalities
        #self.n_labels = n_labels
        self.n_labels = n_labels + 1 # one for valid buffer
        #self.pfield = pfield
        self.device=""
        self.dim = dim
        self.data_sampler = []
        self.meta_data = []
        self.scale_facts = None
        self.depth = None
        self.patchbuffersize = 0
        self.batch_size = 0
        self.pfield = None
        #self.data_type = []
        self.params_updated = True
        self.ready = False
        self.active_budder = 0
        self.name = name
        self.max_depth = -1
        self.weights = []

    def link_data(self,pw_data):
            self.data_sampler = pw_data.data_sampler 
            self.meta_data = pw_data.meta_data 
            self.weights = pw_data.weights 
            
    def add_dataset(self,img,
                        labels,
                        element_size = (1.0,1.0,1.0),
                        background_weight = 1,
                        ignore_labels_in_pdf=[],
                        normalization="none",
                        set_background_label = -1,
                        info=None,
                        debug=False,
                        keep_pdf=False,
                        meta=None,
                        pdf_img=None,
                        compute_bb = False,
                        weight = 1.0,
                        #data_type = "train"
                        ):
        
        self.data_sampler += [sampler(inputs = [img,labels],
                                              element_size=element_size,
                                              dim=self.dim,
                                              background_weight=background_weight,
                                              ignore_labels_in_pdf= [],
                                              normalization=normalization,
                                              set_background_label=set_background_label,
                                              debug=debug,
                                              info=info,
                                              keep_pdf=keep_pdf,
                                              pdf_img=pdf_img,
                                              compute_bb=compute_bb),
                                          ]
        self.meta_data += [meta]
        assert(weight>0)
        self.weights  += [weight]
        #self.data_type += [data_type]
        
    #def get_subset(self,data_type):
    #     return [data for t,data in zip(self.data_type,self.data_sampler) if t == data_type]
    
    #def get_subset_indx(self,data_type):
    #     return np.array([indx for t,indx in zip(self.data_type,range(len(self.data_type))) if t == data_type])
        
    
    def get_parameters(self,depth,
                            pfield,
                            target_element_size,
                            coarsest_scale,
                            detailed_scale,
                            ):
        
        pfield = tls.tt(pfield)
        dim = self.dim
        scale_facts = torch.zeros([dim,depth],dtype=torch.float32)
        
        coarsest_scale = tls.tt(coarsest_scale)
        detailed_scale = tls.tt(detailed_scale)
        
        s = 1
        if depth > 1:
            s = (coarsest_scale/detailed_scale).pow(1.0/(depth-1))
        for d in range(depth):
            scale_facts[:,d] = s**d
        
        params = {}
        params["target_element_size"] = target_element_size
        params["scale_facts"] = scale_facts
        params["depth"] = depth
        
        return params
        
    
    def estimate_parameters(self,cover=0.5,
                                 depth=3,
                                 pfield = (32,),
                                 coarse2fine_min_scale=1.3,
                                 target_element_size_scale = 1.0,
                                 img_size_reduction=torch.median,
                                 target_size_reduction=torch.median,
                                 isotropic = False
                                 #data_type="train"
                                 ):
        
        #pfield = self.pfield
        if type(pfield) == tuple:
            pfield = torch.tensor(pfield)
        #dim = len(pfield)
        dim = self.dim
        
        #sub_set = self.get_subset(data_type)
        
        n_data = len(self.data_sampler)
        #n_data = len(sub_set)
        
        target_element_size = torch.zeros(n_data,dim)
        smallest_img_size_mu = torch.zeros(n_data,dim)
        
        
        for ds,idx in zip(self.data_sampler,range(n_data)):
        #for ds,idx in zip(sub_set,range(n_data)):
            element_size = ds.pw_img.element_size
            valid_shape_mu = ds.pw_img.valid_shape_mu()
            target_element_size[idx,:] = torch.tensor(element_size)
            smallest_img_size_mu[idx,:] = valid_shape_mu
            
        #print(target_element_size)
        target_element_size,_ = target_size_reduction(target_element_size,dim=0)
        target_element_size = target_element_size*tls.tt(target_element_size_scale)
        if isotropic:
            target_element_size[:] = torch.amin(target_element_size)
         
        smallest_img_size_mu,_ = img_size_reduction(smallest_img_size_mu,dim=0)
        #print(smallest_img_size_mu)
        
       # print(coarsest_patch_size_mu)
        #if coarsest_patch_size_mu is not None:
        coarsest_patch_size_mu = smallest_img_size_mu * cover#0.5
        #if detailed_patch_size_mu is not None:
        detailed_patch_size_mu = target_element_size *  pfield
        
        if torch.any(detailed_patch_size_mu*coarse2fine_min_scale>coarsest_patch_size_mu):
            which_dim_is_small = detailed_patch_size_mu*coarse2fine_min_scale>coarsest_patch_size_mu
            print("####################################################")
            print("image is small, adjusting detailed_patch_size_mu :")
            print("before: {}".format(detailed_patch_size_mu))
            detailed_patch_size_mu[which_dim_is_small] = coarsest_patch_size_mu[which_dim_is_small]/coarse2fine_min_scale
            print("after: {}".format(detailed_patch_size_mu))
            print("####################################################")
        
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
            if isotropic:
                scale_facts[:,d] = torch.amin(scale_facts[:,d],dim=0)
            
            
        print("scale facts:\n {}".format(scale_facts))
        
        params = {}
        params["target_element_size"] = target_element_size
        params["scale_facts"] = scale_facts
        params["depth"] = depth
        params["smallest_img_size_mu"] = smallest_img_size_mu
        
        return params


    def resample(self,
                 scale_facts = None,
                 pfield  = (32,),
                 patchbuffersize=1000,
                 min_patches_per_dataset = 10,
                 target_element_size=[1.0,1.0,1.0],
                 target_device="cpu",
                 aug_device="cpu",
                 interpolation=["nearest","nearest"],
                 augmentation = {"aug_rotations":None,"aug_scales":None,"aug_flipdims":None},
                 sampling="n_coarse2fine",#"uniform_coarse2fine",
                 sample_from_label=-1,
                 weights = {'bg':1.0,'classes':[1]},
                 snap="valid",
                 copy_metadata=False,
                 progress_callback=None,
                 pin_mem=True,
                 sampler_debug_scales=-1,
                 patch_update_rate = 1,
                 max_depth = -1,
                 weighted_perm_replacement = None,
                 #data_type="train"
                 ):
      #  print("pin_mem ",pin_mem)
       # print("progress_callback ",progress_callback)
        
        self.ready = False
        self.max_depth = max_depth
        assert(len(self.data_sampler)>0)
        assert(scale_facts is not None)
        assert(pfield is not None)
        
        if len(pfield) == 1:
            pfield *= self.dim
        
        if self.patchbuffersize != patchbuffersize or self.scale_facts is None or tls.different(self.scale_facts,scale_facts) or pfield!=self.pfield:
            self.scale_facts = scale_facts   
            self.depth = scale_facts.shape[1]
            self.pfield = pfield
            self.params_updated = True
            print("changing buffer size {} -> {}".format(self.patchbuffersize,patchbuffersize))
            pw_debug("changing buffer size {} -> {}".format(self.patchbuffersize,patchbuffersize))

            self.patchbuffersize = patchbuffersize 
            self.img_patches = patch_batch(self.depth, 
                                           self.patchbuffersize,
                                           self.n_modalities,
                                           self.pfield,
                                           img_device=target_device,
                                            patch_device=target_device,
                                            aug_device=aug_device)
            self.labels_patches = patch_batch(self.depth, 
                                              self.patchbuffersize,
                                              self.n_labels,
                                              self.pfield,
                                           img_device=target_device,
                                            patch_device=target_device,
                                            aug_device=aug_device)
            if pin_mem:
                self.labels_patches.pin_memory()
                self.img_patches.pin_memory()
            
        
        n_data = len(self.data_sampler)
        
        n_patch_weights = torch.tensor(self.weights)
        n_patch_weights /= n_patch_weights.sum()
        
        #n_data_patches = np.maximum(np.round(patchbuffersize * n_patch_weights),min_patches_per_dataset).to(dtype=torch.int32)
        
        n_data_patches = np.maximum(np.ceil(patchbuffersize * n_patch_weights),min_patches_per_dataset).to(dtype=torch.int32)
        if weighted_perm_replacement is not None:
            data_indx = torch.utils.data.WeightedRandomSampler(n_patch_weights.view(-1), n_data, replacement=weighted_perm_replacement)
        else:
            data_indx = torch.randperm(n_data)
            
        
        start_t = time.time()
        
        self.interrupt_me = False    
        print("sampling {} - {} patches out of {} datasets ".format(n_data_patches.amin(),n_data_patches.amax(),n_data))
        pw_debug("sampling {} - {} patches out of {} datasets ".format(n_data_patches.amin(),n_data_patches.amax(),n_data))
        patch_indx = 0
        for progress,dindx in zip(range(len(data_indx)),data_indx):
                if self.interrupt_me:
                    break
                
                #num_samples = int(min(n_data_patches[dindx].item(),self.patchbuffersize-patch_indx))
                num_samples = min(n_data_patches[dindx].item(),self.patchbuffersize-patch_indx)
               # print("num_samples ",num_samples)
                if num_samples > 0:
                    self.data_sampler[dindx].sample_mu( batch_batch_img=self.img_patches,
                                                       batch_batch_labels=self.labels_patches,
                                                       batch_id=patch_indx,
                                                       num_samples=num_samples,
                                                       scale_facts = self.scale_facts,
                                                       pfield = self.pfield,
                                                       target_element_size_mu = target_element_size,
                                                       interpolation=interpolation,
                                                       target_device=target_device,
                                                       aug_device=aug_device,
                                                       augmentation=augmentation,
                                                       sampling=sampling,
                                                       sample_from_label=sample_from_label,
                                                       copy_metadata=copy_metadata,
                                                       debug_scales=sampler_debug_scales,
                                                       weights=weights,
                                                       snap=snap,
                                                       max_depth=max_depth)
                              
                
                    patch_indx += num_samples    
                
               # print(progress_callback)
                if progress_callback is not None:
                    
                    end_t = time.time()
                    patches_per_seconds = patch_indx/(end_t - start_t)
                    seconds_left = (patchbuffersize-patch_indx)/patches_per_seconds
                    progress_callback(progress,dindx,n_data,seconds_left,num_samples)
            



        end_t = time.time()
        dps = self.patchbuffersize/(end_t - start_t)
       # print("done")
        print("done: total time : {}, datasets per second : {}".format(end_t - start_t,dps))
        pw_debug("done: total time : {}, datasets per second : {}".format(end_t - start_t,dps))
        
        #print("datasets per second : {}".format(dps))

        print("done")
        self.ready = True
        return data_indx


    def resample_old(self,
                 scale_facts = None,
                 pfield  = (32,),
                 patchbuffersize=1000,
                 min_patches_per_dataset = 10,
                 target_element_size=[1.0,1.0,1.0],
                 target_device="cpu",
                 aug_device="cpu",
                 #device="cpu",
                 #interpolation=["bilinear","nearest"],
                 interpolation=["nearest","nearest"],
                 augmentation = {"aug_rotations":None,"aug_scales":None,"aug_flipdims":None},
                 sampling="n_coarse2fine",#"uniform_coarse2fine",
                 sample_from_label=-1,
                 weights = {'bg':1.0,'classes':[1]},
                 snap="valid",
                 copy_metadata=False,
                 progress_callback=None,
                 pin_mem=True,
                 sampler_debug_scales=-1,
                 patch_update_rate = 1,
                 max_depth = -1,
                 #data_type="train"
                 ):
        self.ready = False
        self.max_depth = max_depth
        assert(len(self.data_sampler)>0)
        assert(scale_facts is not None)
        assert(pfield is not None)
        
        if len(pfield) == 1:
            pfield *= self.dim
        
        if self.patchbuffersize != patchbuffersize or self.scale_facts is None or tls.different(self.scale_facts,scale_facts) or pfield!=self.pfield:
            self.scale_facts = scale_facts   
            self.depth = scale_facts.shape[1]
            self.pfield = pfield
            self.params_updated = True
            print("changing buffer size {} -> {}".format(self.patchbuffersize,patchbuffersize))
            pw_debug("changing buffer size {} -> {}".format(self.patchbuffersize,patchbuffersize))

            self.patchbuffersize = patchbuffersize 
            self.img_patches = patch_batch(self.depth, 
                                           self.patchbuffersize,
                                           self.n_modalities,
                                           self.pfield,
                                           img_device=target_device,
                                            patch_device=target_device,
                                            aug_device=aug_device)
            self.labels_patches = patch_batch(self.depth, 
                                              self.patchbuffersize,
                                              self.n_labels,
                                              self.pfield,
                                           img_device=target_device,
                                            patch_device=target_device,
                                            aug_device=aug_device)
            if pin_mem:
                self.labels_patches.pin_memory()
                self.img_patches.pin_memory()
            
        
        n_data = len(self.data_sampler)
        patches_per_dataset  = max(patchbuffersize // n_data,min_patches_per_dataset)
        n_data_samples = patchbuffersize // patches_per_dataset
        n_data_samples = min(n_data_samples,n_data) 
        data_indx = torch.randperm(n_data)[:n_data_samples]
        
        start_t = time.time()
        
       

        if patch_update_rate<1:
            self.img_patches.random_perm()
            self.labels_patches.random_perm()
            patches_per_dataset = max(1,round(patches_per_dataset*patch_update_rate))
        
        self.interrupt_me = False    
        print("sampling {} patches per {} datasets out of {} datasets in total ...".format(patches_per_dataset,n_data_samples,n_data))
        pw_debug("sampling {} patches per {} datasets out of {} datasets in total ...".format(patches_per_dataset,n_data_samples,n_data))
        patch_indx = 0
        for a in range(n_data_samples):
                if self.interrupt_me:
                    break
                #self.data_sampler[data_indx[a]].sampling_map[:] = 0
                
                num_samples = min(patches_per_dataset,self.patchbuffersize-patch_indx)
                if a == n_data_samples - 1:
                    num_samples = self.patchbuffersize-patch_indx
                    print("from the last image, we sample all {} reamining patches".format(num_samples))
                    
                #print("sampling from image index {}".format(data_indx[a]))
                
                self.data_sampler[data_indx[a]].sample_mu( batch_batch_img=self.img_patches,
                                                   batch_batch_labels=self.labels_patches,
                                                   batch_id=patch_indx,
                                                   num_samples=num_samples,
                                                   scale_facts = self.scale_facts,
                                                   pfield = self.pfield,
                                                   target_element_size_mu = target_element_size,
                                                   interpolation=interpolation,
                                                   target_device=target_device,
                                                   aug_device=aug_device,
                                                   augmentation=augmentation,
                                                   sampling=sampling,
                                                   sample_from_label=sample_from_label,
                                                   copy_metadata=copy_metadata,
                                                   debug_scales=sampler_debug_scales,
                                                   weights=weights,
                                                   snap=snap,
                                                   max_depth=max_depth)
                          
                
                patch_indx += patches_per_dataset    
                
                if progress_callback is not None:
                    end_t = time.time()
                    patches_per_seconds = patch_indx/(end_t - start_t)
                    seconds_left = (patchbuffersize-patch_indx)/patches_per_seconds
                    progress_callback(a,n_data_samples,seconds_left)
               # print("data indeces : {}".format(data_indx))



        end_t = time.time()
        dps = self.patchbuffersize/(end_t - start_t)
       # print("done")
        print("done: total time : {}, datasets per second : {}".format(end_t - start_t,dps))
        pw_debug("done: total time : {}, datasets per second : {}".format(end_t - start_t,dps))
        
        #print("datasets per second : {}".format(dps))

        print("done")
        self.ready = True
        return data_indx
    
    def set_buffer(self,
                    img_patches,
                    labels_patches,
                    patchbuffersize,
                    scale_facts,
                    pfield,
                 ):
        self.ready = False
        self.patchbuffersize = patchbuffersize
        self.scale_facts = scale_facts
        self.depth = scale_facts.shape[1]
        self.pfield = pfield
        self.params_updated = True
        
        self.img_patches = img_patches
        self.labels_patches = labels_patches
        
        self.ready = True
        return None
    
    def get_batch(self,indx,mask=None):
        assert(self.ready)
        self.img_patches.copy_to(self.img_batch_patches,indx,mask=mask)
        self.labels_patches.copy_to(self.labels_batch_patches,indx,mask=mask)
        return self.img_batch_patches,self.labels_batch_patches
    
    def batch(self,batch_size,device="cpu"):
            assert(self.dim is not None)
            
            assert(self.patchbuffersize>=batch_size)
            if self.batch_size != batch_size or self.device!=device or self.params_updated:
                self.params_updated = False
                print("reallocating sampling buffer (size {}). target device is {}".format(batch_size,device))
                self.batch_size = batch_size
                self.device = device
                self.img_batch_patches = patch_batch(self.depth, batch_size,self.n_modalities,self.pfield)
                self.labels_batch_patches = patch_batch(self.depth, batch_size,self.n_labels,self.pfield)
                if device != "cpu":
                    self.img_batch_patches.to_gpu()
                    self.labels_batch_patches.to_gpu()

                
            
            seq = torch.randperm(self.patchbuffersize)
            n_batches = self.patchbuffersize // batch_size
            stepsize = self.patchbuffersize // n_batches
            
            #print("self.patchbuffersize ",self.patchbuffersize)
            #print("batch_size ",batch_size)
            #print("n_batches ",n_batches)
            #print("stepsize ",stepsize)
            
            out = []
            last = 0
            while last <= len(seq)-stepsize:
            #while last < len(seq)-stepsize:
                out.append(seq[last:last + stepsize])          
                last += stepsize
            return out








class patch_item():
    def __init__(self, 
                 n_batches,
                 n_modalities,
                 pfield,
                 scale_indx,# = -1,
                 img_device=torch.device('cpu'),
                 patch_device=torch.device('cpu'),
                 aug_device=torch.device('cpu'),
                 img_dtype=torch.float32):
        #patch data
        self.scale_indx = scale_indx
        self.dim = len(pfield)
        self.tensor = torch.zeros([n_batches,n_modalities]+list(pfield), device=patch_device,
                                  requires_grad=False)
        
        #meta data
        
        #image coordinates
        self.rel_coordinates = torch.zeros((n_batches,)+pfield+(len(pfield),), device=img_device,
                                           requires_grad=False)
        
        #next coarser patch coordinates
        self.grid = torch.zeros((n_batches,)+pfield+(len(pfield),), device=patch_device,
                                requires_grad=False)

        #temp data        
        self.offset_rel_0 = torch.zeros([n_batches,len(pfield)],dtype=torch.float32,
                                        requires_grad=False)
        self.offset_rel_1 = torch.zeros([n_batches,len(pfield)],dtype=torch.float32,
                                        requires_grad=False)
        
        self.local_rot_mat_s =  torch.zeros((n_batches,)+(self.dim+1,)*2,dtype=torch.float32, device=aug_device,
                                            requires_grad=False)
        self.global_rot_mat_s =  torch.zeros((n_batches,)+(self.dim+1,)*2,dtype=torch.float32, device=aug_device,
                                             requires_grad=False)
        self.global_rot_mat_s_local =  torch.zeros((n_batches,)+(self.dim+1,)*2,dtype=torch.float32, device=aug_device,
                                                   requires_grad=False)
        
        self.aug_state = {}
        self.wobble = None
        
        #not implemented yet
        self.brain_aug = None
        
    
    def to_gpu(self):
        self.tensor = self.tensor.cuda()
        self.grid = self.grid.cuda()
    
    def to_cpu(self):
        self.tensor = self.tensor.cpu()
        self.grid = self.grid.cpu()

    def pin_memory(self):
        self.tensor.pin_memory()
    
    def clear(self,value=0):
        self.tensor.fill_(value)
        
        
    def copy_to(self,patch_item_target,indx,
                copy_data=True,
                copy_meta=True,
                copy_trafos=False,
                mask=None):
            
            if mask is not None:
                target_device = patch_item_target.tensor.device
                mask_t = mask#.to(device=patch_item_target.tensor.device)
                mask_s = mask#.to(device=self.tensor.device)
                
            if copy_data:
                #patch_item_target.tensor[:,...] = self.tensor[indx,...]
                if indx is None:
                    patch_item_target.tensor[:,...] = self.tensor.clone()
                else:
                    if mask is None:
                        patch_item_target.tensor[:,...] = self.tensor[indx,...]
                    else:
                        patch_item_target.tensor[mask_t,...] = self.tensor[indx[mask_s],...].to(device=target_device)
            
            if copy_meta:    
                if indx is None:
                    patch_item_target.grid[:,...] = self.grid.clone()
                    patch_item_target.rel_coordinates[:,...] = self.rel_coordinates.clone()
                    
                    patch_item_target.offset_rel_0[:,...] = self.offset_rel_0.clone()
                    patch_item_target.offset_rel_1[:,...] = self.offset_rel_1.clone()
                else:
                    if mask is None:
                        patch_item_target.grid[:,...] = self.grid[indx,...]            
                        patch_item_target.rel_coordinates[:,...] = self.rel_coordinates[indx,...]
                        
                        patch_item_target.offset_rel_0[:,...] = self.offset_rel_0[indx,...]
                        patch_item_target.offset_rel_1[:,...] = self.offset_rel_1[indx,...]
                    else:
                        patch_item_target.grid[mask_t,...] = self.grid[indx[mask_s],...].to(device=target_device)
                        patch_item_target.rel_coordinates[mask_t,...] = self.rel_coordinates[indx[mask_s],...]
                        
                        patch_item_target.offset_rel_0[mask_t,...] = self.offset_rel_0[indx[mask_s],...]
                        patch_item_target.offset_rel_1[mask_t,...] = self.offset_rel_1[indx[mask_s],...]
                        
            
            if copy_trafos:
                if indx is None:
                    patch_item_target.local_rot_mat_s[:,...]=self.local_rot_mat_s.clone()
                    patch_item_target.global_rot_mat_s[:,...]=self.global_rot_mat_s.clone()
                    patch_item_target.global_rot_mat_s_local[:,...]=self.global_rot_mat_s_local.clone()
                    if self.wobble is not None:
                        patch_item_target.wobble =  self.wobble.clone()
                        
                    if self.brain_aug is not None:
                        patch_item_target.brain_aug =  self.brain_aug.clone()
                        
                    for key in self.aug_state:
                        patch_item_target.aug_state[key] = self.aug_state[key].clone()
                else:
                    if mask is None:
                        patch_item_target.local_rot_mat_s[:,...]=self.local_rot_mat_s[indx,...]
                        patch_item_target.global_rot_mat_s[:,...]=self.global_rot_mat_s[indx,...]
                        patch_item_target.global_rot_mat_s_local[:,...]=self.global_rot_mat_s_local[indx,...]
                        if self.wobble is not None:
                            patch_item_target.wobble[:,...] =  self.wobble[indx,...]
                        
                        if self.brain_aug is not None:
                            patch_item_target.brain_aug[:,...] =  self.brain_aug[indx,...]
                        
                        for key in self.aug_state:
                            patch_item_target.aug_state[key][:,...] = self.aug_state[key][indx,...]
                    else:
                        patch_item_target.local_rot_mat_s[mask_t,...]=self.local_rot_mat_s[indx[mask_s],...]
                        patch_item_target.global_rot_mat_s[mask_t,...]=self.global_rot_mat_s[indx[mask_s],...]
                        patch_item_target.global_rot_mat_s_local[mask_t,...]=self.global_rot_mat_s_local[indx[mask_s],...]
                        if self.wobble is not None:
                            patch_item_target.wobble[mask_t,...] =  self.wobble[indx[mask_s],...]
                        
                        if self.brain_aug is not None:
                            patch_item_target.brain_aug[mask_t,...] =  self.brain_aug[indx[mask_s],...]
                        
                        for key in self.aug_state:
                            patch_item_target.aug_state[key][mask_t,...] = self.aug_state[key][indx[mask_s],...]
                        
                    
               # patch_item_target.global_patchwork_scales[:,...]=self.global_patchwork_scales[indx,...]
               # patch_item_target.global_patchwork_rot[:,...]=self.global_patchwork_scales[indx,...]
            


        
    def copy_to_patch_item(self,                           
                           pw_img,
                           idx,
                           patch_item_target=None,
                           interpolation='nearest',
                           copy_metadata=False,
                           #padding_mode='zeros'
                           padding_mode=None,
                           ):
        patch_item_target = patch_item_target if patch_item_target is not None else self
        t_s = pw_img.img
        rel_coordinates = self.rel_coordinates
        
        
        
        num_batches = len(idx)
        in_shape = (1,)+(len(idx)*rel_coordinates.shape[1],) + tuple(rel_coordinates.shape[2:-1])+(pw_img.dim,)
        
        
        out_shape = (num_batches,)+ tuple(rel_coordinates.shape[1:-1])
        if len(idx) != rel_coordinates.shape[0]:
            #print(patch_item_target.tensor.shape)
            #print("idx ",len(idx))
            #print("l max",patch_item_target.tensor.shape[1]-1)
            #print("t_s shape",t_s.shape)
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #print(t_s.shape)
            #print(grid_cpu[...,:pw_img.dim].reshape(in_shape))
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            for l in range(patch_item_target.tensor.shape[1]):
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
               # print(t_s[:,l,None,...].shape)
               # print(rel_coordinates[idx,...,:pw_img.dim].reshape(in_shape).shape)
                
                patch_item_target.tensor[idx,l,...] = torch.nn.functional.grid_sample(t_s[:,l,None,...], 
                                            rel_coordinates[idx,...,:pw_img.dim].reshape(in_shape), 
                                            mode=interpolation, 
                                            padding_mode=padding_mode, 
                                            align_corners=True).reshape(out_shape).to(device=patch_item_target.tensor.device)
        else:
           # print("full copy")
            for l in range(patch_item_target.tensor.shape[1]):
                patch_item_target.tensor[idx,l,...] = torch.nn.functional.grid_sample(t_s[:,l,None,...], 
                                            rel_coordinates[...,:pw_img.dim].reshape(in_shape), 
                                            mode=interpolation, 
                                            padding_mode=padding_mode, 
                                            align_corners=True).reshape(out_shape).to(device=patch_item_target.tensor.device)

                
        if copy_metadata:
            #print("copy meta data")
            if patch_item_target!=self:
                #print("not self: copying data")
                patch_item_target.rel_coordinates[idx,...] = self.rel_coordinates[idx,...]
                patch_item_target.grid[idx,...] = self.grid[idx,...]
        
        #SAMPLE BUFFER CHECK
        #debug = patch_item_target.tensor.amax(dim = (2,3,4))>0
        #print("SAMPLE BUFFER CHECK")
        #print(debug.sum(dim=0))


        #print("copy_to_patch_item t_s req grad {}".format(t_s.requires_grad))
        #print("patch_item_target.tensor req grad {}".format(patch_item_target.tensor.requires_grad))
            
class patch_batch():
    def __init__(self, n_scales, n_batches,n_modalities,
                 pfield,
                 #device=torch.device('cpu'),
                 img_device=torch.device('cpu'),
                 patch_device=torch.device('cpu'),
                 aug_device=torch.device('cpu')):
        self.dim = len(pfield)
        self.patchlist = [];
        self.n_batches = n_batches
        self.n_modalities = n_modalities
        self.pfield = pfield
        for s in range(n_scales):
            self.patchlist.append(patch_item(n_batches,
                                             n_modalities,
                                             pfield,
                                             scale_indx=s,
                                             img_device=img_device,
                                             patch_device=patch_device,
#                 device=device,
                                             aug_device=aug_device
                                             ))
        self.n_scales = n_scales
        
    def copy_to_patch_item(self,
                           pw_img,
                           idx,
                           patch_batch_target=None,                           
                           interpolation='nearest',
                           copy_metadata=False,
                           padding_mode=None
                           #padding_mode="zeros"
                           ):
        patch_batch_target = patch_batch_target if patch_batch_target is not None else self
        #print("interpolation:"+interpolation)
        for p,p2 in zip(self.patchlist,patch_batch_target.patchlist):
            p.copy_to_patch_item(
                                 pw_img,
                                 idx,
                                 p2,
                                 interpolation,
                                 copy_metadata,
                                 padding_mode=padding_mode)

    def get_patch_device(self):
        return self.patchlist[0].tensor.device
    
    def get_img_device(self):
        return self.patchlist[0].rel_coordinates.device

    def get_aug_device(self):
        return self.patchlist[0].local_rot_mat_s.device
    

    def to_gpu(self):
        for p in self.patchlist:
            p.to_gpu()
            
    def to_cpu(self):
        for p in self.patchlist:
            p.to_cpu()
            
    def pin_memory(self):
        for p in self.patchlist:
            p.pin_memory()
            
    def batch_max(self,label=None,min=0,scale=0):
        tmp = torch.clamp(self.patchlist[scale].tensor,min=min)
        if self.dim == 2:
            #return self.patchlist[0].tensor.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]
            return tmp.amax(dim=(1,2,3)) if label is None else tmp[:,label,...].amax(dim=(1,2))
        #return self.patchlist[0].tensor.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  if label is None else 
        return tmp.amax(dim=(1,2,3,4)) if label is None else tmp[:,label,...].amax(dim=(1,2,3))
    
    def batch_sum(self,label=None,scale=0):
        if self.dim == 2:
            #return self.patchlist[0].tensor.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]
            return self.patchlist[scale].tensor.sum(dim=(1,2,3)) if label is None else self.patchlist[scale].tensor[:,label,...].sum(dim=(1,2))
        #return self.patchlist[0].tensor.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]  if label is None else 
        return self.patchlist[scale].tensor.sum(dim=(1,2,3,4)) if label is None else self.patchlist[scale].tensor[:,label,...].sum(dim=(1,2,3))
    
    def batch_min(self):
        if self.dim == 2:
            return self.patchlist[0].tensor.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0]
        return self.patchlist[0].tensor.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].min(dim=1)[0]
        
    def clear(self,value=0):
        for p in self.patchlist:
            p.clear(value)
            

    #patch_batch.copy_to(patch_batch,topselect,range(s-1,coarsest_level),copy_meta=True,copy_trafos=True)    
    def copy_to(self,patch_batch2,
                indx,
                scale_indeces=None,
                copy_data=True,
                copy_meta=True,
                copy_trafos=False,
                mask=None):
        #assert(patch_batch2.patchlist[0].tensor.shape[0]==len(selection))
        #ran = range(self.n_scales) if scale_indeces is None else scale_indeces
        #for s in ran:#range(self.n_scales):
        scale_indeces = scale_indeces if scale_indeces is not None else range(self.n_scales)
        for s in scale_indeces:
        #for s in range(self.n_scales):      
            self.patchlist[s].copy_to(patch_batch2.patchlist[s],indx,copy_data,copy_meta,copy_trafos,mask=mask)
            
                
            
            
    def random_perm(self,indx=None):
        print("random_perm might be buggy")
        if indx is None:
            indx = torch.randperm(self.patchlist[0].tensor.shape[0])
        for s in range(self.n_scales):
            self.patchlist[s].copy_to(self.patchlist[s],indx,copy_data=True,copy_meta=True,copy_trafos=True)
            
            #self.patchlist[s].tensor[:,...] = self.patchlist[s].tensor[indx,...]
            #self.patchlist[s].grid[:,...] = self.patchlist[s].grid[indx,...]
            #self.patchlist[s].offset_rel[:,...] = self.patchlist[s].offset_rel[indx,...]
            #self.patchlist[s].offset_rel_0[:,...] = self.patchlist[s].offset_rel_0[indx,...]
            #self.patchlist[s].offset_rel_1[:,...] = self.patchlist[s].offset_rel_1[indx,...]
            #self.patchlist[s].rel_coordinates[:,...] = self.patchlist[s].rel_coordinates.offset_rel[indx,...]
        return indx

class pw_img_data():
    def get_shape(self):
        return self.img.shape[2::]
    
    def __init__(self, inputs, #scales,
                 element_size=(1.0,1.0,1.0),
                 dtype=torch.float32,
                 normalization="none",
                 quiet = True,
                 compute_bb=False):
        if not quiet:
            print("img normalization : {}".format(normalization))
        self.bb = None
        
        is_empty = False
        if type(inputs) == tuple:
            self.dim = len(inputs)-2
            is_empty = True
        else:
            self.dim = len(inputs.shape)-2
        
        if not quiet:
            print("dim is "+str(self.dim))
            
        self.element_size = element_size[:self.dim]
            
        
        if is_empty:
            self.img = torch.zeros(inputs,dtype=dtype,requires_grad=False)
        else:
            self.img = inputs.type(dtype)
        
        #print(type(inputs))
        #print(self.img.is_leaf)
        assert(not self.img.requires_grad)

        self.bb = [torch.zeros(self.dim,dtype=torch.int,requires_grad=False),torch.zeros(self.dim,dtype=torch.int,requires_grad=False)]  
            
        #THIS WILL FAIL IF THE ENTIRE IMAGE IS ZERO
        for d in range(self.dim):
            DIMS = np.arange(0,self.dim+2)
            DIMS = DIMS[DIMS!=d+2]
            #indeces = self.img.abs().amax(dim=tuple(DIMS)).nonzero()  
            indeces = torch.nonzero(self.img.abs().amax(dim=tuple(DIMS)))  
            
            if len(indeces) < 2 or not compute_bb:
                self.bb[0][d] = 0
                self.bb[1][d] = self.img.shape[d+2]
            else:
                self.bb[0][d] = indeces[0]
                self.bb[1][d] = indeces[-1]
            
        if normalization == "-mean/std":
                if self.dim == 3:
                    t = self.img[...,self.bb[0][0]:self.bb[1][0],
                                        self.bb[0][1]:self.bb[1][1],
                                        self.bb[0][2]:self.bb[1][2]]
                else:
                    t = self.img[...,self.bb[0][0]:self.bb[1][0],
                                        self.bb[0][1]:self.bb[1][1]]
                n_voxels = torch.tensor(t.shape[2:]).prod()
                std_,m_ = torch.std_mean(t.reshape([t.shape[0],t.shape[1],n_voxels]),dim=2,keepdim=True)
                std_ = std_[:,:,None] if self.dim == 2 else std_[:,:,None,None]
                gamma = 0.00001
                std_ += gamma
                m_ = m_[:,:,None] if self.dim == 2 else m_[:,:,None,None]
                self.img = (self.img-m_) / std_
                #self.std_ = std_
                #self.m_ = m_
        if normalization == "/mean":
                if self.dim == 3:
                    t = self.img[...,self.bb[0][0]:self.bb[1][0],
                                        self.bb[0][1]:self.bb[1][1],
                                        self.bb[0][2]:self.bb[1][2]].type(torch.float32)
                else:
                    t = self.img[...,self.bb[0][0]:self.bb[1][0],
                                        self.bb[0][1]:self.bb[1][1]].type(torch.float32)
                
                n_voxels = tls.tt(t.shape[2:]).prod()
                m_ = torch.mean(t.reshape([t.shape[0],t.shape[1],n_voxels]),dim=2,keepdim=True)
                gamma = 0.00001
                m_ += gamma
                #print(" {}".format(m_))
                m_ = m_[:,:,None] if self.dim == 2 else m_[:,:,None,None]
                #print("std/mean before : {}".format(torch.std_mean(self.tensors[0])))
                
                for l in range(self.img.shape[1]):   
                    if not quiet:
                        print("std/mean before : {} {}".format(l,torch.std_mean(self.img[0,l,...])))
                
                
                #print(self.tensors[0].shape)
                #print(self.tensors[0].shape)
              #  print(" {}".format(self.tensors[0].shape))
                #print(" {}".format(m_.shape))
                #print("m: {}".format(m_))
                
                self.img = (self.img) / m_.type(dtype)                
                #print("std/mean after : {}".format(torch.std_mean(self.tensors[0])))
            
                for l in range(self.img.shape[1]):   
                    if not quiet:
                        print("std/mean after : {} {}".format(l,torch.std_mean(self.img[0,l,...])))
            
       
        self.indx = None
        #self.create_indx()
        
    def bb_mu(self):
        #return [tls.tt(self.bb[0],dtype=torch.float32)*tls.tt(self.element_size,dtype=torch.float32),
        #        tls.tt(self.bb[1],dtype=torch.float32)*tls.tt(self.element_size,dtype=torch.float32)]
        #return [tls.vox_shape2physical_shape(self.bb[0],self.element_size,dtype=torch.float32,device=self.bb[0].device),
        #        tls.vox_shape2physical_shape(self.bb[1],self.element_size,dtype=torch.float32,device=self.bb[0].device)]
        return [tls.vox_pos2physical_pos(self.bb[0],self.element_size,dtype=torch.float32,device=self.bb[0].device),
                tls.vox_pos2physical_pos(self.bb[1],self.element_size,dtype=torch.float32,device=self.bb[0].device)]
    
    
    def shape_mu(self):
        #return tls.tt(self.tensors[0].shape[2:],dtype=torch.float32)*tls.tt(self.element_size,dtype=torch.float32)
        return tls.vox_shape2physical_shape(self.img.shape[2:],self.element_size,dtype=torch.float32,device=self.bb[0].device)
        
    def valid_shape_mu(self):
        bb_mu = self.bb_mu()
        return bb_mu[1] - bb_mu[0]
    
    def create_indx_(self):
        t = self.img
        
        grid_indx = torch.zeros(tuple(t.shape[2:])+(self.dim,),requires_grad=False)
        #print("self.dim {}".format(self.dim))
        if self.dim == 2:
            dx = torch.arange(0,t.shape[2])
            dy = torch.arange(0,t.shape[3])
            grid_y, grid_x = torch.meshgrid(dx, dy, indexing="ij")
            grid_indx[...,0] = 2.0 * (grid_x / (float(t.shape[2]-1)) -0.5) 
            grid_indx[...,1] = 2.0 * (grid_y / (float(t.shape[3]-1)) -0.5) 
        if self.dim == 3:
            dx = torch.arange(0,t.shape[2])
            dy = torch.arange(0,t.shape[3])
            dz = torch.arange(0,t.shape[4])

            grid_z, grid_y, grid_x = torch.meshgrid(dx, dy, dz, indexing="ij")
            grid_indx[...,0] = 2.0 * (grid_x / (float(t.shape[2]-1)) -0.5) 
            grid_indx[...,1] = 2.0 * (grid_y / (float(t.shape[3]-1)) -0.5) 
            grid_indx[...,2] = 2.0 * (grid_z / (float(t.shape[4]-1)) -0.5) 
        
        self.grid_indx = grid_indx
        #print(self.grid_indx)
        
    def clear(self,value=0):
        self.img.fill_(value)
    
    def plot(self,f=None):
            print("not implemented yet")
            return
       
           

class sampler():
    def __init__(self, inputs,
                       dim = 2,
                       pdf_img=None,
                       debug=False,
                       background_weight = 1,
                       ignore_labels_in_pdf=[],
                       element_size = (1.0,1.0,1.0),
                       set_background_label = -1.0,
                       normalization = "none",
                       verbose=False,
                       info=None,
                       keep_pdf=False,
                       compute_bb=False):
        
        self.info = info
        self.n_labels = 1 if len(inputs[1].shape) == dim else inputs[1].shape[0]
        self.n_modalities = 1 if len(inputs[0].shape) == dim else inputs[0].shape[0]
            
        if verbose:
            print("labels : {}".format(self.n_labels))
            print("modalities : {}".format(self.n_modalities))

        #t_img = torch.from_numpy(inputs[0]).float()
        t_img = tls.tt(inputs[0],dtype=torch.float32,requires_grad=False)
       # print("t_img.requires_grad {}".format(t_img.requires_grad))

        while len(t_img.shape) < dim+2: 
            t_img.unsqueeze_(0)
        #t_label = torch.from_numpy(inputs[1]).float()
        
        
        t_label = tls.tt(
                         inputs[1]
                         ,dtype=torch.float32,requires_grad=False)
        
        
        
        #print(t_label.shape)
        #print(t_label.amax(dim=(1,2)))
        
       # print("t_label.requires_grad {}".format(t_label.requires_grad))
        
        while len(t_label.shape) < dim+2: 
            t_label.unsqueeze_(0)
    
        self.pw_img = pw_img_data(t_img, 
                                 element_size=element_size[:dim],
                                   normalization=normalization,compute_bb=compute_bb)
    
        #if set_background_label is not None:
        #    t_label[t_label<0.0000000001] = set_background_label
            
        ibb = self.pw_img.bb 
        mask = torch.zeros(self.pw_img.img.shape[2:],requires_grad=False)
        if dim == 2:
            mask[ibb[0][0]:ibb[1][0],ibb[0][1]:ibb[1][1]] = 1
        else:
            mask[ibb[0][0]:ibb[1][0],ibb[0][1]:ibb[1][1],ibb[0][2]:ibb[1][2]] = 1
            
            
        l_shape = list(t_label.shape)
        l_shape[1] = 1
        t_label = torch.cat(
            (t_label,torch.ones(l_shape)),
            dim=1
            )
        
        
        t_label *= mask[None,None,...]    
        #if dim == 2:
        #    t_label[ibb[0][0]:ibb[0][1],ibb[1][0]:ibb[1][1]] = torch.where(t_label[ibb[0][0]:ibb[0][1],ibb[1][0]:ibb[1][1]]<0.0000000001,
        #                                                                   set_background_label,
        #                                                                   t_label[ibb[0][0]:ibb[0][1],ibb[1][0]:ibb[1][1]])
        #print(t_label.shape)
        #print(t_label.amax(dim=(0,2,3)))
        self.pw_label = pw_img_data(t_label,element_size=element_size[:dim],compute_bb=compute_bb)
        #print(self.pw_label.img.amax(dim=(0,2,3)))
        
        #print("wtf")
        self.weights = {}
        if pdf_img is None:
            #n_pos = np.sum(inputs[1]>0)
            #n_neg = np.sum(inputs[1]<1)
            #print("creating pdf from labels with bg weight {}".format(background_weight))
            pdf_img = inputs[1]
           
            
            if self.n_labels>1:
                    pdf_img_ = np.max(pdf_img,axis=0)
                    n_pos = np.sum(pdf_img_)
                    n_pos_ = np.sum(pdf_img)
                    pdf_img_[:] = 0
                    self.weights["classes"] = []
                    for l in range(self.n_labels):
                        if l not in ignore_labels_in_pdf:
                            label_i_img = (pdf_img[l,...]>0)
                            w = (n_pos_+0.00001)/(np.sum(label_i_img)+0.00001)
                            self.weights["classes"] += [w]
                            #print("label {}: w {}".format(l,w))
                            #print("{}:".format(label_i_img.shape))
                            #print("{}:".format(pdf_img_.shape))
                            pdf_img_ += label_i_img * w 
                        else:
                            print("excluding label {} from sampling".format(l))
                    pdf_img = (pdf_img_ * (n_pos+0.0000001) / (np.sum(pdf_img_)+0.0000001)).astype(np.float32)
                    
            else:
                assert(len(pdf_img.shape)==dim+1)
                assert(pdf_img.shape[0]==1)
                
                pdf_img = (pdf_img[0,...]  > 0).astype(np.float32)
                #self.label_weights += w
                self.weights["classes"] = 1
                
            n_pos = np.sum(pdf_img)
           # print("n_pos {}: ".format(n_pos))

            n_neg = pdf_img.size - n_pos
            background_w = background_weight*(n_pos+1)/(n_neg+1)
            pdf_img[pdf_img==0] = background_weight*(n_pos+1)/(n_neg+1)
            self.weights["bg"] = background_w
            #pdf_img[pdf_img<1] = background_weight*(n_pos+1)/(n_neg+1)
            self.pdf_img = pdf_img
            
          #  print("shape {}".format(pdf_img.shape))
            self.shape =np.array( pdf_img.shape,dtype=np.int32)
            #n_pos = np.sum(pdf_img)
            self.pdf = tls.pdf_2_cumsum(pdf_img.reshape((-1,)))
            
            
            

            #bg_weight = (n_pos+0.0001)/(n_neg+0.0001)
            #self.pdf = tls.pdf_2_cumsum(pdf_img.reshape((-1,))+bg_weight)
        else:
            if pdf_img == "no_pdf":
               # print("no pdf will be generated")
                self.shape  = np.array(inputs[0].shape[1:],dtype=np.int32)
            else:
               # print("pdf function is given")
                self.shape = np.array(pdf_img.shape,dtype=np.int32)
                self.pdf = tls.pdf_2_cumsum(pdf_img.reshape((-1,)))
            
        if keep_pdf:
            self.pdf_img = pdf_img
        
        self.dim = dim
        self.debug = debug
        if debug:
            self.sampling_map = torch.zeros((1,1,)+tuple(pdf_img.shape,),requires_grad=False)
           # self.scatterer = tls.scatterer()
        
        #self.pdf = self.pdf_2_cumsum(inputs[0])
        #self.pdf = self.pdf_2_cumsum(inputs[1])
    
        
    #def update_pdf(self):
        
    
    def sample_mu(self,
                batch_batch_img,
                batch_batch_labels,
                batch_id,
                num_samples,
                scale_facts,
                pfield=(32,),
                base_scale_indx = 0,
                target_element_size_mu = None,
                interpolation=["nearest","nearest"],
                target_device="cpu",
                aug_device="cpu",
                augmentation = {"aug_rotations":None,"aug_scales":None,"aug_flipdims":None,"aug_scales2":None},
                sampling="pdf_fine2coarse",
                sample_from_label = -1,
                weights = {'bg':1.0,'classes':[1]},
                snap="valid",
                padding_mode_img=pw_settings.default_img_padding_mode,
                padding_mode_label="zeros",
                warn_if_ooim=False,
                copy_metadata=False,
                debug_scales = -1,
                max_depth = -1,
                ):

        max_patches = batch_batch_img.patchlist[0].tensor.shape[0]
        assert(batch_id+num_samples <= max_patches)        
        
        #n_scales = scale_facts.shape[1]
        pfield_size_vox = torch.tensor(pfield,dtype=torch.float32)
        bb = self.pw_img.bb_mu()
        
        img_element_size_mu = tls.tt(self.pw_img.element_size)     
        
        center_normalization = False
        bb_0 = bb[0][None,:]
        bb_1 = bb[1][None,:]
       # bb_1_max = bb_1.clone()
        
       # print(pfield_size_vox)
       # print(target_element_size_mu)
        
        #pfield_size_mu_target = pfield_size_vox * tls.tt(target_element_size_mu,dtype=torch.float32)
        pfield_size_mu_target = tls.vox_shape2physical_shape(pfield_size_vox,target_element_size_mu,dtype=torch.float32,device=pfield_size_vox.device)
        
        fov = bb_1 - pfield_size_mu_target - bb_0           
        patching_strategy = "n_coarse2fine"
            
        #if sampling == "uniform_coarse2fine":
        #            position_mu_target = torch.rand(num_samples,self.dim) * fov + bb_0
                    #assert(torch.all(position_mu_target>=bb_0))
                    #assert(torch.all(position_mu_target<=bb_1-pfield_size_mu_target))
        
                    
        if "pdf" in sampling:
            position_vox = torch.zeros(num_samples,self.dim,requires_grad=False)
            for r in range(num_samples):
                indx = tls.rand_pic_array(self.pdf)
                position_vox[r,:] = torch.tensor(tls.ind2sub(self.shape, indx),dtype=torch.float32)

                #if not self.pdf[indx]>0:
                #    print("score: ",self.pdf[indx])
              #  position_vox[r,0] = 300
              #  position_vox[r,1] = 300
                
                #position_vox[r,2] = 81
                
                #if self.debug:
                #    pos_top_left_int = position_vox[r,:].floor().to(torch.int32)
                #    if self.dim == 2:
                #        self.sampling_map[pos_top_left_int[0],pos_top_left_int[1]] += 1
                #    else:
                #        self.sampling_map[pos_top_left_int[0]:pos_top_left_int[0]+int(pfield_size_vox[0]),
                #                          pos_top_left_int[1]:pos_top_left_int[1]+int(pfield_size_vox[1]),
                #                          pos_top_left_int[2]:pos_top_left_int[2]+int(pfield_size_vox[2])] += 1
                
                
            #position_mu_target = position_vox * img_element_size_mu
            position_mu_target = tls.vox_pos2physical_pos(position_vox,img_element_size_mu,dtype=position_vox.dtype,device=position_vox.device)
            pos_noise = torch.rand(position_mu_target.shape) * pfield_size_mu_target
            #pos_noise = 0
            
            if "coarse2fine" in sampling:
                patching_strategy = "position_coarse2fine"    
            else:
                patching_strategy = "position_fine2coarse"    
                position_mu_target -= pos_noise
                position_mu_target = torch.maximum(bb_0,position_mu_target)
                position_mu_target = torch.minimum(bb_1-pfield_size_mu_target,position_mu_target)
                assert(sample_from_label==-1)
       
                assert(torch.all(position_mu_target>=bb_0))
                assert(torch.all(position_mu_target<=bb_1-pfield_size_mu_target))
                center_normalization = True
            
            if False:
                print("####################")
                print("DEBUG REMOVE ME")
                print("####################")
                position_vox[:,0] = self.pw_img.get_shape()[0]/2
                position_vox[:,1] = self.pw_img.get_shape()[1]/2
                position_mu_target = tls.vox_pos2physical_pos(position_vox,img_element_size_mu,dtype=position_vox.dtype,device=position_vox.device)
                center_normalization = False 
        
        
        if sampling == "n_coarse2fine":
            #assert(sample_from_label>-1)
            position_mu_target = num_samples
            
        if sampling == "n_coarse2fine_valid":
            patching_strategy = "n_coarse2fine_valid"
            #assert(sample_from_label>-1)
            position_mu_target = num_samples            
            
        if sampling == "n_coarse2fine_all":
            patching_strategy = "n_coarse2fine_all"
            position_mu_target = num_samples            
        #print(position_mu_target.shape)
       # print(bb_0)
        
        
        
            
        pw_debug("calling patch_all")
        patchwriter.patch_all(
                    pw_img_list=[self.pw_img,self.pw_label],
                    batch_list_list=[batch_batch_img,batch_batch_labels],
                    interpolation_list=interpolation,
                    batch_id=batch_id,
                    position_mu_target=position_mu_target,
                    scale_facts=scale_facts,
                    patching_strategy=patching_strategy,
                    base_scale_indx=base_scale_indx,
                    sample_from_label=sample_from_label,
                    #augmentation = augmentations,
                    augmentation = augmentation,
                    aug_device=aug_device,
                    target_device=target_device,
                    target_element_size_mu=target_element_size_mu,
                    warn_if_ooim=warn_if_ooim,
                    crop_bb = True,
                    copy_metadata=copy_metadata,
                    weights=weights,
                    center_normalization=center_normalization,
                    snap=snap,
                    padding_mode_img=padding_mode_img,
                    padding_mode_label=padding_mode_label,
                    max_depth=max_depth,
                )                      
        pw_debug("called patch_all")            
        
        if self.debug:
            #scale = 0
            #if True:
            for scale in range(len( batch_batch_img.patchlist)):
                if scale == debug_scales or debug_scales == -1:
                    scatterer = tls.scatterer()
                    #grid = patch.patchlist[base_scale_indx].rel_coordinates
                    
                    grid = batch_batch_img.patchlist[scale].rel_coordinates[batch_id:batch_id+num_samples,...]
                    patch = batch_batch_img.patchlist[0].tensor[batch_id:batch_id+num_samples,...]
                    #tmp = 
                    #print("scattering {} patches ".format(len(positions_mu_c)))    
                    n_labels = patch.shape[1]
                    shape = self.sampling_map.shape
                    result = torch.zeros((1,n_labels,)+tuple(shape)[2:],
                                     device="cpu", 
                                     requires_grad=False)
                
        
                
                    
                    #result_counts = torch.zeros((1,1,)+tuple(shape_out_vox),
                    #                 device=target_device, 
                    #                 requires_grad=False)
        
        
                    scatterer.scatter_2_img_add(
                               grid,
                               patch,
                               result,
                               self.sampling_map,        
                               batch_size=num_samples)
        #if self.debug:
        #batch_batch_img.copy_to_patch_item( 
        #             self.img_pyramid,
        #            np.arange(num_samples)+batch_id,
        #            interpolation="nearest")    

        #batch_batch_labels.copy_to_patch_item( 
        #             self.label_pyramid,
        #            np.arange(num_samples)+batch_id,
        #            interpolation="nearest")      
    
    
    
if False:    
#%%
    #ref_scale = 
    sampling = "uniform_coarse2fine"
    sampling = "pdf_fine2coarse"
    dataset.data_sampler[0].sampling_map[:] = 0
    
    dataset.resample(aug_device="cuda",sampling=sampling,target_element_size=[0.25,0.25],patchbuffersize=3,min_patches_per_dataset=1)    
    

    #batch = dataset.batch(200,device="cuda")
    #batch = dataset.batch(200,device="cuda")
    #batch = dataset.batch(2,device="cuda")
    
    

        


