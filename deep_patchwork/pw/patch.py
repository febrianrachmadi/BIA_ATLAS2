#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:28:25 2020

@author: skibbe
"""

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
import scipy
import pw
from typing import Dict, Tuple, Sequence, List
#from torch_sampling import choice
from . import batch_mat as bmat
from . import units
#from . import g_kernel
from .tools import pw_debug
import copy
from .settings import pw_settings 




class patchwriter():
    
    #default_img_padding_mode = "border"
    
    
    
    @staticmethod
    def get_patch_mu(
                    pw_img,
                    batch_item_next,
                    batch_id,
                    position_mu,
                    scale_fact,
                    batch_id_previous=None,
                    batch_item_previous=None,
                    augmentation = {},
                    interpolation=None,#"nearest",#interpolation=["bilinear","nearest"],
                    aug_device="cpu",
                    target_device="cpu",
                    target_element_size=None,
                    crop_bb = False,
                    warn_if_ooim=False,
                    ):
   
        
       # position_mu[:,2] = 0.5
        #aug_state = batch_item_previous.aug_state if hasattr(batch_item_previous, 'aug_state') else {}
            
        position_mu = tls.tt(position_mu,device=aug_device)
        
        num_batches = position_mu.shape[0]
        
        
        if batch_id_previous is None:
            batch_id_previous = batch_id
        else:
            print("WARNING: overriding previous batch id")
            
        pfield = batch_item_next.tensor.shape[2:]
        #  img_pyramid = img_pyramid_list[0]
        
        if target_element_size is None:
            print("WARNING: target voxel resolution not set, using image voxel resolution")
        img_element_size_mu = tls.tt(pw_img.element_size,device=aug_device)                
        
        target_element_size_mu = tls.tt(target_element_size,device=aug_device) if target_element_size is not None else tls.tt(img_element_size_mu,device=aug_device)
        
        scale_fact_t  = tls.tt(scale_fact,device=aug_device)
        
        pfield_size_vox = tls.tt(pfield,device=aug_device)
       

        
        #pfield_of_view_size = tls.vox_shape2physical_shape(pfield_size_vox,target_element_size_mu*scale_fact_t,dtype=target_element_size_mu.dtype,device=aug_device)
        pfield_of_view_size = tls.vox_shape2physical_shape(pfield_size_vox,target_element_size_mu*scale_fact_t,dtype=torch.float32,device=aug_device)
        #pfield_of_view_size = torch.clamp(pfield_of_view_size,min=0.0001)
        
        if pw_img.dim==2:                   
            mat_inv_scale_fact = torch.tensor([[scale_fact_t[0],0,0],
                                              [0,scale_fact_t[1],0],
                                              [0,0,1]],dtype = torch.float32,device=aug_device)
            mat_scale_fact = torch.tensor([[1.0/(scale_fact_t[0]),0,0],
                                      [0,1.0/(scale_fact_t[1]),0],
                                              [0,0,1]],dtype = torch.float32,device=aug_device)     
        else:
            mat_inv_scale_fact = torch.tensor([[scale_fact_t[0],0,0,0],
                                              [0,scale_fact_t[1],0,0],
                                              [0,0,scale_fact_t[2],0],
                                              [0,0,0,1]],dtype = torch.float32,device=aug_device)
            mat_scale_fact = torch.tensor([[1.0/(scale_fact_t[0]),0,0,0],
                                      [0,1.0/(scale_fact_t[1]),0,0],
                                      [0,0,1.0/(scale_fact_t[2]),0],
                                              [0,0,0,1]],dtype = torch.float32,device=aug_device)  
        
        shape_mu = pw_img.shape_mu()
        shape_mu[shape_mu==0] = 1
       
        bb_start = position_mu
        bb_end = bb_start + pfield_of_view_size
      
        
        pfield_not_flat = pfield_of_view_size>0
        #assert(pfield_not_flat.all())
        
            
        bb_center = (bb_start + bb_end) / 2.0
      
        
        grid = torch.ones((num_batches,)+pfield+(pw_img.dim+1,),dtype = torch.float32,device=aug_device)
        
        if pw_img.dim == 2:                      
            #print(pfield_of_view_size)
            dx = tls.arange_closed(0,pfield_of_view_size[0],pfield_size_vox[0],device=aug_device)
            dy = tls.arange_closed(0,pfield_of_view_size[1],pfield_size_vox[1],device=aug_device)
            
            #grid_y , grid_x = torch.meshgrid(dx, dy)
            grid_x , grid_y = torch.meshgrid(dx, dy, indexing="ij")
            
            grid[:,...,0] = grid_x
            grid[:,...,1] = grid_y

            ########################
            #
            #   physical space to relative grid space
            #   --> [to use pytorch grid interpolator]
            #   --> xz will be swapped
            #
            ########################
            mu2rel_scale = torch.tensor([[1.0/shape_mu[0],0,0],
                                      [0,1.0/shape_mu[1],0],
                                      [-0.5,-0.5,1]],dtype = torch.float32,device=aug_device)
            mu2rel_shift = torch.tensor([[0,2.0,0],
                                      [2.0,0,0],
                                      [0,0,1]],dtype = torch.float32,device=aug_device)

            mu2rel = torch.mm(mu2rel_scale,mu2rel_shift)

        if pw_img.dim == 3:       
            dx = tls.arange_closed(0,pfield_of_view_size[0],pfield_size_vox[0],device=aug_device)
            dy = tls.arange_closed(0,pfield_of_view_size[1],pfield_size_vox[1],device=aug_device)
            dz = tls.arange_closed(0,pfield_of_view_size[2],pfield_size_vox[2],device=aug_device)

            #grid_z, grid_y , grid_x = torch.meshgrid(dx, dy, dz)
            grid_x, grid_y , grid_z = torch.meshgrid(dx, dy, dz, indexing="ij")
            
            grid[:,...,0] = grid_x
            grid[:,...,1] = grid_y
            grid[:,...,2] = grid_z
            
            ########################
            #
            #   physical space to relative grid space
            #   --> [to use pytorch grid interpolator]
            #   --> xz will be swapped
            #
            ########################
            
            mu2rel_scale = torch.tensor([[1.0/shape_mu[0],0,0,0],
                                      [0,1.0/shape_mu[1],0,0],
                                      [0,0,1.0/shape_mu[2],0],
                                      [-0.5,-0.5,-0.5,1]],dtype = torch.float32,device=aug_device)

            mu2rel_shift = torch.tensor([[0,0,2.0,0],
                                      [0,2.0,0,0],
                                      [2.0,0,0,0],
                                      [0,0,0,1]],dtype = torch.float32,device=aug_device)
            
            mu2rel = torch.mm(mu2rel_scale,mu2rel_shift)

  
        bm = bmat.batch_mat(pw_img.dim,dtype = torch.float32,device=aug_device)
        
        
        trans_inv_mat_rot = bm.transl_mat(-bb_center)
        trans_mat_rot = bm.transl_mat(bb_center)
        
        bb_shift = bm.transl_mat(bb_start)
        
        
        ########################
        #
        #   Augmentation matrix
        #
        ########################
        
        
        aug_mat,aug_state = tls.draw_augmentation_mat(augmentation=augmentation,
                                            n_samples=num_batches,
                                            dim=pw_img.dim,
                                            device=aug_device,
                                            aug_state=batch_item_previous.aug_state if batch_item_previous is not None else {})
        
        #print(aug_mat[0,...])          
        batch_item_next.aug_state = copy.deepcopy(aug_state)
        
        if True:
            aug_mat = torch.matmul(mat_scale_fact,aug_mat)
            aug_mat = torch.matmul(aug_mat,mat_inv_scale_fact)
        #aug_mat = torch.matmul(mat_inv_target_size,aug_mat)
        
        
        #if pdebug:
        #    print("aug scale+rot+scale2 matrix \n {}".format(aug_mat[:min(aug_mat.shape[0],npdebug),:]))
            
        ########################
        #
        #   In order to not accumulate augmentation matrices,
        #   we "undo previous rotations and scales for the patch"
        #
        ########################    
        if batch_item_previous is not None: 
            aug_prev = torch.inverse(batch_item_previous.global_rot_mat_s_local[batch_id_previous:batch_id_previous+num_batches,...]).to(device=aug_device)
            aug_mat = torch.matmul(aug_mat,aug_prev) 
        
        local_rot_mat_s = aug_mat                
        global_rot_mat_s_local = aug_mat    
        
        mat = torch.matmul(trans_inv_mat_rot,aug_mat)
        
        global_rot_mat_s = torch.matmul(mat,trans_mat_rot)
   
        if batch_item_previous is not None:                            
           global_rot_mat_s = torch.matmul(global_rot_mat_s,batch_item_previous.global_rot_mat_s[batch_id_previous:batch_id_previous+num_batches,...].to(device=aug_device))              
           global_rot_mat_s_local = torch.matmul(local_rot_mat_s,batch_item_previous.global_rot_mat_s_local[batch_id_previous:batch_id_previous+num_batches,...].to(device=aug_device))                   
        
        
        
        
        
        mat = torch.matmul(bb_shift,global_rot_mat_s)
        
        #mat = torch.matmul(mat_inv_target_size,mat)
        
        mat = torch.matmul(mat,mu2rel)
        
        batch_item_next.local_rot_mat_s[batch_id:batch_id+num_batches,...] = local_rot_mat_s
        batch_item_next.global_rot_mat_s[batch_id:batch_id+num_batches,...] = global_rot_mat_s
        batch_item_next.global_rot_mat_s_local[batch_id:batch_id+num_batches,...] = global_rot_mat_s_local
        
        #grid = torch.einsum('b...xy,bxd->b...xy',grid ,mat)
        grid = torch.einsum('b...x,bxy->b...y',grid ,mat)


                
        grid_0 = bb_start
        grid_1 = bb_end
        
        crop_offsets_0 = grid_0.clone().to(device="cpu")
        crop_offsets_1 = grid_1.clone().to(device="cpu")
        
        t_s = pw_img.img
        
        grid_cpu = grid.to(device=t_s.device)
        in_shape = (1,)+(grid_cpu.shape[0]*grid_cpu.shape[1],) + tuple(grid_cpu.shape[2:-1])+(pw_img.dim,)
        out_shape = (num_batches,)+ (t_s.shape[1],)+tuple(grid_cpu.shape[1:-1])
        
        if interpolation is not None:
            
            patch = torch.nn.functional.grid_sample(t_s, 
                                        grid_cpu[...,:pw_img.dim].reshape(in_shape), 
                                        mode=interpolation, 
                                        padding_mode='zeros', 
                                        align_corners=True).reshape(out_shape).to(device=target_device)
            
            
            batch_item_next.tensor[batch_id:batch_id+num_batches,...] = patch
            
        batch_item_next.rel_coordinates[batch_id:batch_id+num_batches,:] = grid_cpu[...,:pw_img.dim]
        
        
        if "wobble" in augmentation and augmentation["wobble"] is not None:
            
            
            #print("0")
            wpa = augmentation["wobble"]
            if hasattr(batch_item_previous, 'wobble'):
                wobble = batch_item_previous.wobble
            else:
                if False:
                    d_shape = tls.tt(wpa["gridsize"],device=target_device) if "gridsize"  in wpa else tls.tt((4,)*pw_img.dim,device=target_device) 
                                    
                    wobble = 1-2*torch.rand((num_batches,pw_img.dim)+tuple(d_shape.tolist()),device=target_device)
                    wfact = tls.tt(wpa["amount"],device=target_device).flip(dims=(0,)) if "amount"  in wpa else 0.25
                    wfact = wfact*(1/d_shape)
                    wobble *= wfact[None,:,None,None,None] if pw_img.dim == 3 else wfact[None,:,None,None]            
                    wobble *= torch.rand((wobble.shape[0],)+(1,)*(pw_img.dim+1),device=target_device)

                i_shape = torch.tensor(pw_img.get_shape())
                w_stages = wpa["gridsize"]
                w_amount = wpa["amount"]
                
                waxis = torch.tensor((True,) *pw_img.dim if "axis" not in wpa else wpa["axis"])
                wobble = None
                for ws,wa in zip(w_stages,w_amount):
                    d_shape = torch.ceil(i_shape*ws).to(torch.int32)
                    d_shape = tls.tt(d_shape,device=target_device)
                    
                    wobble_ = 1-2*torch.rand((num_batches,pw_img.dim)+tuple(d_shape.tolist()),device=target_device)
                    #wfact = tls.tt(wpa["amount"],device=target_device).flip(dims=(0,)) if "amount"  in wpa else 0.25
                    #wfact = tls.tt(wa/len(w_amount),device=target_device)
                    wfact = tls.tt(wa,device=target_device)
                    wfact = wfact*(1/d_shape)
                    wobble_ *= wfact[None,:,None,None,None] if pw_img.dim == 3 else wfact[None,:,None,None]            
                    #wobble_ *= wpa["prob"][0]+(wpa["prob"][1]-wpa["prob"][0])*torch.rand((wobble_.shape[0],)+(1,)*(pw_img.dim+1),device=target_device)
                    wobble_ *= wpa["prob"][0]+(wpa["prob"][1]-wpa["prob"][0])*torch.rand((wobble_.shape[0],)+(1,)*(pw_img.dim+1),device=target_device)
                    
                    if wobble is None:
                        wobble = wobble_
                    else:
                        wobble = wobble_ + torch.nn.functional.interpolate(
                            wobble,
                            wobble_.shape[2:],
                            mode='bilinear' if pw_img.dim == 2 else "trilinear")
                if (waxis==False).any():  
                    wobble[:,waxis==False,...] = 0
                    
                
                    #print("")
               # gkernel = tls.gauss((3,)*pw_img.dim,
               #                     sigma=(1.5,)*pw_img.dim,norm="sum")
               # gkernel = gkernel[None,None,...].expand((pw_img.dim,1,-1,-1)).clone()  
                #wobble = units.layersND.FConv(pw_img.dim,wobble,gkernel,groups=pw_img.dim,padding="same")
                
            batch_item_next.wobble = wobble.clone()
            #print("8")
            wobble_g = torch.nn.functional.grid_sample(
                                    wobble, 
                                    batch_item_next.rel_coordinates[batch_id:batch_id+num_batches,:].to(wobble.device), 
                                    mode="bilinear", 
                                    padding_mode="border", 
                                    align_corners=True)
            #print("9")            
            if pw_img.dim == 3:
                wobble_g = wobble_g.permute([0,2,3,4,1])
            else:
                wobble_g = wobble_g.permute([0,2,3,1])
                
            batch_item_next.rel_coordinates[batch_id:batch_id+num_batches,:] += wobble_g.to(batch_item_next.rel_coordinates.device)#.to(device=batch_item_next.rel_coordinate.device)
            
        if False:
            if "blobs_add" in augmentation and augmentation["blob_add"] is not None:
                
                
                #print("0")
                wpa = augmentation["blob_add"]
                if hasattr(batch_item_previous, 'blob_add'):
                    blob_add = batch_item_previous.blob_add
                else:
    
                    i_shape = torch.tensor(pw_img.get_shape())
                    w_stages = wpa["gridsize"]
                    w_amount = wpa["amount"]
                    
                    blob_add = None
                    for ws,wa in zip(w_stages,w_amount):
                        d_shape = torch.ceil(i_shape*ws).to(torch.int32)
                        d_shape = tls.tt(d_shape,device=target_device)
                        
                        wfact = tls.tt(wa,device=target_device)
                        blob_add_ = torch.rand((num_batches,pw_img.dim)+tuple(d_shape.tolist()),device=target_device)
                        blob_add_ *= (blob_add_>wfact).to(blob_add_.dtype)
                        
                        if wobble is None:
                            blob_add = blob_add_
                        else:
                            blob_add = blob_add_ + torch.nn.functional.interpolate(
                                blob_add,
                                blob_add_.shape[2:],
                                mode='bilinear' if pw_img.dim == 2 else "trilinear")
                    
                    blob_add /= len(w_stages)
                    
                batch_item_next.blob_add = blob_add.clone()
                #print("8")
                if False:
                    blob_add_g = torch.nn.functional.grid_sample(
                                            blob_add, 
                                            batch_item_next.rel_coordinates[batch_id:batch_id+num_batches,:].to(blob_add.device), 
                                            mode="bilinear", 
                                            padding_mode="border", 
                                            align_corners=True)
                
        
        batch_item_next.offset_rel_0[batch_id:batch_id+num_batches,:] =  crop_offsets_0
        batch_item_next.offset_rel_1[batch_id:batch_id+num_batches,:] =  crop_offsets_1
        
        ########################
        #
        #   If current patch is NOT the first (coarsest) one,
        #   we need the mapping from the previous (corser) level to the current 
        #   (fineer, more detailed) level
        #
        ########################    
  
        if batch_item_previous is not None:

            if False:
                if pw_img.dim==2:                   
                    trans_inv_mat_rot_target = torch.tensor([[target_element_size_mu[0],0,0],
                                                      [0,target_element_size_mu[1],0],
                                                      [0,0,1]],dtype = torch.float32,device=aug_device)
                    trans_mat_rot_target = torch.tensor([[1.0/(target_element_size_mu[0]),0,0],
                                              [0,1.0/(target_element_size_mu[1]),0],
                                                      [0,0,1]],dtype = torch.float32,device=aug_device)      
    
                else:
                    trans_inv_mat_rot_target = torch.tensor([[target_element_size_mu[0],0,0,0],
                                                      [0,target_element_size_mu[1],0,0],
                                                      [0,0,target_element_size_mu[2],0],
                                                      [0,0,0,1]],dtype = torch.float32,device=aug_device)
                    trans_mat_rot_target = torch.tensor([[1.0/(target_element_size_mu[0]),0,0,0],
                                              [0,1.0/(target_element_size_mu[1]),0,0],
                                              [0,0,1.0/(target_element_size_mu[2]),0],
                                                      [0,0,0,1]],dtype = torch.float32,device=aug_device)  
            
            patch_2_patch_grid = torch.ones((num_batches,)+pfield+(pw_img.dim+1,),dtype = torch.float32,device=aug_device)
                            
            crop_offsets_1_previous = batch_item_previous.offset_rel_1[batch_id_previous:batch_id_previous+num_batches,:]
            crop_offsets_0_previous = batch_item_previous.offset_rel_0[batch_id_previous:batch_id_previous+num_batches,:]
            
            # extents of the previous (coarser) patch in physical coordinates
            src_size = (crop_offsets_1_previous[0,:] - crop_offsets_0_previous[0,:])
            # bounding box in physical coordinates of the current patch
            # given in relative coordinates to the previous patch
            # (in original xyz aligned image space)
            start_ = crop_offsets_0 - crop_offsets_0_previous
            end_ = crop_offsets_1 - crop_offsets_0_previous 
            # extents of the current patch in physical coordinates          
            target_size = end_[0,:] - start_[0,:]
            
            
            if pw_img.dim == 3:             
                gdx = tls.arange_closed(0,target_size[0],pfield[0])
                gdy = tls.arange_closed(0,target_size[1],pfield[1])
                gdz = tls.arange_closed(0,target_size[2],pfield[2])
                
                patch_2_patch_grid[...,0],patch_2_patch_grid[...,1],patch_2_patch_grid[...,2] = torch.meshgrid(gdx, gdy, gdz, indexing="ij")
             
                for b in range(num_batches):
                    patch_grid_shift = torch.tensor([[1.0,0,0,0],
                                          [0,1.0,0,0],
                                          [0,0,1.0,0],
                                          [start_[b,0],start_[b,1],start_[b,2],1]],dtype = torch.float32,device=aug_device)
                 
                    patch_2_patch_grid[b,...] = torch.matmul(patch_2_patch_grid[b,...] ,patch_grid_shift)


            else:
                gdx = tls.arange_closed(0,target_size[0],pfield[0])
                gdy = tls.arange_closed(0,target_size[1],pfield[1])
                patch_2_patch_grid[...,0],patch_2_patch_grid[...,1] = torch.meshgrid(gdx, gdy, indexing="ij")
                
                for b in range(num_batches):
                    patch_grid_shift = torch.tensor([[1.0,0,0],
                                          [0,1.0,0],
                                          [start_[b,0],start_[b,1],1]],dtype = torch.float32,device=aug_device)
                 
                    patch_2_patch_grid[b,...] = torch.matmul(patch_2_patch_grid[b,...] ,patch_grid_shift)
            
  
            center__  = ((start_+end_)/2.0).to(aug_device)
            trans_inv_mat_rot = bm.transl_mat(-center__)
            trans_mat_rot = bm.transl_mat(center__)
            
           # pratio_inv = bm.batch_diag(1.0/pfield_size_vox[None,...].type(torch.float32))
           # pratio = bm.batch_diag(pfield_size_vox[None,...].type(torch.float32))
            
            ########################
            #
            #   physical space to relative grid space
            #   --> [to use pytorch grid interpolator]
            #   --> xz will be swapped
            #
            ########################            
            ts = src_size
            
            # where patch is flat set to 1
            ts[ts==0] = 1
                   
            if pw_img.dim==3:   
                mu2rel_scale = torch.tensor([[1.0/ts[0],0,0,0],
                              [0,1.0/ts[1],0,0],
                              [0,0,1.0/ts[2],0],
                              [-0.5,-0.5,-0.5,1]],dtype = torch.float32,device=aug_device)
                mu2rel_shift = torch.tensor([[0.0,0,2.0,0],
                            [0,2.0,0,0],
                            [2.0,0,0,0],
                            [0,0,0,1]],dtype = torch.float32,device=aug_device)
            else:
                mu2rel_scale = torch.tensor([[1.0/ts[0],0,0],
                              [0,1.0/ts[1],0],
                              [-0.5,-0.5,1]],dtype = torch.float32,device=aug_device)
                mu2rel_shift = torch.tensor([[0.0,2.0,0],
                            [2.0,0,0],
                            [0,0,1]],dtype = torch.float32,device=aug_device)   
                
            mu2rel = torch.mm(mu2rel_scale,mu2rel_shift)

            ########################
            #
            #   computing & applying the transformation
            #
            ########################            
            mat = trans_inv_mat_rot
            T_current = batch_item_next.global_rot_mat_s_local[batch_id:batch_id+num_batches,...].to(device=aug_device)
            T_previous = batch_item_previous.global_rot_mat_s_local[batch_id:batch_id+num_batches,...].to(device=aug_device)
            
            R0 = torch.matmul(T_current,torch.inverse(T_previous))
            
            #mat = torch.matmul(mat,trans_inv_mat_rot_target)
            
            mat = torch.matmul(mat,R0)
            
            #mat = torch.matmul(mat,trans_mat_rot_target)
            
            mat = torch.matmul(mat,trans_mat_rot)
            mat = torch.matmul(mat,mu2rel)  
            patch_2_patch_grid = torch.einsum('b...x,bxy->b...y',patch_2_patch_grid,mat)
            
            
            
            
            batch_item_next.grid[batch_id:batch_id+num_batches,...,0] =  patch_2_patch_grid[...,0]
            batch_item_next.grid[batch_id:batch_id+num_batches,...,1] =  patch_2_patch_grid[...,1]#pos_grid_y
            if pw_img.dim == 3:
                batch_item_next.grid[batch_id:batch_id+num_batches,...,2] =  patch_2_patch_grid[...,2]#pos_grid_z
   
            
        

                

    @staticmethod
    def patch_step(
                    pw_img,
                    batch_list,
                    batch_id,
                    positions_mu, #position(s) or number
                    scale_facts,
                    patching_strategy,
                    state=None,
                    base_scale_indx=0,
                    sample_from_patch=None,
                    #augmentation = {"aug_rotations":None,"aug_scales":None,"aug_flipdims":None},
                    augmentation = {},#,{"aug_rotations":None,"aug_scales":None,"aug_scales2":None,"aug_rotations2":None},
                    aug_device="cpu",
                    target_device="cpu",
                    target_element_size_mu=None,
                    warn_if_ooim=False,
                    crop_bb = True,
                    snap="valid",#"allbutfirst",#"all",
                    bg_eps = 0.000001,
                    border_compensation = 0.5,
                    #weights = {'bg':0.0,'classes':100000000,'pos_noise':0.5} 
                    weights = {'bg':1.0,'classes':[1]},
                    #weights = {'bg':1.0,'classes':[10000000000,10000000000]},
                    pos_noise=0.0,
                    sampling="training",
                    saliency_epsilon = 0.01,
                    saliency_reduction = "max",
                    verbosity=0,
                    ):
               #  print("pos_noise  "+str(pos_noise))
              #   print(state)
                 #if sample_from_patch is not None:
                 #    print("sdsd")
               
                 assert(border_compensation<=0.5)
                 #weights = img_pyramid.weights
                 pfield = batch_list.patchlist[0].tensor.shape[2:]
                 #num_coarse2fine_samples = positions_mu if type(positions_mu) == int else 0
                 #assert(sample_from_patch == None or num_coarse2fine_samples==0)
                 bb_0_target = None 
                 if patching_strategy == "position_fine2coarse":
                    if len(positions_mu.shape) == 1:
                         positions_mu = positions_mu[None,:]
                    num_samples = positions_mu.shape[0]
                 elif patching_strategy == "position_coarse2fine":
                    if len(positions_mu.shape) == 1:
                         positions_mu = positions_mu[None,:]
                    num_samples = positions_mu.shape[0]
                 elif patching_strategy == "n_coarse2fine":
                    num_samples =  positions_mu
                 elif patching_strategy == "n_coarse2fine_valid":
                    num_samples =  positions_mu
                 elif patching_strategy == "n_coarse2fine_all":
                    num_samples =  positions_mu
                 elif patching_strategy == "debug":
                    num_samples =  positions_mu
                 
                 #if num_coarse2fine_samples==0:
                 #    if len(positions_mu.shape) == 1:
                 #        positions_mu = positions_mu[None,:]
                 #    num_samples = positions_mu.shape[0]
                 #else:
                 #   bb_0_target = None
                 
                 n_scales = scale_facts.shape[1]
                 pfield_size_vox = torch.tensor(pfield,dtype=torch.float32)
                 bb = pw_img.bb_mu()
                 #img_element_size_mu = img_pyramid.element_size
                 
                 
                 if state is None:
                     i = 0
                     state = {}
                 else:                   
                     i = state["i"] + 1
                     bb_0 = state["bb_0"]
                     bb_1 = state["bb_1"]
                     bb_1_max = state["bb_1_max"]
                     bb_0_min = state["bb_0_min"]
                     
                     bb_0_target = state["bb_0_target"]
                     assert(i<n_scales)
                 ii = n_scales - i - 1
                 scale_fact = scale_facts[:,ii]#1.5**torch.tensor([ii,ii,ii])
                 
                 pfield_size_mu_target = pfield_size_vox * tls.tt(target_element_size_mu,dtype=torch.float32)
                 
                 pfield_size_mu = pfield_size_mu_target *  scale_fact
                 pfield_element_size_mu =  scale_fact * tls.tt(target_element_size_mu,dtype=torch.float32)
        
                 ##############################
                 #      initialize
                 ##############################
                 if i == 0:#base_scale_indx:
                     bb_0 = bb[0][None,:]
                     bb_1 = bb[1][None,:]
                     bb_1_max = bb_1.clone()
                     bb_0_min = bb_0.clone()
                     
             
                     batch_item_previous = None
                     batch_item_next = batch_list.patchlist[ii]
                    
                     if "position" in patching_strategy == "position_fine2coarse":
                     #if num_coarse2fine_samples==0:
                         bb_0_target = positions_mu
             
                 else:
                     batch_item_previous = batch_list.patchlist[ii+1]
                     batch_item_next = batch_list.patchlist[ii]

                 #position_fine:  target position given
            
                 if patching_strategy == "debug":
                         position_center_mu = ( bb_1 + bb_0)# / 2.0
                         position_center_mu[0][0]/= 2.0
                         position_center_mu[0][1]/= 2.0
                         position_center_mu[0][2]/= 2.0                         
                         
                         position_center_mu = position_center_mu+ torch.zeros(num_samples,pw_img.dim) 
                         
                 elif patching_strategy == "position_fine2coarse":
                 #if num_coarse2fine_samples==0:   
                     #MAKE DIFF BASESCALE DEPENDEnt
                     diff = pfield_size_mu - pfield_size_mu_target*scale_facts[:,base_scale_indx]
                     
                     
                     position_mu =  torch.rand(num_samples,pw_img.dim)  * diff + (bb_0_target-diff) 
                     position_center_mu = position_mu + pfield_size_mu / 2.0
                    

                 else:
                    if sample_from_patch is None or i == 0:
                        
                        ##############################
                        # starting position for coarsest scale given
                        ##############################                        
                        if patching_strategy == "position_coarse2fine" and i == 0:
                            
                            position_center_mu = positions_mu
                            
                            if pos_noise> 0:# weights["pos_noise"] > 0:
                               # print("meeep")
                                pos_noise = torch.randn(position_center_mu.shape) * (pfield_size_mu * pos_noise)#weights["pos_noise"])      
                                position_center_mu += pos_noise
                                #print("BLA")
                            #position_mu = position_center_mu - pfield_size_mu / 2.0
                        
                        ##############################                        
                        #otherwise pick uniformly from image 
                        ##############################                        
                        else:
                            fov = bb_1 - bb_0
                            #if True:
                            #border_compensation = 0
                            if border_compensation > 0:
                                #position_center_mu =  torch.rand(num_samples,img_pyramid.dim)  * (fov+pfield_size_mu) + (bb_0 -pfield_size_mu / 2.0)
                                position_center_mu =  torch.rand(num_samples,pw_img.dim)  * (fov+pfield_size_mu*(1 + border_compensation)) + (bb_0 -(1 + border_compensation)*pfield_size_mu / 2.0)
                               # position_center_mu[:,...] = bb_0-pfield_size_mu / 2.0
                                #position_center_mu[:,...] = bb_1+(1 + border_compensation)*pfield_size_mu / 2.0
                                
                                assert(torch.all(position_center_mu>=bb_0-(1 + border_compensation)*pfield_size_mu / 2.0))
                                assert(torch.all(position_center_mu<=bb_1+(1 + border_compensation)*pfield_size_mu / 2.0))
                                
                            else:
                                position_center_mu =  torch.rand(num_samples,pw_img.dim)  * fov + bb_0 
                                
                            
                            
                            #position_center_mu =  bb_1 - torch.rand(num_samples,img_pyramid.dim)  * fov 
                            
                            #position_center_mu = 0 * torch.rand(num_samples,img_pyramid.dim)  * fov + bb_0 
                            #fov = bb_1 - bb_0
                            #position_center_mu =  bb_1 - torch.rand(num_samples,img_pyramid.dim)  * fov 
                            
                            #position_mu = position_center_mu - pfield_size_mu / 2.0
                            
                        if False:
                            #can over/under-flow. to prevent, disable pos_noise
                            if (torch.any(position_center_mu<bb_0) or torch.any(position_center_mu>bb_1)):
                                print("scale: {}".format(ii))
                                print("bb_0 : {}".format(bb_0))
                                print("position_center_mu : {}".format(position_center_mu))
                                print("bb_1 : {}".format(bb_1))
                                print("osition_center_mu<bb_0 : {}/{}".format((position_center_mu<bb_0).sum(),position_center_mu.shape[0]))
                                print("osition_center_mu>bb_1 : {}/{}".format((position_center_mu>bb_1).sum(),position_center_mu.shape[0]))
                            
                        
                        
                    else:
                        
                       # print("sampled from patch {}".format(ii))
                        sampling_type = [0,0]
                        sampled_positions = torch.zeros([num_samples,pw_img.dim],dtype=torch.float32)
                        sample_buffer_zero_stat = 0
                        for batch_indx in range(num_samples):#range(sample_from_patch.shape[0]):
                            #class_weights = tls.tt(weights["classes"])[:,None,None,None]
                            class_weights = tls.tt(weights["classes"])#
                            bg_weight = tls.tt(weights["bg"])#
                            
                            n_voxels = sample_from_patch[0,0,...].numel()#sample_buffer.numel()
                            
                            
                            if "sample_from_class_indx" in weights:
                                class_ind = weights["sample_from_class_indx"]
                               # print("sample_from_class_indx {}".format(class_ind))
                                class_labels = sample_from_patch[batch_id+batch_indx,class_ind,...]
                            else:
                                if sampling == "training":
                                    class_labels = sample_from_patch[batch_id+batch_indx,:-1,...]
                                elif sampling == "inference":
                                    class_labels = sample_from_patch[batch_id+batch_indx,:,...]
                                    #print(class_labels.shape)
                                    #print(class_labels.amin()," ",class_labels.amax())
                                    
                                else:
                                    assert(False)
                                
                            
                           # print(class_labels.amin())
                            
                            if sampling == "training":
                                valid_background = sample_from_patch[batch_id+batch_indx,-1,...] >0.5
                                
                                #if patching_strategy == "n_coarse2fine":
                                if patching_strategy in ["n_coarse2fine","position_coarse2fine"]:
                                    n_labels = (class_labels>0).sum(dim=(1,2) if pw_img.dim==2 else (1,2,3))
                                    
                                    dstat1 = class_labels.amax()
                                    #print("class_labels: ",class_labels.amax())

                                    #n_labels = (sample_from_patch[batch_id+batch_indx,:,...]).sum(dim=(1,2) if img_pyramid.dim==2 else (1,2,3))
                                    #w = (n_pos_+0.00001)/(np.sum(label_i_img)+0.00001)
                                    class_weights = class_weights * (n_voxels+0.00001)/(n_labels+0.00001)
                                    
                                    class_weights = class_weights.view((class_weights.numel(),)+(1,)*pw_img.dim)
                                    sample_buffer = ((class_labels>0)*class_weights).sum(dim=0)
                                    
                                    #print("class_labels 2: ",class_labels.amax())
                                    dstat2 = class_labels.amax()
                                    # I may need to adjust n_voxels to valid_n_voxels?
                                    #sample_buffer_bg = class_labels.amax(dim=0) < 0
                                    #sample_buffer_bg = (class_labels.amax(dim=0) < 0)  
                                    sample_buffer_bg = torch.logical_and(
                                        torch.logical_not(class_labels.amin(dim=0) > 0) ,
                                        valid_background
                                        )
                                    
                                    if True:
                                        sample_buffer[sample_buffer_bg] = 1.0*bg_weight# (bg_weight*n_voxels)/(sample_buffer_bg).sum() 
                                    else: # count only "valid" bg labels
                                        sample_buffer[sample_buffer_bg] = (bg_weight*n_voxels)/(sample_buffer_bg).sum() 

                                    #print("###############")
                                    #print("sample_buffer ",sample_buffer.shape)
                                    #print("sample_buffer_bg ",sample_buffer_bg.shape)
                                    #print("bg_weight ",bg_weight)
                                    #print("bg_weight ",bg_weight)
                                    
                                    #print("class_labels 2: ",class_labels.amax())
                                    dstat3 = class_labels.amax()
                                    if sample_buffer.amax()<0.00000001:
                                        if False:
                                            print("sample_buffer is zero, should not happen, might be due to augmentation; adding constant")
                                            print(dstat1)
                                            print(dstat2)
                                            print(dstat3)
                                        sample_buffer_zero_stat += 1
                                        sample_buffer+=0.0001
                                        
                                                                        
                                if patching_strategy == "n_coarse2fine_valid":
                                    tmp_ = torch.abs(class_labels.amax(dim=0))
                                    sample_buffer =  tmp_>0  
                                    
                                if patching_strategy == "n_coarse2fine_all":
                                    ts = list(class_labels.shape[1:])
                                    sample_buffer = torch.ones(ts)
                                                                    
                                

                            elif sampling == "inference":
                                
                                if patching_strategy == "n_coarse2fine_valid":
                                    tmp_ = torch.abs(class_labels.amax(dim=0))
                                    sample_buffer =  tmp_>0  
                                else:
                                    class_labels_ = (class_labels>1)*(class_labels-1)
                                    n_labels = (class_labels_).sum(dim=(1,2) if pw_img.dim==2 else (1,2,3))
                                    
                                    if n_labels.amax() > saliency_epsilon:                                
                                        sampling_type[0] += 1
                                        class_weights = class_weights.view((class_weights.numel(),)+(1,)*pw_img.dim).to(device=class_labels_.device)            
                                        
                                        if saliency_reduction == "max":
                                            sample_buffer = (class_labels_*class_weights).amax(dim=0)
                                        elif saliency_reduction == "sum":
                                            sample_buffer = (class_labels_*class_weights).sum(dim=0)
                                        else:
                                            print("provide a valid saliency reduction method")
                                            assert(False)
                                        assert(sample_buffer.amax()>0)
                                    else: # uniform (valid voxels)
                                        #out of image is set to 0
                                        sample_buffer = (class_labels.amax(dim=0) > 0).type(class_labels.dtype)
                                        sampling_type[1] += 1
                                    
                            else:
                                assert(False)
                                
 
                            buffer_size = sample_buffer.numel()
                            index = torch.utils.data.WeightedRandomSampler(sample_buffer.view(-1), 1, replacement=True)
                            sampled_positions[batch_indx,:] = tls.unravel_index(tls.tt(list(index)),sample_buffer.shape)
                        
                        #if sample_buffer_zero_stat>0:
                        if sample_buffer_zero_stat>0:
                            #print("")
                            pw_debug("sample_buffer is zero, should not happen, might be due to augmentation, 2D slice from 3D; or sth like that. added a small constant","SAMPLING")  
                            pw_debug("{} cases  out of {} are zero".format(sample_buffer_zero_stat,num_samples),"SAMPLING")  
                            #print()
                            #print(sample_buffer_zero_stat," cases of ",num_samples," are zero")   
                        if verbosity > 0 :
                            print("sampling_type :{}".format(sampling_type))
                        position_center_mu = bb_0 + state["pfield_element_size_mu"] * sampled_positions
                       
                        
                        
                        if False:
                            #can overflow. to prevent, use 
                            #bb_1 = bb_0 + pfield_size_mu
                            #isntead of 
                            #bb_1 = torch.minimum(bb_0 + pfield_size_mu,bb_1_max)
                            if (torch.any(position_center_mu<bb_0) or torch.any(position_center_mu>bb_1)):
                                print("scale: {}".format(ii))
                                print("sampled_positions : {}".format(sampled_positions))
                                print("bb_0 : {}".format(bb_0))
                                print("position_center_mu : {}".format(position_center_mu))
                                print("bb_1 : {}".format(bb_1))
                                print("osition_center_mu<bb_0 : {}/{}".format((position_center_mu<bb_0).sum(),position_center_mu.shape[0]))
                                print("osition_center_mu>bb_1 : {}/{}".format((position_center_mu>bb_1).sum(),position_center_mu.shape[0]))
                            
                        
                        #assert(torch.all(position_center_mu>=bb_0))
                        #assert(torch.all(position_center_mu<=bb_1))
                        if pos_noise > 0: #weights["pos_noise"] > 0:
                            pos_noise = torch.randn(position_center_mu.shape) * (state["pfield_element_size_mu"] * pos_noise)#weights["pos_noise"])      
                            position_center_mu += pos_noise
                        
                        #position_mu = position_center_mu - (pfield_size_mu / 2.0)

                        
                        
                 #if snap == "all" or (snap == "allbutfirst" and i!=0):
                 if True:    
                        if snap == "all":
                            
                            position_center_mu = torch.maximum(torch.minimum(position_center_mu,bb_1),bb_0)

                        if snap == "valid":
                            if i == 0 :# and False:
                                position_center_mu = torch.maximum(torch.minimum(position_center_mu,bb_1),bb_0)
                            else:
                                position_center_mu = torch.maximum(torch.minimum(position_center_mu,bb_1-(pfield_size_mu / 2.0)),bb_0+(pfield_size_mu / 2.0))

                        if snap == "img":
                            
                                position_center_mu = torch.maximum(torch.minimum(position_center_mu,bb_1_max-(pfield_size_mu / 2.0)),bb_0_min+(pfield_size_mu / 2.0))
                            
                        if False:
                            #center = bb_0 + (bb_1 - bb_0)/2.0
                            center = (bb_1 + bb_0)/2.0
                            partition = ((position_center_mu)>center)
                            #position_center_mu = torch.minimum(partition*position_center_mu,bb_1-pfield_size_mu/2.0) \
                            #            + torch.maximum((~partition)*position_center_mu,bb_0+pfield_size_mu/2.0)
                            position_center_mu = torch.minimum(partition*position_center_mu,bb_1) \
                                        + torch.maximum((~partition)*position_center_mu,bb_0)
                            #partition = ((position_center_mu)>center)
                            #position_mu = torch.minimum(partition*position_center_mu,bb_1-pfield_size_mu/2.0) \
                            #            + torch.maximum((~partition)*position_center_mu,bb_0+pfield_size_mu/2.0)
                            print("partition.sum(): {} {}".format(partition.sum(),(~partition).sum()))
                            #position_mu = position_mu_ + torch.maximum(partition*position_mu,bb_0)
                        
                 position_tl_mu = position_center_mu - (pfield_size_mu / 2.0)    
                 bb_0 = position_center_mu - (pfield_size_mu / 2.0)   
                 bb_1 = position_center_mu + (pfield_size_mu / 2.0)
                 
     
                 patchwriter.get_patch_mu(
                                     pw_img,
                                     batch_item_next,
                                     batch_id,
                                     position_tl_mu,
                                     scale_fact,
                                     batch_item_previous = None if i == 0 else batch_item_previous,
                                     augmentation=augmentation,
                                     aug_device=aug_device,
                                     target_device=target_device,
                                     target_element_size=target_element_size_mu,
                                     crop_bb = crop_bb,
                                     warn_if_ooim=warn_if_ooim)    
                 if False:
                     bb_0 = position_center_mu - (pfield_size_mu / 2.0)   
                     bb_1 = position_center_mu + (pfield_size_mu / 2.0)   
                     
                     shape_mu = pw_img.shape_mu()
                     mu2vox = torch.tensor([[1.0/shape_mu[0],0,0],
                                          [0,1.0/shape_mu[1],0],
                                          [0,0,1]],dtype = torch.float32)
                     vox2mu = torch.tensor([[shape_mu[0],0,0],
                                          [0,shape_mu[1],0],
                                          [0,0,1]],dtype = torch.float32)
                
                     A=torch.cat((position_tl_mu,torch.ones([position_tl_mu.shape[0],1])),dim=1)
                     B=batch_item_next.global_rot_mat_s.to(A.device)
                     for bindx in range(A.shape[0]):
                         C = torch.mm(B[bindx,...],mu2vox)   
                         C = torch.mm(C,vox2mu)   
                         
                    #     A[bindx,:] = torch.matmul(A[bindx,:],B[bindx,...])   
                         A[bindx,:] = torch.matmul(A[bindx,:],C)   
                         
                     #bb_0 = A[:,:-1] - (pfield_size_mu / 2.0)   
                     #bb_1 = A[:,:-1] + (pfield_size_mu / 2.0) 
                     
                     diff = -(A[:,:-1])*(A[:,:-1]<0)
                     
                     
                 
                 
                # print("CENTER : {}".format(position_center_mu))     
                     
                 state["bb_0"] = bb_0
                 state["bb_1"] = bb_1 
                 state["bb_1_max"] = bb_1_max
                 state["bb_0_min"] = bb_0_min
                 
                 state["bb_0_target"] = bb_0_target
                 state["i"] = i
                 state["ii"] = ii
                 state["pfield_element_size_mu"] = pfield_element_size_mu       
                 state["pfield_size_mu"] = pfield_size_mu
                         
                 return state
         

    @staticmethod
    def patch_all(
                    pw_img_list,
                    batch_list_list,
                    interpolation_list,
                    batch_id,
                    position_mu_target,
                    scale_facts,
                    patching_strategy="position_fine2coarse",
                    base_scale_indx=0,
                    sample_from_label = -1,
                    augmentation = {},#{"aug_rotations":None,"aug_scales":None,"aug_flipdims":None,"aug_rotations2":None},
                    aug_device="cpu",
                    target_device="cpu",
                    target_element_size_mu=None,
                    warn_if_ooim=False,
                    crop_bb = True,
                    copy_metadata=False,
                    center_normalization = None,#"tl",
                    #weights = {'bg':1.0,'classes':[10000000,10000000]},
                    weights = {'bg':1.0,'classes':[1]},
                    pos_noise=0.0,
                    border_compensation = 0.5,
                    snap="valid",
                    padding_mode_img=pw_settings.default_img_padding_mode,
                    padding_mode_label="zeros",
                    max_depth = -1,
                    ):                   
        
        #coarse2fine_samples = type(position_mu_target) == int 
        #assert((sample_from_label>-1 and coarse2fine_samples) or sample_from_label<0)
        if patching_strategy == "position_fine2coarse":
            num_samples =  position_mu_target.shape[0] 
        elif patching_strategy == "position_coarse2fine":
            num_samples =  position_mu_target.shape[0] 
        elif patching_strategy == "n_coarse2fine":
            num_samples =  position_mu_target
        elif patching_strategy == "n_coarse2fine_valid":
            num_samples =  position_mu_target
        elif patching_strategy == "n_coarse2fine_all":
            num_samples =  position_mu_target
        elif patching_strategy == "debug":
           num_samples =  position_mu_target       
           
            
        #num_samples =  position_mu_target if type(position_mu_target) == int else position_mu_target.shape[0]                    
        state = None
        sample_from_patch = None
        
        max_depth = scale_facts.shape[1] if max_depth == -1 else max_depth
        #for i in range(base_scale_indx,scale_facts.shape[1]):
        for i in range(base_scale_indx,max_depth):
         #   print("position_mu_target : {}".format(position_mu_target))
            pw_debug("patching scale {}".format(i))
         
         
            state = patchwriter.patch_step(pw_img = pw_img_list[0],
                                            batch_list = batch_list_list[0],
                                            batch_id = batch_id,
                                            positions_mu = position_mu_target,
                                            scale_facts = scale_facts,
                                            patching_strategy = patching_strategy,
                                            state = state if i>base_scale_indx else None,
                                            #state = state if i>0 else None,
                                            base_scale_indx = base_scale_indx,
                                            sample_from_patch = sample_from_patch,
                                            augmentation = augmentation,
                                            aug_device=aug_device,
                                            target_device=target_device,                                        
                                            target_element_size_mu=target_element_size_mu,                    
                                            crop_bb=crop_bb,#
                                            warn_if_ooim=warn_if_ooim,
                                            weights=weights,
                                            pos_noise=pos_noise,
                                            border_compensation=border_compensation,
                                            snap=snap,
                                            )
            

                
            if sample_from_label > -1:
                #print("sample_from_label :",sample_from_label)
                patch_scale_id = state["ii"]
                img = pw_img_list[sample_from_label]
                batch = batch_list_list[sample_from_label].patchlist[patch_scale_id]
                interp = interpolation_list[sample_from_label]
                batch_list_list[0].patchlist[patch_scale_id].copy_to_patch_item( 
                   img,
                   np.arange(num_samples)+batch_id,
                   patch_item_target=batch,
                   interpolation=interp,
                   copy_metadata=copy_metadata,
                   padding_mode=padding_mode_label)      
                sample_from_patch = batch.tensor
               
            #if True:
            #    for bind in range(len(batch_list_list)):
            #        if bind == 0:
            #            topselect = torch.randperm(batch_list_list[0].patchlist[0].tensor.shape[0])
            #        batch_list_list[bind].copy_to(batch_list_list[bind],
            #                                      topselect,
            ##                                      range(0,scale_facts.shape[1]),#range(state['ii'],scale_facts.shape[1]),
             #                                     copy_meta=True,copy_trafos=True)            

        pw_debug("patched all scales ")
        if center_normalization is not None and center_normalization:# and not coarse2fine_samples:
      #      assert(False)
            #needs revision
            pfield = batch_list_list[0].patchlist[0].tensor.shape[2:]
            pfield_size_vox = torch.tensor(pfield,dtype=torch.float32)
            pfield_size_mu_target = pfield_size_vox * tls.tt(target_element_size_mu,dtype=torch.float32)
            
            if center_normalization == "tl":
            #center = torch.cat(((position_mu_target + pfield_size_mu_target / 2).flip(dims=(1,)),torch.ones([num_samples,1])),dim=1)
                center = torch.cat((position_mu_target.flip(dims=(1,)) ,torch.ones([num_samples,1])),dim=1)
            else:
                center = torch.cat(((position_mu_target + pfield_size_mu_target / 2).flip(dims=(1,)),torch.ones([num_samples,1])),dim=1)
                
            shape_mu = pw_img_list[0].shape_mu()
            dim = pw_img_list[0].dim
            if dim == 3:
                mu2rel_scale = torch.tensor([[1.0/shape_mu[2],0,0,0],
                                          [0,1.0/shape_mu[1],0,0],
                                          [0,0,1.0/shape_mu[0],0],
                                          [-0.5,-0.5,-0.5,1]],dtype = torch.float32,device=aug_device)
                mu2rel_shift = torch.tensor([[2.0,0,0,0],
                                          [0,2.0,0,0],
                                          [0,0,2.0,0],
                                          [0,0,0,1]],dtype = torch.float32,device=aug_device)
            else:
                 mu2rel_scale = torch.tensor([[1.0/shape_mu[1],0,0],
                                          [0,1.0/shape_mu[0],0],
                                          [-0.5,-0.5,1]],dtype = torch.float32,device=aug_device)
                 mu2rel_shift = torch.tensor([[2.0,0,0],
                                          [0,2.0,0],
                                          [0,0,1]],dtype = torch.float32,device=aug_device)               
            
            mu2rel = torch.mm(mu2rel_scale,mu2rel_shift)
            if dim == 3:
                p_center = (batch_list_list[0].patchlist[0].rel_coordinates[:,-1,-1,-1,:]-batch_list_list[0].patchlist[0].rel_coordinates[:,0,0,0,:])
            else:
                p_center = (batch_list_list[0].patchlist[0].rel_coordinates[:,-1,-1,:]-batch_list_list[0].patchlist[0].rel_coordinates[:,0,0,:])
                
            center_rel_wanted = torch.mm(center,mu2rel)[:,:dim]
            if center_normalization == "center":
                center_rel_wanted -= p_center / 2.0
            if dim == 3:
                center_rel_is = batch_list_list[0].patchlist[0].rel_coordinates[:,0,0,0,:]
                center_diff = center_rel_is - center_rel_wanted
                for indx  in range(len(batch_list_list[0].patchlist)):
                    batch_list_list[0].patchlist[indx].rel_coordinates -= center_diff[:,None,None,None,:]
            else:
                center_rel_is = batch_list_list[0].patchlist[0].rel_coordinates[:,0,0,:]
                center_diff = center_rel_is - center_rel_wanted
                for indx  in range(len(batch_list_list[0].patchlist)):
                    batch_list_list[0].patchlist[indx].rel_coordinates -= center_diff[:,None,None,:]
           
        
        pw_debug("copy remaing patches from images/labels")
        for img,batch,interp,indx in zip(pw_img_list,batch_list_list,interpolation_list,range(len(interpolation_list))):
            #                 print("PUT THIS BACK !")
            if sample_from_label != indx:
                             batch_list_list[0].copy_to_patch_item( 
                                img,
                                np.arange(num_samples)+batch_id,
                                patch_batch_target=batch,
                                interpolation=interp,
                                copy_metadata=copy_metadata,
                                padding_mode=padding_mode_img)                 
            else:
                pass
                #print("skipping gatherering image data from data list index {} (aready done during saliency map sampling)".format(indx))
        pw_debug("done copying remaing patches from images/label")

