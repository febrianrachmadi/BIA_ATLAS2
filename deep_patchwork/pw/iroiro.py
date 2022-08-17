#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 21:50:22 2020

@author: skibbe
"""

    def sample_MP(self,batch_batch_img,batch_batch_labels,
               batch_id,
               num_samples,
               pfield=(0,0,0),
               scale_interval=None,
               non_linear_def_spacing=30,
                non_linear_def_factor=0):
        
        #pfield = patches.patchlist[0].tensor.shape[2:]
        
        sample_scale=True  
        if scale_interval is None:
            scale = (1,1,1)
            sample_scale=False    

        
        max_patches = batch_batch_img.patchlist[0].tensor.shape[0]
        assert(batch_id+num_samples <= max_patches)
        
        processes = []
        for n in range(num_samples):
            indx = tls.rand_pic_array(self.pdf)
            
            pos_top_left = tls.ind2sub(self.shape, indx) 
            pos_centered = np.floor(pos_top_left + 0.5*np.array(pfield) * (0.5*np.random.rand(len(pfield))-1))
            pos_centered = np.floor(pos_top_left - 0.5*np.array(pfield)) + 3*np.random.randn(len(pfield))
            pos_centered = np.minimum(pos_centered, self.shape-1).astype(np.int32)
            
            if sample_scale:
                scale=  list(np.array(scale_interval[:self.dim]) + np.random.rand(self.dim) * np.array(scale_interval[self.dim:]))

            p = mp.Process(target=self.img_pyramid.write_patch_static, args=([self.img_pyramid,self.label_pyramid],
                                    [batch_batch_img,batch_batch_labels],
                                    batch_id,
                                    pos_centered,
                                    batch_batch_img.n_scales,
                                    None,
                                    scale,
                                    0,
                                    "random",
                                    ["bilinear","nearest"],
                                    non_linear_def_spacing,
                                    non_linear_def_factor,))
            
            
                    
            p.start()
            processes.append(p)
            
            
            batch_id += 1
        
        for p in processes:
                p.join()
            




SAMPLER>..
 def write_patch_jit(self,batch_list,batch_id,
                    position,
                    n_scales,
                    base_scale=None,
                    scale = (1,1,1),
                    rotate=0,
                    mode = "deterministic",#mode="centered",#mode = "deterministic",
                    interpolation="bilinear",
                    non_linear_def_spacing=30,
                    non_linear_def_factor=0):#mode="centered"):#mode = "deterministic"):#mode = "random"):
                    #mode = "deterministic"):
        
        #print("new write patch")
        write_patch_static_jit([self],
                                   [batch_list],
                                   batch_id,
                                   position,
                                   n_scales,
                                   base_scale,
                                   scale,
                                   rotate,
                                   mode,
                                   interpolation,
                    non_linear_def_spacing,
                    non_linear_def_factor)

@torch.jit.script
def gkernel(dim:int=2,w_size:int=3,sigma:float=1):
    if dim==2:
        dx = torch.arange(0,w_size)
        YY, XX = torch.meshgrid(dx, dx)
        XX = (XX - (w_size//2)).float()
        YY = (YY - (w_size//2)).float()
        
        g_filt  = torch.exp(-(XX**2+YY**2)/sigma**2)
        g_filt  /= torch.sum(g_filt)
        return g_filt
    else:
        dx = torch.arange(0,w_size)
        ZZ, YY, XX = torch.meshgrid(dx, dx, dx)
        XX = (XX - (w_size//2)).float()
        YY = (YY - (w_size//2)).float()
        ZZ = (ZZ - (w_size//2)).float()
        
        g_filt  = torch.exp(-(XX**2+YY**2+ZZ**2)/sigma**2)
        g_filt  /= torch.sum(g_filt)
        return g_filt
    
#print("new write patch 2")
@torch.jit.script
def abs2rel(pos,shape):
    return 2.0 * ((pos)/(shape-1) - 0.5) 

@torch.jit.script
def non_linear_def(shape,spacing:int=30,factor:float=5):
    dim = len(shape)
    shape_t = shape
    #shape_t = torch.tensor(shape)
    noise_shape = (shape_t/spacing).int()
    
    #print("dim: {}".format(dim))
   # print("shape_t: {}".format(shape_t))
    #print("noise_shape: {}".format(noise_shape))
    d_xyz = 0.5/shape_t
    
    nf = factor * spacing
    if dim==2:
        n_weight = torch.zeros([1,dim,1,1]);
        n_weight[0,:,0,0] = d_xyz * nf
    else:
        n_weight = torch.zeros([1,dim,1,1,1]);
        n_weight[0,:,0,0,0] = d_xyz * nf
         
    
   # print((1,dim,))
   # print(list(noise_shape.numpy()))
    #gshape = torch.
    #dim_t = torch.tensor([1,dim], dtype=torch.int32)
    #print(dim_t)
    #print(noise_shape)
    
    #gshape = torch.cat((dim_t,noise_shape),dim=0).tolist()
    
    #gshape = [1,dim]+noise_shape.tolist():List[int]
    #print(gshape)
    if dim == 3:
        gshape = [int(1),int(3),int(noise_shape[0].item()),int(noise_shape[1].item()),int(noise_shape[2].item())]
    else:
        gshape = [int(1),int(2),int(noise_shape[0].item()),int(noise_shape[1].item())]
    
    noise_grid = torch.randn(gshape,dtype = torch.float32)
    
    
    noise_grid*=n_weight
    
    #print(noise_grid.shape)
   
        
    if dim==2:
        noise_grid[0,:,0,:] = 0
        noise_grid[0,:,:,0] = 0
        noise_grid[0,:,-1,:] = 0
        noise_grid[0,:,:,-1] = 0
    else:
        noise_grid[0,:,0,0,:] = 0
        noise_grid[0,:,0,:,0] = 0
        noise_grid[0,:,0,0,:] = 0
        noise_grid[0,:,-1,-1,:] = 0
        noise_grid[0,:,-1,:,-1] = 0
        noise_grid[0,:,-1,-1,:] = 0

    
    if dim == 3:
        kernel = gkernel(dim=3,w_size=5,sigma=2.0)
        #print(kernel.shape)
        kernel = torch.cat((kernel[None,None,:,:],kernel[None,None,:,:],kernel[None,None,:,:]),dim=0)
        #print(kernel.shape)
        noise_grid = torch.nn.functional.conv3d(noise_grid,kernel,padding=2,groups=3)
    else:
        kernel = gkernel(dim=2,w_size=5,sigma=2.0)
        kernel = torch.cat((kernel[None,None,:,:],kernel[None,None,:,:]),dim=0)
        noise_grid = torch.nn.functional.conv2d(noise_grid,kernel,padding=2,groups=2)
        
    
    
    if dim==2:
        noise_grid[0,:,0,:] = 0
        noise_grid[0,:,:,0] = 0
        noise_grid[0,:,-1,:] = 0
        noise_grid[0,:,:,-1] = 0
    else:
        noise_grid[0,:,0,0,:] = 0
        noise_grid[0,:,0,:,0] = 0
        noise_grid[0,:,0,0,:] = 0
        noise_grid[0,:,-1,-1,:] = 0
        noise_grid[0,:,-1,:,-1] = 0
        noise_grid[0,:,-1,-1,:] = 0
    #print("JIT2")
    return noise_grid


@torch.jit.script
def pyramidal_shifts(base_scale:int,
                     scale:Tuple[int],
                     scales:Tuple[int],
                     img_pyramid,
                     trans_mat,
                     trans_inv_mat,
                     trafo,
                     #rot_mat,
                     grid_bb,
                     mode = "random"
                     ):
    offsets = {}
    init = True       
    print("JIT3")
    #make sure base scale image comes first
    for s in [base_scale]+list(reversed(scales)) :
        if s < base_scale:
            continue
        if s == base_scale:
            if init:
                init = False
            else:
                continue
        best_img = min(max(0,round(s*np.min(scale))),max(scales)) 
        
        dim = img_pyramid.dim
        #compute bounding box
        if dim == 2:
            res_mat = torch.tensor([[(2**s),0,0],[0,(2**s),0],[0,0,1]],dtype = torch.float32)
        else:
            res_mat = torch.tensor([[(2**s),0,0,0],[0,(2**s),0,0],[0,0,(2**s),0],[0,0,0,1]],dtype = torch.float32)
            
        mat = torch.mm(trans_inv_mat,res_mat)
        torch.mm(mat,trafo,out=mat)
        torch.mm(mat,trans_mat,out=mat)
        grid_bb_ = torch.matmul(grid_bb,mat).clone()
        
        #mat = torch.mm(res_mat,trans_inv_mat)
        #torch.mm(trafo,mat,out=mat)
        #torch.mm(trans_mat,mat,out=mat)
        #grid_bb_ = torch.matmul(grid_bb,mat).clone()
        
        
        grid_0 = grid_bb_[0,:dim]
        grid_1 = grid_bb_[-1,:dim]
         
        offset = torch.zeros(dim)
        
        #initialize patch in full resolution
        if s == base_scale:
            #image boundaries
            if dim == 2:
                bb_img_0 = torch.tensor([-1.0,-1.0]) 
                bb_img_1 = torch.tensor([1.0,1.0])  
            else:
                bb_img_0 = torch.tensor([-1.0,-1.0,-1.0]) 
                bb_img_1 = torch.tensor([1.0,1.0,1.0])  
                
            #bb_img_0 = torch.mm(rot_mat,torch.tensor([[-1.0],[-1.0],[1.0]]))[0,:2]
            #bb_img_1 = torch.mm(rot_mat,torch.tensor([[1.0],[1.0],[1.0]]))[0,:2]
            
            #bb of patch in full res
            bb_inner_0 = grid_bb_[0,:dim].clone()
            bb_inner_1 = grid_bb_[-1,:dim].clone()
                                
            #bb of next (higher/coarser) scale patch
            #-> at the beginning, img bb
            bb_outer_0 = bb_img_0.clone()
            bb_outer_1 = bb_img_1.clone()
            
        else:
            bb_corrections = True
            #align img top/left with inner patch bb t/l
            offset = bb_inner_0 - grid_0
           
            if bb_corrections:
                #correct if bb is out of image bottom/right
                overshoot = torch.clamp((grid_1 + offset) - bb_outer_1,min=0)
                offset -= overshoot
            else:
                overshoot  = 0
     
            #compute valid margin between inner/outer bb
            margin_best = ( (grid_1-grid_0) - (bb_inner_1-bb_inner_0)-overshoot)
            margin_constraint =       (grid_0+offset -bb_outer_0)
            margin = torch.min(margin_best,margin_constraint)
            offset -= margin * (torch.rand(dim) if mode == "random" else 0.5) 
            
            #correct if bb is out of image (if possible)
            if bb_corrections:
                clamp_left = torch.clamp(bb_img_0 - (grid_0 + offset),min=0)
                clamp_right = torch.clamp((grid_1 +  offset) - (bb_img_1 ),min=0)
                
                clamp_right[clamp_left>0] = 0
                offset += clamp_left - clamp_right 


            #grid + offset is current patch FOV bb
            
            #set next outer bb to current inner bb
            bb_outer_0 = grid_0 + offset
            bb_outer_1 = grid_1 + offset 
        
        #print("offset: {} : {}".format(s,offset))
        offsets[s] = offset.clone()
    return offsets  


def write_patch_static_jit(
                    img_pyramid_list,
                    batch_list_list,
                    batch_id,
                    position,
                    n_scales,
                    base_scale=None,
                    scale = (1,1,1),
                    rotate=0,
                    mode = "random",
                    interpolation=["bilinear","nearest"],
                    non_linear_def_spacing=30,
                    non_linear_def_factor=0):
        
          
        
        
        
        
                    
        #list of output batches
        batch_list = batch_list_list[0]
        #list of input images (usually img and label)
        pfield = batch_list.patchlist[0].tensor.shape[2:]

        #reference img pyramid
        #we assume all have same attributes (img sizes etc.)       
        img_pyramid = img_pyramid_list[0]
        
        scales = range(n_scales)
        #if scales is None:
        #    scales = range(len(img_pyramid.tensors))
        if base_scale is None:
            base_scale = min(scales)
            
            
        t = img_pyramid.tensors[0]
        shape = torch.tensor(t.shape)
        shape_t = shape[2:].float()
        
        #print("!!!! {}".format(img_pyramid.dim))
        #print("!!!! {}".format(scale))
        scale_t = torch.tensor(scale[:img_pyramid.dim],dtype=torch.float32)
        
        
        # scaled image should have same borders than original image
        position_offset = (scale_t-1)*torch.tensor(pfield)/2.0
        position_scaled = position_offset + torch.tensor(position)* (shape_t-2.0*position_offset)/shape_t
        
        position_t = (position_scaled + (torch.tensor(pfield) /2))*(2**base_scale) 
       
     
        bb_start = torch.round(position_t)  -  (torch.tensor(pfield) /2.0)
        bb_end = bb_start + torch.tensor(pfield)
        
    
        pos_rel = abs2rel((position_t).float(),shape[2:])
        #pos_rel_scaled = pos_rel
        #print(position_t) 
        #print(pos_rel)
        
        grid = torch.ones((1,)+pfield+(img_pyramid.dim+1,),dtype = torch.float32)
        grid_ = torch.zeros(grid.shape,dtype = torch.float32)

        #grid_distortion = torch.zeros(grid.shape,dtype = torch.float32)

        if img_pyramid.dim == 2:
            dx = torch.arange(bb_start[0],bb_end[0])
            dy = torch.arange(bb_start[1],bb_end[1])
            grid_y, grid_x = torch.meshgrid(dx, dy)
            
            grid[0,:,:,0] = 2.0 * (grid_x / (float(t.shape[3]-1)) -0.5) 
            grid[0,:,:,1] = 2.0 * (grid_y / (float(t.shape[2]-1)) -0.5) 
            
            trans_inv_mat = torch.tensor([[1,0,0],
                                          [0,1,0],
                                          [-pos_rel[1],-pos_rel[0],1]],dtype = torch.float32)
            trans_mat = torch.tensor([[1,0,0],
                                      [0,1,0],
                                      [pos_rel[1],pos_rel[0],1]],dtype = torch.float32)
            #if trafo is None:
                #scale = (1,1)
            scale_mat = torch.tensor([[scale[0],0,0],
                                      [0,scale[1],0],
                                      [0,0,1]],dtype = torch.float32)
            trafo = scale_mat
            has_rotation = False
            if rotate!=0:
                has_rotation = True
                rot_mat = torch.tensor([[math.cos(rotate),math.sin(rotate),0],[-math.sin(rotate),math.cos(rotate),0],[0,0,1]],dtype = torch.float32)
            #trafo = torch.mm(rot_mat,scale_mat)
    
            grid_bb = torch.cat((grid[None,0,0,0,:],grid[None,0,-1,-1,:]),dim=0)
        
        if img_pyramid.dim == 3:
            dx = torch.arange(bb_start[0],bb_end[0])
            dy = torch.arange(bb_start[1],bb_end[1])
            dz = torch.arange(bb_start[2],bb_end[2])
            #grid_y, grid_x , grid_z = torch.meshgrid(dx, dy, dz)
            grid_z, grid_y , grid_x = torch.meshgrid(dx, dy, dz)
            
            grid[0,...,0] = 2.0 * (grid_x / (float(t.shape[4]-1)) -0.5) 
            grid[0,...,1] = 2.0 * (grid_y / (float(t.shape[3]-1)) -0.5) 
            grid[0,...,2] = 2.0 * (grid_z / (float(t.shape[2]-1)) -0.5) 
                        
            trans_inv_mat = torch.tensor([[1,0,0,0],
                                          [0,1,0,0],
                                          [0,0,1,0],
                                          [-pos_rel[2],-pos_rel[1],-pos_rel[0],1]],dtype = torch.float32)
            trans_mat = torch.tensor([[1,0,0,0],
                                          [0,1,0,0],
                                          [0,0,1,0],
                                          [pos_rel[2],pos_rel[1],pos_rel[0],1]],dtype = torch.float32)
            
            scale_mat = torch.tensor([[scale[0],0,0,0],
                                      [0,scale[1],0,0],
                                      [0,0,scale[2],0],
                                      [0,0,0,1]],dtype = torch.float32)
               
            trafo = scale_mat 
            
            has_rotation = False
            if isinstance(rotate, torch.Tensor):
                has_rotation = True
                rotate/=torch.norm(rotate)
                qr = rotate[0]
                qi = rotate[1]
                qj = rotate[2]
                qk = rotate[3]
                #print(rotate)
                
                rot_mat = torch.tensor([[1.0-2.0*(qj**2+qk**2),2*(qi*qj+qk*qr),  2*(qi*qk-qj*qr),  0],
                                          [2*(qi*qj-qk*qr),     1-2*(qi**2+qk**2),2*(qj*qk+qi*qr),  0],
                                          [2*(qi*qk+qj*qr),     2*(qj*qk-qi*qr),  1-2*(qi**2+qj**2),0],
                                          [0,0,0,1]],dtype = torch.float32)
                print(rot_mat)
                print(rot_mat.det())
        
            grid_bb = torch.cat((grid[None,0,0,0,0,:],grid[None,0,-1,-1,-1,:]),dim=0)
        
        if mode not in ["centered"]:
            shifts_ = pyramidal_shifts(
                base_scale,
                scale,
                scales,
                img_pyramid,
                trans_mat,
                trans_inv_mat,
                scale_mat,#trafo,#scale_mat,#,
                grid_bb,
                mode = mode#"deterministic"
                )
            #print(shifts_)
            #shifts_r = torch.ones(img_pyramid.dim+1)
            #shifts_r[:img_pyramid.dim] = shifts_[0]
            #shifts_r = torch.mm(rot_mat,shifts_r[:,None])
            #print(shifts_r)
        else:
            print("no pyramidal_shifts")
        
        if non_linear_def_factor>0:
            noise_grid = non_linear_def(shape=shape_t,
                                        spacing=non_linear_def_spacing,
                                        factor=non_linear_def_factor)
       # print(shifts_)
        ref_pos = torch.zeros([img_pyramid.dim])
        
        
        crop_offsets_0 = {}
        crop_offsets_1 = {}
        init = True  
        #for s in list(reversed(scales)):
        #make sure base scale image comes first
        for s in [base_scale]+list(reversed(scales)) :
            if s < base_scale:
                continue
            if s == base_scale:
                if not init:
                    continue      
        
            #if img_pyramid.dim == 2:
            #best_img = min(max(0,round(s*np.mean(scale))),max(scales)) 
            
           # best_img = 0

            if img_pyramid.dim == 2:
                res_mat = torch.tensor([[(2**s),0,0],[0,(2**s),0],[0,0,1]],dtype = torch.float32)
            else:
                res_mat = torch.tensor([[(2**s),0,0,0],
                                                [0,(2**s),0,0],
                                                [0,0,(2**s),0],
                                                [0,0,0,1]],dtype = torch.float32)
            
            if init:
                init = False            
            
            mat = torch.mm(trans_inv_mat,res_mat)
            torch.mm(mat,trafo,out=mat)
            torch.mm(mat,trans_mat,out=mat)
            #torch.matmul(grid,mat,out=grid_)
            
            if img_pyramid.dim == 2:
                grid_bb = torch.cat((grid[None,0,0,0,:],grid[None,0,-1,-1,:]),dim=0).clone()
            else:
                grid_bb = torch.cat((grid[None,0,0,0,0,:],grid[None,0,-1,-1,-1,:]),dim=0).clone()
            torch.matmul(grid_bb,mat,out=grid_bb)
            
            
            #if mode not in ["centered"]:    
                #grid_bb[...,:img_pyramid.dim] += shifts_[s] 
                #grid_[...,:img_pyramid.dim] += shifts_[s] 
            #print("!")
            if has_rotation:
                center_ = (grid_bb[1,:] - grid_bb[0,:]) / 2.0 + grid_bb[0,:]
                center_[img_pyramid.dim] = 1
            
                if img_pyramid.dim==2:
                    trans_inv_mat_rot = torch.tensor([[1,0,0],
                                              [0,1,0],
                                              [-center_[0],-center_[1],1]],dtype = torch.float32)
                    trans_mat_rot = torch.tensor([[1,0,0],
                                              [0,1,0],
                                              [center_[0],center_[1],1]],dtype = torch.float32)
                else:
                    trans_inv_mat_rot = torch.tensor([[1,0,0,0],
                                              [0,1,0,0],
                                              [0,0,1,0],
                                              [-center_[0],-center_[1],-center_[2],1]],dtype = torch.float32)
                    trans_mat_rot = torch.tensor([[1,0,0,0],
                                              [0,1,0,0],
                                              [0,0,1,0],
                                              [center_[0],center_[1],center_[2],1]],dtype = torch.float32)
                    
                if mode not in ["centered"]:    
                    grid_bb[...,:img_pyramid.dim] += shifts_[s] 
                    if img_pyramid.dim==2:
                        shifts_m = torch.tensor([[1,0,0],
                                                  [0,1,0],
                                                  [shifts_[s][0],shifts_[s][1],1]],dtype = torch.float32)
                    else:
                        shifts_m =torch.tensor([[1,0,0,0],
                                              [0,1,0,0],
                                              [0,0,1,0],
                                              [shifts_[s][0],shifts_[s][1],shifts_[s][2],1]],dtype = torch.float32)
                         
                        
                    torch.mm(mat,shifts_m,out=mat)

                torch.mm(mat,trans_inv_mat_rot,out=mat)
                torch.mm(mat,rot_mat,out=mat)
                torch.mm(mat,trans_mat_rot,out=mat)
                torch.matmul(grid,mat,out=grid_)
                
            else:
                torch.matmul(grid,mat,out=grid_)
                if mode not in ["centered"]: 
                    grid_bb[...,:img_pyramid.dim] += shifts_[s] 
                    grid_[...,:img_pyramid.dim] += shifts_[s] 
                
            grid_0 = grid_bb[0,:img_pyramid.dim]
            grid_1 = grid_bb[1,:img_pyramid.dim]
            offset = torch.zeros(img_pyramid.dim)                
            
            crop_offsets_0[s] = grid_0.clone()
            crop_offsets_1[s] = grid_1.clone()
  
    
  
            if non_linear_def_factor>0:
                noise_grid_s =  torch.nn.functional.grid_sample(noise_grid, 
                                                        grid_[...,:img_pyramid.dim], 
                                                        mode="bilinear", 
                                                        padding_mode='zeros', 
                                                        align_corners=True)
               
                grid_[0,...,0] += noise_grid_s[0,0,...]
                grid_[0,...,1] += noise_grid_s[0,1,...]
                if img_pyramid.dim == 3:
                    grid_[0,...,2] += noise_grid_s[0,2,...]
                    
            for bl in range(len(batch_list_list)):
                avaliable_scales = len(img_pyramid_list[bl].tensors)
                #best_img = min(max(0,round(s*np.min(scale))),max(scales)) 
                best_img = min(max(0,round(s*np.min(scale))),avaliable_scales-1) 
                
                t_s = img_pyramid_list[bl].tensors[best_img]
                
                

                #rep_dims = torch.Size((t_s.shape[1],)+(1,)*(img_pyramid.dim+1))
                patch = torch.nn.functional.grid_sample(t_s, 
                                                        grid_[...,:img_pyramid.dim], 
                                                        mode=interpolation[bl] if isinstance(interpolation,list)  else interpolation, 
                                                        padding_mode='zeros', 
                                                        align_corners=True)
                
                #print(grid_.shape)
                #print(grid_)
                #print(grid)                
                #print(t_s.shape)
                #print("t_s minmax: {} {}".format(t_s[0,0,...].min(),t_s[0,0,...].max()))
                #print("patch minmax: {} {}".format(patch[0,0,...].min(),patch[0,0,...].max()))
                #print(batch_list_list[bl].patchlist[s].tensor[batch_id,...].shape)

                #print(patch.shape)
                batch_list_list[bl].patchlist[s].tensor[batch_id,...] = patch
                batch_list_list[bl].patchlist[s].offsets[batch_id,:] =   0
                #print("batch_list_list minmax: {} {}".format(batch_list_list[bl].patchlist[s].tensor[batch_id,...].min(),batch_list_list[bl].patchlist[s].tensor[batch_id,...].max()))
        #print(crop_offsets_0)
        #for s in range(1,len(crop_offsets)):
        #for s in range(1+base_scale,len(crop_offsets)):   
        for s in range(1+base_scale,max(scales)+1):   
            diff =  crop_offsets_0[s-1] - crop_offsets_0[s]
            #print("diff {} {} {}".format(s-1,s,diff))
            patch_size = crop_offsets_1[s] - crop_offsets_0[s]
            #crop_offset_px = torch.floor(torch.tensor(pfield)*diff/patch_size)
            #crop_offset_px = torch.ceil(torch.tensor(pfield)*diff/patch_size)
            #print(torch.tensor(pfield))
            #print(torch.tensor(diff))
            #print(torch.tensor(patch_size))
            
            crop_offset_px = torch.round(torch.tensor(pfield)*diff/patch_size)
            
            crop_offset_px = torch.min(crop_offset_px,torch.tensor(pfield,dtype=torch.float32)/2)
            
            crop_offset_px = torch.clamp(crop_offset_px,min=0)
            crop_offset_px = torch.flip(crop_offset_px,dims=[0])
            #batch_list.patchlist[s].offsets[batch_id,:] = torch.flip(crop_offset_px,dims=[0])
            
            for bl in range(len(batch_list_list)):
                batch_list.patchlist[s].offsets[batch_id,:] = crop_offset_px
            #batch_list.patchlist[s].offsets[batch_id,:] = 0
            #print(torch.flip(crop_offset_px,dims=[0]))
            
            