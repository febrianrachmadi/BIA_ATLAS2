import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict

import matplotlib.pyplot as plt
#import torch.optim as optim
import math
import random
import inspect
import time
import torch.nn as nn
import time
from datetime import date
import torch.optim as optim
import types
import copy
#from . import pw_debug
#%%

from . import vis as pw_vis
from . import tools as tls
from . import data as pw_data
from .tools import pw_debug


from .units import model_factory
from .units import layersND
    
from .patch import patchwriter
from .losses import patchwork_loss
from .settings import pw_settings 


#import bad_grad_viz as bad_grad

class deep_patch_work(nn.Module):
#    default_patch_forwarding_padding_mode = "zeros"
    #default_patch_forwarding_padding_mode = "border"
    #default_img_padding_mode = "zeros"
    
    def __init__(self,pw_layout={
            "fin":1,
            "dim":2,
            "normalize_inputs":None,
            "mode":"normal",
            "prop_norm":"BatchNorm",
            "labels":None,
            "pfield":(32,),
            "scales":
                [
                    {"fout":1,"params":{"feats":[16,32,64]},"arc":None,"fin":[1]},
                    {"fout":1,"params":{"feats":[16,32,64]},"arc":None,"fin":[2]},
                ]
            },grid_crop='nearest',
                optimize_units=False,verbose=0):
        super().__init__()
      #  print("yea")
        #self.loaded = False
        self.checkpoint = None
        if type(pw_layout) == str:
            pw_debug("loading patchwork from file :"+pw_layout)    
            self.checkpoint = torch.load(pw_layout)
            self.train_state = self.checkpoint["train_state"]
            #self.optimizer.load_state_dict
            pw_layout = self.checkpoint["pw_layout"]
            if "pw_settings" in pw_layout:
                assert(pw_settings.default_patch_forwarding_padding_mode==pw_layout["pw_settings"].default_patch_forwarding_padding_mode)
                assert(pw_settings.default_img_padding_mode==pw_layout["pw_settings"].default_img_padding_mode)
            else:
                print("w_settings were not tracked that time. settings might have changed")
        else:
            self.train_state = types.SimpleNamespace()#tls.cnt
            self.train_state.iter = 0
            pw_debug("creating a new patchwork")    
            pw_layout["pw_settings"] = pw_settings
            
        
        self.pw_layout = copy.deepcopy(pw_layout)
        
        #if "forward_padding_mode" in self.pw_layout:
        #    self.default_patch_forwarding_padding_mode = self.pw_layout["forward_padding_mode"]
        #if "img_padding_mode" in self.pw_layout:
        #    self.default_img_padding_mode = self.pw_layout["img_padding_mode"]
            
        
        if self.checkpoint is None:
            self.pw_layout["scales"] = []
            #self.pw_layout["scales"] = {}
            #del self.pw_layout["scales"]
            #self.pw_layout["scales"] = {}
        
        #print(self.pw_layout)
            
            if type(pw_layout["scales"]) == dict:
                pw_u = pw_layout["scales"]
                pw_l = []
                patchlevels = pw_u["patchlevels"]
                #self.pw_layout["patchlevels"] = patchlevels
                
                #preprocess = pw_layout["preprocess"] if "preprocess" in pw_layout else None
                #patchlevels = pw_u["scale_facts"].shape[1]
                for d in range(patchlevels):
                    layer_dict = {"fout":pw_layout["labels"] if d==0 else pw_u["out"],
                              "params":copy.deepcopy(pw_u["params"]),
                              "arc":pw_u["arc"],
                              "fin":[d,d+1] if d < patchlevels-1 else [d],
                              #"s":pw_u["scale_facts"][:,d],
                              }
                    if "preprocess" in pw_u:
                        #print("yea")
                        if d == 0:
                            layer_dict["preprocess"] = pw_u["preprocess"] 
                        else:
                            layer_dict["preprocess"] = "same" 
                    pw_l += [layer_dict]
                
                pw_layout["scales"] = pw_l
                
                
        
        #self.optimizer = None
        #self.pw_layout = pw_layout
        
        
        
        self.scatterer = tls.scatterer()
        self.grid_crop = grid_crop
        coarsest_layer = len(pw_layout["scales"])
        self.depth = coarsest_layer
        
        self.units = torch.nn.ModuleList()
        self.pw_layers = torch.nn.ModuleDict()
        norm_factory = torch.nn.Identity
        if pw_layout["prop_norm"] is not None:
            norm_factory = getattr(layersND,pw_layout["prop_norm"])

        self.im_mod = pw_layout["fin"]
        self.dim = pw_layout['dim']
        
        self.normalization = pw_layout["normalize_inputs"] 
        if self.normalization == "Instance":
            print("Instance Norm")
            self.normalize_inputs = torch.nn.InstanceNorm2d(1,affine=False) if self.dim==2 else torch.nn.InstanceNorm3d(1,affine=False)
        else:
            self.normalize_inputs =  torch.nn.Identity()
        print("normalization : {}".format(self.normalization))
        
        self.unit_in = []
        self.preprocess_nets = torch.nn.ModuleDict()
        
        preprocess_net = None
        #self.pass_scale_info = False
        for s in range(coarsest_layer):
            lo = pw_layout['scales'][s]
            
            preprocessed_fout = pw_layout["fin"] 
            if "preprocess" in lo:
                pnet = lo["preprocess"]
                if type(pnet) == str and pnet=="same":
                    self.preprocess_nets[str(s)] = self.preprocess_nets[preprocess_net[0]] 
                    preprocessed_fout = preprocess_net[1]
                else:
                    preprocess_settings = pnet["params"]
                    preprocess_settings['dim'] = pw_layout['dim']
                    preprocess_settings['fin'] = pw_layout["fin"]
                    #preprocess_settings['fout'] = pw_layout["fout"]
                    preprocessed_fout = preprocess_settings['fout']
                    arc = eval(pnet["arc"])
                    self.preprocess_nets[str(s)] = arc(settings=preprocess_settings)
                    
                    
                    preprocess_net = [str(s),preprocessed_fout]
                
                
            in_scales = lo['fin']
            
            #fin = 0
            fin = [0,0]
            for f in in_scales:
                if (f == s): #or (self.mode=="feed"): 
                        #fin += preprocessed_fout#pw_layout["fin"] 
                        fin[0] = preprocessed_fout#pw_layout["fin"] 
                else:
                        #fin += pw_layout['scales'][f]["fout"]
                        fin[1] += pw_layout['scales'][f]["fout"]
            
            #print(self.checkpoint)
            #print("WTF")
            if self.checkpoint is None:
                unit_net_settings = lo['params']
                unit_net_settings['name'] = "s"+str(s)
                unit_net_settings['fin'] = fin
                unit_net_settings['fout'] = lo['fout']
                unit_net_settings['dim'] = pw_layout['dim']
                unit_net_settings['pfield'] = pw_layout['pfield']
                unit_net_settings['scale'] = [s,coarsest_layer-1]
                if ("copy" in pw_layout) and (s in pw_layout["copy"]):
                    self.pw_layout["scales"]+= [
                                                   {
                                                   "fin":pw_layout['scales'][s]["fin"],
                                                   "fout":pw_layout['scales'][s]["fout"],
                                                   }]
                else:
                    arc = lo["arc"]
                    #print(self.pw_layout["scales"])
                    #print(self.pw_layout["scales"][s])
                   # print("UAAA")
                    
                    self.pw_layout["scales"]+= [{"arc":arc(params=unit_net_settings,verbose=verbose),
                                               "fin":pw_layout['scales'][s]["fin"],
                                               "fout":pw_layout['scales'][s]["fout"],
                                               }]
                
           # print(len(self.pw_layout["scales"]))
            #print(s)
                
            if ("copy" in pw_layout) and (s in pw_layout["copy"]):
                print(pw_layout["copy"])                
                st = pw_layout["copy"][s]
                unit = self.units[st]
            else:
                #print("#####################")
                #print(self.pw_layout["scales"][s][1])
               # print(self.pw_layout["scales"][s] )
                
                unit =  model_factory(self.pw_layout["scales"][s]["arc"],verbose=verbose)
            
                
                

            
            if False:           
                arc = eval(lo["arc"])
                if inspect.isclass(arc):
                    unit_net_settings = lo['params']
                    unit_net_settings['name'] = "s"+str(s)
                    unit_net_settings['fin'] = fin
                    unit_net_settings['fout'] = lo['fout']
                    unit_net_settings['dim'] = pw_layout['dim']
                    unit_net_settings['pfield'] = pw_layout['pfield']
                    unit_net_settings['scale'] = [s,coarsest_layer-1]
                    if ("copy" in pw_layout) and (s in pw_layout["copy"]):
                        print(pw_layout["copy"])                
                        st = pw_layout["copy"][s]
                        unit = self.units[st]
                    else:
                        unit =  arc(params=unit_net_settings)
                else:
                    unit = arc

            if s == 0:
                self.n_labels = lo['fout'] if ("labels" not in pw_layout or pw_layout["labels"] is None) else pw_layout["labels"] 
        
            #if hasattr(self.units[s], 'scale_info'):
            #    self.pass_scale_info = True
            
            
            #mynet = torch.jit.trace(mynet_,dummy_input,
            #                        check_trace = False,
            #                        )
            
            if optimize_units:
                print("optimizing unit")
                dummy_input = ()
                #dummy_input_d = fin if type(fin) is list else (fin,)
              
                for cin in fin:
                    if cin>0:
                        tmp = torch.rand((2,cin,)+unit_net_settings['pfield'])
                        print("dummy shape {}\{}:{}".format(s,cin,tmp.shape))
                        dummy_input += (tmp,)
 
                #print(len(dummy_input))
                
                unit_ = torch.jit.trace(unit.forward,(dummy_input,),
                                        check_trace = False,
                                        )
                self.units.append(unit_)
                
            else:
                self.units.append(unit)
            
            
            self.unit_in += [lo['fin']]
            if s > 0:
                self.pw_layers["nprop"+str(s)] = norm_factory(self.dim,lo['fout'])
            else:
                self.pw_layers["nprop"+str(s)] = torch.nn.Identity()
                
                
        if self.checkpoint != None:
            self.load_state_dict(self.checkpoint['net'])
            
        pw_debug("init patchwork done")   
    def model_save(self,fname,userdict={}):
        if self.train_state.iter >0:
            
            state = {
                    "pw_layout":self.pw_layout,
                    'net': self.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "train_state": self.train_state,
                    #"best_eval_accuracy":best_eval_accuracy,
                    #"best_eval_loss":best_eval_loss,
                    #"running_eval_accuracy":running_eval_accuracy,
                    #"running_eval_loss":running_eval_loss,    
                    #"pfield":self.pfield,
                    }
            
            if hasattr(self,"scheduler") is not None:
                state["scheduler"]=self.scheduler.state_dict()
                
            state = {**state, **userdict}
            
            torch.save(state, fname)
            
    def train_epoch(self,
              # sampler
              augmentation = {},
              intensity_augmentation = None,
              sample_from_label = 1,
              sampling = "n_coarse2fine",
              aug_device = "cpu",
              patchbuffersize = 1000,
              weights = None,#{'bg':n_labels,'classes':[1.0]},
              snap = "valid",
              scale_facts = None,
              sampler_debug_scales = 0,
              target_element_size = None,
              progress_callback = None,
              # training
              pw_loss = {"loss":patchwork_loss,
                         "params":{
                                "criterion":None,
                                "activation":lambda x,ind=None: nn.Sigmoid(x),
                                "prior" :None,
                                 "prefix" : "train/",
                                 "ignore_output_channels":[],
                                 "label_is_weight":None,
                                  "task":"segmentation",
                             }},
              #activation = nn.Sigmoid(),
              batch_size = 20,
              resample = 10,
              train_device = "cpu",
              optimizer = {"optim":optim.Adam,"params":{"lr":0.001, "eps":1e-07}},#(mynet.parameters(), lr=0.001, eps=1e-07)
              scheduler = None,
              #prior = None,
              # dataset
              dataset_train = None,
              dataset_eval = None,
              # other
              model_path = None,
              writer = None,
              clear = False,
              #ignore_output_channels = [],
              #label_is_weight = None,
              #task="segmentation",
              patch_update_rate=1,
              do_resampling = True,
              max_depth = -1,
              generate_batch_mask = None):

            pw_debug("entering patchwork training function (train_epoch)")   
            
            
            #keep last used settings for inference
            self.train_state.snap = snap
            self.train_state.scale_facts = scale_facts
            self.train_state.target_element_size = target_element_size
            self.train_state.augmentation = augmentation

            
            if self.train_state.iter == 0 or clear:
                 pw_debug("clearing state")   
                 self.train_state.iter = 0
                 self.train_state.epoch = 0
                 self.train_state.iter_after_resample = 0
                 self.train_state.running_smoothing = 0.99
                 
                 self.train_state.stats = types.SimpleNamespace()#tls.cnt()
                 self.train_state.stats.best_eval_accuracy = 0
                 self.train_state.stats.best_eval_loss = 1000000000000000
                 self.train_state.stats.running_eval_accuracy = 0
                 self.train_state.stats.running_eval_loss = 0
                 self.train_state.stats.resample = []
                 
                 
                 
                 self.optimizer = optimizer["optim"](self.parameters(),**optimizer["params"])
                 if scheduler is not None:
                     self.scheduler = scheduler["scheduler"](self.optimizer,**scheduler["params"])
            
                 
            pw_debug("moving network to device:"+train_device)   
            self.to(device=train_device)
            #network freshly loaded from file     
            #need to restore states
            if self.checkpoint  != None:
                pw_debug("restoring state from checkpoint")   
                self.optimizer = optimizer["optim"](self.parameters(),**optimizer["params"])
                self.optimizer.load_state_dict(self.checkpoint['optimizer'])
                if scheduler is not None:
                    self.scheduler.load_state_dict(self.checkpoint['scheduler'])
                
            
            #create training set in parallel
            #self.branches = nn.Parallel(block1, block2)
            #x1, x2 = self.branches(x, x)
            pw_debug("using data container :{}".format(dataset_train.name))     
            if do_resampling and (self.train_state.epoch % resample == 0 or not dataset_train.ready):# and dataset_eval is not None:
                pw_debug("resampling data (epoch: {}, ready: {})".format(self.train_state.epoch,dataset_train.ready))
                #clear_output(wait=True)
                self.train_state.iter_after_resample = 0
                
                self.train_state.stats.resample += [self.train_state.epoch,self.train_state.iter]

                start_t = time.time()
                #if writer is not None:
                #    writer.add_scalar("resmaple",1,self.train_state.iter)
                #skip_resample_info = self.train_state.iter
                dataset_train.resample(
                            aug_device=aug_device,
                            sampling=sampling,
                            target_element_size=target_element_size,
                            patchbuffersize=patchbuffersize,
                            scale_facts = scale_facts,
                            augmentation=augmentation,
                            progress_callback=progress_callback,
                            sample_from_label=sample_from_label,
                            weights=weights,
                            snap=snap,
                            sampler_debug_scales=sampler_debug_scales,
                            patch_update_rate=patch_update_rate,
                            pfield=self.pw_layout["pfield"],
                            max_depth=max_depth)  
                #return
                
                end_t = time.time()
                pw_debug("done resampling. It took {} seconds".format(end_t - start_t))   
                print("done")
                print("total time : {}".format(end_t - start_t))
                dps = patchbuffersize/(end_t - start_t)
                print("datasets per second : {}".format(dps))
                pw_debug("with datasets per second] {}".format(dps))
            
            else:
                assert(dataset_train.ready)

            pw_debug("getting training batch indeces")      
            batch_train = dataset_train.batch(batch_size,device=train_device)
            
            if dataset_eval is not None:
                batch_eval = dataset_eval.batch(batch_size,device=train_device)
                batch_indeces = zip(batch_train,batch_eval)
            else:
                batch_indeces = zip(batch_train,range(len(batch_train)))
            
            self.batch_indeces = batch_indeces
            #for i,indx in zip(range(len(batch)),batch):
            batch_mask = None
            pw_debug("entering training loop with {} batches".format(len(batch_train)))
            for indx_train,indx_eval in batch_indeces:
                    pw_debug("performing iteration {} ({})".format(self.train_state.iter,dataset_train.name))
                    if (writer is not None):# and (skip_resample_info!=self.train_state.iter):
                        writer.add_scalar("resmaple",self.train_state.iter_after_resample,self.train_state.iter)
                        
                    #skip_resample_info = -1
                
                    self.zero_grad()
                    
                    patch_imgs, patch_labels = dataset_train.get_batch(indx_train,mask=batch_mask)
                    
                    if intensity_augmentation is not None:
                        pw_debug("intensity augmentation")
                        intensity_augmentation["aug_fun"](patch_imgs.patchlist, self.dim,intensity_augmentation["aug_params"])
                        #tls.aug_patchbatch(patch_imgs.patchlist, self.dim,intensity_augmentation)
                    pw_debug("prediction")
                    
                    prediction = self(patch_imgs,
                                      set_out_scale=(0 if max_depth == -1 else (scale_facts.shape[1]-max_depth))
                                      )
                    
                    #loss, accuracy = patchwork_loss(  prediction,
                    #                                   patch_labels,
                    #                                   criterion,
                    #                                   activation,
                    #                                   prior = prior,
                    #                                   writer=None if writer is None else (writer,self.train_state.iter),
                    #                                   prefix = "train/",
                    #                                   ignore_output_channels=ignore_output_channels,
                    #                                   label_is_weight=label_is_weight,
                    #                                   task=task,
                    #                                  )
                    
                    #print("bla")
                    #print(prediction)
                    #print("blu")
                    #print(pw_loss["params"])
                    pw_debug("loss")
                    loss, accuracy,batch_mask = pw_loss["loss"](  prediction,
                                             patch_labels,
                                             patch_imgs,
                                             writer=None if writer is None else (writer,self.train_state.iter),
                                             n_scales = scale_facts.shape[1],
                                             **pw_loss["params"],
                                             start_scale=0 if max_depth == -1 else (scale_facts.shape[1]-max_depth),
                                             #criterion,
                                             #activation,
                                             #prior = prior,
                                             #prefix = "train/",
                                             #ignore_output_channels=ignore_output_channels,
                                             #label_is_weight=label_is_weight,
                                             #task=task,
                                            )
                    pw_debug("gradient updates")
                   # print("backward")
                    
                    
                    if False:
                        def plot_grad_flow(named_parameters):
                                ave_grads = []
                                layers = []
                                for n, p in named_parameters:
                                    if (p.requires_grad) and ("OUT" in n):
                                    #if(p.requires_grad) and ("bias" not in n):
                                        print("#############################")
                                        print(n)
                                        print(p.grad)
                                        layers.append(n)
                                        ave_grads.append(p.grad.abs().mean())
                                        print(p.shape)
                                        print(p)
                                        print(p.abs().mean())
                                        print(p.grad.abs().mean())
                           
                    
                    loss.backward()
                    
                    #plot_grad_flow(self.named_parameters())
                    #assert(False)
                    
                    
                    self.optimizer.step()
                    self.train_state.iter += 1
                    self.train_state.iter_after_resample += 1
                    
                    if (self.train_state.iter+1) % 50 == 0:
                        if dataset_eval is not None:
                            with torch.no_grad():
                                self.eval()

                                patch_imgs_eval, patch_labels_eval = dataset_eval.get_batch(indx_eval)
                                prediction_eval = self(patch_imgs_eval)
                                
                                eval_loss,eval_accuracy = patchwork_loss(
                                       prediction_eval,
                                       patch_labels_eval,                                       
                                       criterion,
                                       activation,
                                       writer=None if writer is None else (writer,self.train_state.iter),
                                       prefix = "eval/"
                                      )
                                
                                if self.train_state.stats.running_eval_accuracy == -1:
                                   self.train_state.stats.running_eval_accuracy = eval_accuracy
                                   self.train_state.stats.running_eval_loss = eval_loss
                                else:                    
                                    alpha = self.train_state.stats.running_smoothing
                                    self.train_state.stats.running_eval_accuracy = alpha * self.train_state.stats.running_eval_accuracy + (1-alpha) * eval_accuracy 
                                    self.train_state.stats.running_eval_loss = alpha * self.train_state.stats.running_eval_loss + (1-alpha) * eval_loss 
                            
                                if model_path is not None:
                                    if self.train_state.running_eval_accuracy>self.train_state.best_eval_accuracy or self.train_state.running_eval_loss<self.train_state.best_eval_loss:
                                        print("updating best choice model")
                                        print(self.train_state.stats)
                                        fname = model_path + ("_best_eval_accuracy.tar" if self.train_state.running_eval_accuracy>self.train_state.best_eval_accuracy else "_best_eval_loss.tar") 
                                        self.model_save(fname)
                                self.train()
                        
            if scheduler is not None:
                pw_debug("updating scheduler")
                self.scheduler.step()  
                if (writer is not None):
                    pw_debug("updating tensorboard log")
                    #writer.add_scalar("lr",self.scheduler.get_lr()[0],self.train_state.iter)
                    writer.add_scalar("lr",self.scheduler.get_last_lr()[0],self.train_state.iter)
                    
                    
            pw_debug("epoch done")    
            self.train_state.epoch += 1
            
            
                 
            
        
    def get_dim(self):
        return self.dim
    
    def get_offset(self):
         return self.units[0].get_offset()
     
    def reset_parameters(self):
        for l in self.units:
            #print("unit {} {}".format(l,hasattr(l,'reset_parameters')))
            if hasattr(l,'reset_parameters'):
                l.reset_parameters()
                
    def rep2same(self,t1,t2):
        d = t1.shape[1]*t2.shape[1]
        repetitions =  np.ones(2+self.dim,dtype=np.int32)
        repetitions[1] = d // t1.shape[1]
        t1 = t1.repeat(tuple(repetitions)) 
        repetitions[1] = d // t2.shape[1]
        t2 = t2.repeat(tuple(repetitions))
        return t1, t2  
    
    
    def step(self,
             patch_batch,
             s,
             coarsest_level,
             outputs,
             padding_mode):
        
            #print("ueee")
           # pw_debug("entering step")   
        
            device = self.get_device()
            if str(s) in self.preprocess_nets:
                assert(not self.normalization == "std_mean")
                assert(not self.normalization == "std_mean_separately")
                assert(not self.normalization == "/mean")
                t = self.normalize_inputs(self.preprocess_nets[str(s)](patch_batch.patchlist[s].tensor.to(device)))
                #t = self.normalize_inputs(self.preprocess_nets[str(s)](patch_batch.patchlist[s].tensor))
            else:
                t = self.normalize_inputs(patch_batch.patchlist[s].tensor.to(device))
                #t = self.normalize_inputs(patch_batch.patchlist[s].tensor)
            
            if s == coarsest_level and self.normalization == "std_mean":
                n_voxels = torch.tensor(t.shape[2:]).prod()
                std_,m_ = torch.std_mean(t.reshape([t.shape[0],t.shape[1],n_voxels]),dim=2,keepdim=True)
                std_ = std_[:,:,None] if self.dim == 2 else std_[:,:,None,None]
                gamma = 0.001
                std_ += gamma
                m_ = m_[:,:,None] if self.dim == 2 else m_[:,:,None,None]
                self.std_ = std_
                self.m_ = m_
                
            if s == coarsest_level and self.normalization == "/mean":
                n_voxels = torch.tensor(t.shape[2:]).prod()
                m_ = torch.mean(t.reshape([t.shape[0],t.shape[1],n_voxels]),dim=2,keepdim=True)
                gamma = 0.0000001
                m_ += gamma
                m_ = m_[:,:,None] if self.dim == 2 else m_[:,:,None,None]
                self.m_ = m_
                #print(self.m_.shape)
                
            if self.normalization == "/mean":
                n_voxels = torch.tensor(t.shape[2:]).prod()
                t = t/self.m_
            
            if self.normalization == "std_mean_separately":
                n_voxels = torch.tensor(t.shape[2:]).prod()
                std_,m_ = torch.std_mean(t.reshape([t.shape[0],t.shape[1],n_voxels]),dim=2,keepdim=True)
                std_ = std_[:,:,None] if self.dim == 2 else std_[:,:,None,None]
                gamma = 0.001
                std_ += gamma
                m_ = m_[:,:,None] if self.dim == 2 else m_[:,:,None,None]
                t = (t-m_)/std_
            
            if self.normalization == "std_mean":
                n_voxels = torch.tensor(t.shape[2:]).prod()
                t = (t-self.m_)/self.std_
                # std_t,m_t = torch.std_mean(t.reshape([t.shape[0],t.shape[1],n_voxels]),dim=2,keepdim=True)
                # print(m_t.shape)
                # print(m_t[:,0,0])
                # print(std_t[:,0,0])
            
            T = None
            ins = ()
            ins_info = ()
            for f in self.unit_in[s]:
                if (f == s):
                        T = t
                        #ins += (t,)
                        ins_info += (0,)*t.shape[1]
                        if "input_stats" in self.debug:
                            s_std,s_mean = torch.std_mean(t)
                            print("image input: m: {} v: {}".format(s_mean,s_std))
                            #mean_stat = t[0,:,...].mean(dims=())
                else:
                    assert(f>s)                                                
                    ins_info += (s-f,)*outputs[str(f)].shape[1]
                    if True:
                        for f2 in range(f,s,-1):
                            tmp2 = torch.nn.functional.grid_sample( 
                                self.pw_layers["nprop"+str(f)](outputs[str(f)]) if f2 == f else tmp2,
                                patch_batch.patchlist[f2-1].grid.to(device=outputs[str(f2)].device), 
                                mode=self.grid_crop, 
                                padding_mode=padding_mode, 
                                align_corners=True)
                    else:
                        # I was thinking that this one might be better, but it is not working.
                        # Must think about it
                        for f2 in range(f,s,-1):
                            tmp2 = torch.nn.functional.grid_sample(
                                outputs[str(f)] if f2 == f else tmp2,
                                patch_batch.patchlist[f2-1].grid.to(device=outputs[str(f2)].device),
                                mode=self.grid_crop,
                                padding_mode=padding_mode,
                                align_corners=True)
                        tmp2 = self.pw_layers["nprop"+str(f)](tmp2)


                    if "input_stats" in self.debug:
                            s_std,s_mean = torch.std_mean(tmp2)
                            print("prop data : m: {} v: {}".format(s_mean,s_std))
                            
                    ins += (tmp2,)
            if len(ins)>0:
                if T is not None:
                    x =  (T,torch.cat(ins,dim=1))
                else:
                    x = (torch.cat(ins,dim=1),)
            else:
                x = (T,)
                    
            if "keep_inputs" in self.debug:
                self.debug_data["input"+str(s)] = x
                
            
            
            #y = self.units[s](x)
           # print(y.amax(dim=(1,2,3,4)))
           # return y
            
            #ins info not used yet
            #print(type(x))
            #print(len(x))
            
            return self.units[s](x)
            #return self.units[s](x)
            #outputs[str(s)] = self.units[s](x)
    
    def forward(self,
                patch_batch,
                padding_mode=pw_settings.default_patch_forwarding_padding_mode,#patchwriter.default_img_padding_mode,
                set_out_scale=0,debug={}):
                #set_out_scale=0,debug=[]):
        
        self.std_ = None
        self.m_ = None

        self.debug = debug
        if "keep_inputs" in self.debug and  not hasattr(self, 'debug_data'):
            self.debug_data = {}
            
        outputs = {}
        coarsest_level = len(patch_batch.patchlist)-1 #if max_depth == -1 else max_depth
       # std_=None
      #  m_=None
        for s in range(coarsest_level,set_out_scale-1,-1):
            
                outputs[str(s)] = self.step(
                     patch_batch,
                     s,
                     coarsest_level,
                     outputs,
                     padding_mode,
                     #std_=std_,
                     #m_=m_
                     )
        
        return outputs
    
    
    
    def heuristic_patching_saliency(self,
                            s,
                            base_scale_indx,
                            best_sample,
                            saliency_map,
                            best_split,
                            num_samples,
                            activation_func,
                            saliency_clamp_min,
                            outputs,
                            sample_labels,
                            coarsest_level,
                            patch_batch,
                            state,
                           # states,
                            batch_id,
                            pw_img,
                            valid_mask
                            ):
        if s!=base_scale_indx:
            if best_sample == "uniform": 
                pass
            else:
                if saliency_map is None:
                    #if False:
                    #if best_sample == "all" or best_sample == "one_out_of_all" or best_sample == "valid_with_score":
                    if best_sample == "sort_score" or best_sample == "sort_valid": # or best_sample == "valid_with_score":
                        assert(num_samples%best_split==0)
                        sample_from_patch = torch.clamp(activation_func(outputs[str(s)][:,:sample_labels,...],range(sample_labels)),min=saliency_clamp_min)
                        
                        org_shape = sample_from_patch.shape
                        view_shape_ = (org_shape[0],org_shape[1]) + (np.array(sample_from_patch.shape[2:]).prod(),)
                        sample_from_patch_ = sample_from_patch.view(view_shape_)
                        
                        
                        max_responses = sample_from_patch.amax(dim=(2,3,4) if self.dim==3 else (1,2))
                        n_valid_labels = max_responses.shape[1]
                        #print("max_responses : {}".format(max_responses))
                        best_scores = torch.zeros((num_samples*n_valid_labels,)).type(torch.LongTensor)           
                        
                        for l in range(n_valid_labels):
                            best_scores[l::n_valid_labels] = torch.argsort(max_responses[:,l],descending=True)
                         #   print("best_scores[l::n_valid_labels]  : {}".format(best_scores[l::n_valid_labels] ))
                        
                        if False:
                            unique_scores = torch.zeros((num_samples*n_valid_labels,)).type(torch.bool)
                            uniqueList = []
                            for el in range(num_samples*n_valid_labels):
                                if best_scores[el] not in uniqueList:
                                           unique_scores[el] = True
                                           uniqueList.append(best_scores[el])
                            best_scores = best_scores[unique_scores]

                        topselect = best_scores[:num_samples//best_split]
                        topselect = torch.cat((topselect,)*best_split,dim=0)
                        
                        
                        if hasattr(self,"force_shuffle") and self.force_shuffle:
                            print("topselect force_shuffle")
                            #topselect = torch.randperm(num_samples)
                            #topselect = torch.arange(num_samples-1,-1,-1)
                            topselect[:] = best_scores[len(best_scores)//2]
                            #topselect = torch.arange(0,num_samples)
                        
                        # in order to select the best candidates 
                        # we need to update
                        #   - all previous outputs
                        #   - patch batch (up to current scale, including meta data)
                        #   - state bounding boxes
                        #   - sample patches                                                
                        
                        if True:
                            for s_ in range(s,coarsest_level):
                                outputs[str(s_)] = outputs[str(s_)][topselect,...]
                             
                            patch_batch.copy_to(patch_batch,topselect,range(s-1,coarsest_level),copy_meta=True,copy_trafos=True)    
                        else:
                            #for s_ in range(0,coarsest_level):
                            for key in outputs:
                                outputs[key] = outputs[key][topselect,...]
                            
                            patch_batch.copy_to(patch_batch,topselect,range(s-1,coarsest_level),copy_meta=True,copy_trafos=True)    
                            #patch_batch.copy_to(patch_batch,topselect,range(0,coarsest_level),copy_meta=True,copy_trafos=True)    
                            
                        if False:
                            for key in states:
                                if states[key] is not None:
                                    states[key]["bb_0"] = states[key]["bb_0"][topselect,:]
                                    states[key]["bb_1"] = states[key]["bb_1"][topselect,:]
                            state["bb_0"] = state["bb_0"][topselect,:]
                            state["bb_1"] = state["bb_1"][topselect,:]
                            
                        if True:
                            for i in range(len(state)):
                                if state[i] is not None:
                                    state[i]["bb_1"] = state[i]["bb_1"][topselect,:]
                                    state[i]["bb_0"] = state[i]["bb_0"][topselect,:]
                        
                        if False:
                            state["bb_0"] = state["bb_0"][topselect,:]
                            state["bb_1"] = state["bb_1"][topselect,:]
                        
                        if self.std_ is not None:
                           # print("updating std")
                            self.std_ = self.std_[topselect,...]
                            self.m_ = self.m_[topselect,...]
                        
                        
                        
                        sample_from_patch = activation_func(outputs[str(s)][:,:sample_labels,...],range(sample_labels))#.clone()
                    
                #    else:
                    elif best_sample == "score":
                        sample_from_patch = activation_func(outputs[str(s)][:,:sample_labels,...],range(sample_labels))#.clone()
                    elif best_sample == "valid":
                        pass
                    else:
                        assert(False)
                
                #print(sample_from_patch.amax(dim=(1,2,3,4)))
                #for b in range(sample_from_patch.shape[0]):
                #    bla2 = sample_from_patch[b,0,:,:,:].amax(dim=2)
                #    plt.imshow(bla2)
                #    plt.pause(0.1)
                            
                grid = patch_batch.patchlist[s].rel_coordinates[batch_id:batch_id+num_samples,...]
                in_shape = (1,)+(grid.shape[1]*num_samples,) + tuple(grid.shape[2:-1])+(pw_img.dim,)
                out_shape = (num_samples,1,)+ tuple(grid.shape[1:-1])
                valid = torch.nn.functional.grid_sample(valid_mask, 
                                                grid[...,:pw_img.dim].reshape(in_shape), 
                                                mode="nearest",#'bilinear',#"nearest", 
                                                padding_mode='zeros', 
                                                align_corners=True).reshape(out_shape).to(device=outputs[str(s)].device)
                
                
                if saliency_map is not None:
                   # print("saliency map {}".format(saliency_map.shape))
                    sample_from_patch = torch.nn.functional.grid_sample(saliency_map, 
                                                grid[...,:pw_img.dim].reshape(in_shape), 
                                                mode="nearest",#'bilinear',#"nearest", 
                                                padding_mode='zeros', 
                                                align_corners=True).reshape(out_shape).to(device=outputs[str(s)].device)
                    
                if best_sample == "valid" or best_sample == "sort_valid":
                    sample_from_patch = valid
                else:
                    sample_from_patch += valid
        
                max_saliency = sample_from_patch.reshape([sample_from_patch.shape[0],-1]).amax(dim=(1,)) < 0.000000001
                sample_from_patch[max_saliency,...] += 0.000000001
              #  print("max_saliency  ",max_saliency.sum())
        
                return sample_from_patch
        return None
    
    def heuristic_patching_step(self,
                                i,
                                states,#states,
                                hp_patch_step,
                                hp_saliceny,
                                hp_copy,
                                hp_apply,
                                profile,
                                #total_patching_time,
                                #total_network_time,
                                #total_scatter_time,
                                outputs,
                                callback,
                                progress,
                                progress_max,
                                scale_facts,
                                shape_out_vox,
                                result,
                                result_counts,
                                n_labels,
                                target_device,
                                patch_batch,
                                batch_id,
                                num_samples,
                                activation_func,
                                classify,
                                ):
        if profile:
            timer = time.time()
            
        
        
        
        #get the current image patches
        if states[-1] is not None:
            sample_from_patch = hp_saliceny(states[-1]["ii"],states)
            #pmax = sample_from_patch.amax(dim=(1,2,3,4))
            #if (pmax==0).any():
            #    print("funck")
            #print("sample_from_patch ",sample_from_patch.shape)
            
            #print("min : ",sample_from_patch.amin()," max ",sample_from_patch.amax())
        #if state is not None:
        #    sample_from_patch = hp_saliceny(state["ii"],state)
            #sample_from_patch = state["sample_from_patch"] if "sample_from_patch" in state else None#sample_from_patch, #fix
        else:
            sample_from_patch = None
            
        state = copy.deepcopy(states[-1])
        #sample_from_patch = None
        state = hp_patch_step(state=state,i=i,sample_from_patch=sample_from_patch)
        
        #d2 = copy.deepcopy(state)
        #print(d2)

         #apply the network
       
        s = state['ii'] # scale index equals inverse index
        
        #print("Scale s: {}".format(s))
        hp_copy(s)    
        if self.intensity_augmentation is not None:
           # print("intensity_augmentation")
            #tls.aug_patchbatch([patch_batch.patchlist[s]], self.dim,self.intensity_augmentation)
            self.intensity_augmentation["aug_fun"]([patch_batch.patchlist[s]], self.dim,self.intensity_augmentation["aug_params"])
            
        
        if profile:
            self.profiling["total_patching_time"] += (time.time() - timer)
           # print("total_patching_time ",total_patching_time)
            timer = time.time()
        
        outputs[str(s)] = hp_apply(s)
        
        if profile:                    
            self.profiling["total_network_time"] += (time.time() - timer)
        
        #create_sample_batch()
        #sample_from_patch = hp_saliceny(s,state)
        #state["sample_from_patch"] = hp_saliceny(s,state)
        #state["sample_from_patch"] = hp_saliceny(s)

                    
        if callback is not None:
            progress += 1
            callback(progress,progress_max)
   
        scale_fact = scale_facts[:,s]#1.5**torch.tensor([ii,ii,ii])                        
        shape_out_vox_ = (shape_out_vox / scale_fact).floor().int()#tls.tt(target_element_size_mu,dtype=torch.float32)       
        shape_out_vox_ = torch.clamp(shape_out_vox_,min=1)
        
        if str(s) not in result:
            result[str(s)] = torch.zeros((1,n_labels,)+tuple(shape_out_vox_),
                                     device=target_device, 
                                     requires_grad=False)
                
            result_counts[str(s)] = torch.zeros((1,1,)+tuple(shape_out_vox_),
                                 device=target_device, 
                                 requires_grad=False)
            
        grid = patch_batch.patchlist[s].rel_coordinates[batch_id:batch_id+num_samples,...]
        predictions = activation_func(outputs[str(s)])
        
        #print("predictions {} {}".format(predictions.amin(),predictions.amax()))
        
        
        if classify is not None:
            predictions = (predictions > classify).type(predictions.dtype)
            
        
        #print("scale: {}".format(s))
        if s == 0 and hasattr(self, "pred_debug"):
            #if hasattr(self, "pred_debug"):
            #    self.pred_debug += [predictions[:,:n_labels,...]]
            #else:
            #self.pred_debug = [predictions[:,:n_labels,...]]
            self.pred_debug += [predictions.clone()]
            
        if True:
            self.scatterer.scatter_2_img_add(
                                        grid,
                                        predictions[:,:n_labels,...],
                                        result[str(s)],
                                        result_counts[str(s)],   
                                        batch_id = batch_id,
                                        batch_size=num_samples)
        else:
            #print("predictions : {}".format(predictions.shape))
            #print("grid : {}".format(grid.shape))
            invers_coords = predictions[:,:4,...].clone()
            #invers_coords = (-1.0+2.0*invers_coords/ torch.tensor([180, 210],device=grid.device).reshape([1,2,1,1]))
            
            invers_coords = torch.cat(
                (torch.atan2(invers_coords[:,1,...],invers_coords[:,0,...])[None,...],
                 torch.atan2(invers_coords[:,3,...],invers_coords[:,2,...])[None,...]),dim=0)
            invers_coords = invers_coords/(2*math.pi) + 0.5
            invers_coords = (-1.0+2.0*invers_coords)#.repeat([3,1,1,1])#.reshape([1,2,1,1])#*inv_scaling.reshape([1,2,1,1])
            #invers_coords[2:,...] = 0
            #print("{}".format(invers_coords.shape))
            #invers_coords = invers_coords.permute(dims=[1,2,0])[None,...,[1,0]]
            
        #invers_coords = predictions[:,:2,...].clone()-1
            
            
            invers_coords = invers_coords.permute(dims=[1,2,3,0])
            inverse_grid = torch.zeros([predictions.shape[0],6,predictions.shape[2],predictions.shape[3]]).to(grid.device)
            inverse_grid[:,:2,...] = grid.permute(dims=[0,3,1,2])
            
            invers_coords = invers_coords.contiguous()[...,[1,0]]
            inverse_grid = inverse_grid.contiguous()
            
            #print("predictions 2 : {}".format(inverse_grid.shape))
            #print("grid 2 : {}".format(invers_coords.shape))
            
            self.scatterer.scatter_2_img_add(
                                        invers_coords,
                                        inverse_grid,
                                        result[str(s)],
                                        result_counts[str(s)],   
                                        batch_id = batch_id,
                                        batch_size=num_samples)
            
        if profile:                
            self.profiling["total_scatter_time"] += (time.time() - timer)
        return state
             
    #for i in range(base_scale_indx,coarsest_level):
    #def heuristic_patching_search(self,i,coarsest_level,split,hp_step,states,patch_batch,outputs,state=None):
    def heuristic_patching_search(self,i,coarsest_level,split,hp_step,states,patch_batch,outputs,full_backup_tree,state=[None]):
        if i<coarsest_level:
            #print(split)
            #print(i)
            #full_backup_tree = False
            full_backup_tree_ = full_backup_tree and (np.array(split)>1).any()
            if full_backup_tree_:
                patch_batch_backup = pw_data.patch_batch(
                     patch_batch.n_scales,
                     patch_batch.n_batches,
                     patch_batch.n_modalities,
                     patch_batch.pfield,
                     img_device="cpu",
                     patch_device="cpu",
                     aug_device="cpu",
                     )
                
                patch_batch.copy_to(patch_batch_backup,
                            None,
                            scale_indeces=None,
                            copy_data=True,
                            copy_meta=True,
                            copy_trafos=True)
                outputs_backup = {}
                for key in outputs:
                    outputs_backup[key] = outputs[key].clone()
                
            
            if full_backup_tree_:
                state_backup = copy.deepcopy(state)
            assert(split[i]>0)
            #if split[i]>1:
            #    print("not working properly for scale ",i)
            for a in range(split[i]):
             #   print("{} {}".format(i,a))
                #print("scale: {} subtree :{} states:{}".format(i,a,len(state)))
                if full_backup_tree_:
                    state_in = copy.deepcopy(state_backup)
                else:
                    state_in = state#copy.deepcopy(state)
               
                    
                if full_backup_tree_:
                    patch_batch_backup.copy_to(patch_batch,
                                None,
                                scale_indeces=None,
                                copy_data=True,
                                copy_meta=True,
                                copy_trafos=True)
                    for key in outputs_backup:
                    #    outputs = {}
                        outputs[key] = outputs_backup[key].clone()
                
                #if i == 2:
                #    for b in range(len(patch_batch.patchlist)):
                #        print("amax :",patch_batch.patchlist[b].tensor.amax())
                #    print("")

                #state_in = state
                #state_next = hp_step(i,state_in)
                #self.heuristic_patching_search(i+1,coarsest_level,split,hp_step,states,patch_batch,outputs,state_next)
                state_next = hp_step(i,state_in)
                self.heuristic_patching_search(i+1,coarsest_level,split,hp_step,states,patch_batch,outputs,full_backup_tree,state+[state_next])
            
        
    def heuristic_patching(self,
                pw_img,
                scale_facts=None,
                n_labels = None,
                sample_labels=None,                
                activation_func = lambda t,indx=None : torch.nn.Sigmoid()(t),   #torch.nn.Identity(),   # torch.nn.Sigmoid(),   #torch.nn.Identity(),   #torch.nn.Sigmoid(),   
                set_out_scale=0,
                augmentation=None,#{},
                intensity_augmentation =None,
                #img_device=torch.device('cpu'),
                patch_device=torch.device('cpu'),
                aug_device=torch.device('cpu'),
                target_device=torch.device('cpu'),
                target_element_size=None,
                #outout_element_size=None,
                crop_bb = True,
                copy_metadata=False,
                weights = {'bg':1.0,'classes':[1]},
                pos_noise=0.5,                
                warn_if_ooim=False,
                num_samples = 50,
                best_split = 2,
                runs = 1, 
                snap = None,#"img",
                border_compensation = 0.5,
                debug=[],#["input_stats"],
                best_sample = "sort_score",#"sort_score","uniform","valid","score","sort_valid"
                saliency_epsilon = 0.001,
                saliency_clamp_min = 0.0,
                saliency_reduction = "max",
                patching_strategy = "n_coarse2fine",
                verbosity = 0,
                pfield = None,#(32,32,32),
                classify=None,
              #  use_predictions = [0],
                result = None,
                result_counts = None,
                saliency_map = None,
                callback = None,
               # padding_mode_img = patchwriter.default_img_padding_mode,
                verbose = False,
                split = None,
                full_backup_tree = True,
              #  padding_mode_label = "zeros",
                ):
        
        
        self.intensity_augmentation  = intensity_augmentation 
        assert(classify is None or type(classify)==float)
        
        if best_sample not in ["sort_score","sort_valid"]:
           full_backup_tree = False
           
        else:
            if full_backup_tree:
                print("creating patch backups during search (larger search space when using patch sorting)")
            else:
                print("optimizing all patches during search (less memory, more focussed)")
        
        #scale_facts = mynet.train_state.scale_facts,
        #augmentation=mynet.train_state.augmentation,
        #snap=mynet.train_state.snap,
        #pfield = mynet.pw_layout["pfield"],
        #outout_element_size = mynet.train_state.target_element_size

        #scale_facts = tls.set_default(scale_facts) mynet.train_state.scale_facts if scale_facts is 
        scale_facts = tls.set_default(scale_facts,self.train_state.scale_facts)
        augmentation = tls.set_default(augmentation,self.train_state.augmentation)
      #  if augmentation is not None and "aug_flipdims"  in augmentation:
       #     del augmentation["aug_flipdims"] 
            #print("")
        #print(augmentation)
        
        snap = tls.set_default(snap,self.train_state.snap)
        pfield = tls.set_default(pfield,self.pw_layout["pfield"])
        target_element_size_mu = tls.set_default(target_element_size,self.train_state.target_element_size)

        # print("###############################")
        # print("target_element_size: ",target_element_size)
        # print("self.train_state.target_element_size: ",self.train_state.target_element_size)
        # print("target_element_size_mu: ",target_element_size_mu)
        # print("###############################")

        #outout_element_size = tls.set_default(outout_element_size,self.train_state.target_element_size)
       # print(pfield)
        #print(self.pw_layout["pfield"])
        
        if result_counts is None:
            result = {}
            result_counts = {}
        
        
        self.eval()
        profile = False
        with torch.no_grad(): 
            if profile:
                #print("clear timer")
                self.profiling = {
                
                "total_patching_time":0,
                "total_network_time":0,#time.time()    
                "total_scatter_time":0,     
                "total_start_time":time.time(),
                }
            self.std_ = None
            self.m_ = None
            local_device = "cpu"
            self.debug = debug
            

            if n_labels is None:
                n_labels = self.n_labels         
            sample_labels = n_labels if sample_labels is None else sample_labels
            coarsest_level = self.depth#-1
            base_scale_indx = set_out_scale
            
            if verbose:
                print("sample labels : {}".format(sample_labels))
                print("n_labels : {}".format(n_labels))
            
            
            img_device_org = pw_img.img.device
            img_device = patch_device
            pw_img.img = pw_img.img.to(device=img_device)
                    
            #img_device = patch_device
            
            
            img_element_size_mu = tls.tt(pw_img.element_size,device=local_device)          
            #outout_element_size = tls.tt(outout_element_size,device=local_device) if outout_element_size is not None else img_element_size_mu
            
            img = pw_img.img
            shape_ = torch.tensor(img.shape,device=local_device)
            shape_vox = shape_[2:].float()
            shape_mu = tls.vox_pos2physical_pos(shape_vox, img_element_size_mu)
            
            #f = tls.tt(scale_facts[:,0] / scale_facts[:,base_scale_indx],device=img_device)
            
            #shape_out_vox = (shape_mu / outout_element_size).round().int()
            shape_out_vox = (shape_mu / target_element_size_mu).round().int()
            
            
            
            if crop_bb:
                valid_mask = torch.zeros(img.shape[2:],dtype=torch.float32,device=img_device,
                                         requires_grad=False)[None,None,...]
                bb = pw_img.bb
                if self.dim == 3:
                    valid_mask[...,bb[0][0]:bb[1][0],bb[0][1]:bb[1][1],bb[0][2]:bb[1][2]] = 1
                    #valid_mask[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1],bb[0][2]:bb[1][2]] = 1
                    
                if self.dim == 2:
                    valid_mask[...,bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]] = 1
            else:
                valid_mask = torch.ones(img.shape[2:],dtype=torch.float32,device=img_device,
                                        requires_grad=False)[None,None,...]
                    
            n_modalities = pw_img.img.shape[1]
            
            
            #pfield = (32,)*self.dim
            
            patch_batch = pw_data.patch_batch(
                 self.depth,
                 num_samples,
                 n_modalities,
                 pfield,
                 img_device=img_device,
                 patch_device=patch_device,
                 aug_device=aug_device,
                 )
            
            
    
            progress = 0
            progress_max = runs*(coarsest_level-base_scale_indx)
            
       
                    
            if split is None:
                split = [1 for a in range(base_scale_indx,coarsest_level)]
            elif type(split) == list:
                pass
            elif callable(split):
                split = [split(a) for a in range(base_scale_indx,coarsest_level)]
            else:
                split = [split for a in range(base_scale_indx,coarsest_level)]
            
            print("split: {}".format(split))
            for r in range(runs):
                if verbose:
                    print("run {} of {}".format(r,runs))
                #state = None           
                #sample_from_patch = None
                batch_id = 0
                position_mu_target = num_samples
                #patching_strategy = "n_coarse2fine"
                outputs = {}
                
                #states = []
                
                hp_saliceny = lambda s,state: self.heuristic_patching_saliency(
                #hp_saliceny = lambda s: self.heuristic_patching_saliency(
                                    s,
                                    base_scale_indx,#fix 
                                    best_sample,#fix 
                                    saliency_map,
                                    best_split,#fix 
                                    num_samples,#fix 
                                    activation_func,#fix 
                                    saliency_clamp_min,#fix 
                                    outputs,
                                    sample_labels,#fix 
                                    coarsest_level,#fix 
                                    patch_batch, 
                                    state,
                                    #states,
                                    batch_id,#fix 
                                    pw_img,#fix 
                                    valid_mask#fix 
                                    )
                
                hp_patch_step = lambda state,i,sample_from_patch=None:  patchwriter.patch_step(
                        pw_img = pw_img, #fix 
                        batch_list = patch_batch,
                        batch_id = batch_id, #fix 
                        positions_mu = position_mu_target, #fix 
                        scale_facts = scale_facts, #fix 
                        patching_strategy = patching_strategy, #fix 
                        #state = state,
                        state = state if i>base_scale_indx else None, #fix 
                        base_scale_indx = base_scale_indx, #fix 
                        sample_from_patch = sample_from_patch, #fix 
                        augmentation = augmentation, #fix 
                        aug_device=aug_device, #fix 
                        target_device=target_device,                      #fix                    
                        target_element_size_mu=target_element_size_mu,       #fix               
                        crop_bb=crop_bb,# #fix 
                        warn_if_ooim=warn_if_ooim, #fix 
                        border_compensation = border_compensation, #fix 
                        snap = snap, #fix 
                        weights=weights, #fix 
                        pos_noise=pos_noise, #fix 
                        sampling="inference", #fix 
                        saliency_epsilon = saliency_epsilon, #fix 
                        saliency_reduction = saliency_reduction, #fix 
                        verbosity=verbosity #fix  
                        )
                
                hp_copy = lambda s :  patch_batch.patchlist[s].copy_to_patch_item( 
                    pw_img, #fix 
                    np.arange(num_samples)+batch_id, #fix 
                    patch_item_target=patch_batch.patchlist[s], #fix 
                    interpolation="nearest", #fix 
                    copy_metadata=True, #fix 
                    #padding_mode=padding_mode_img, #fix 
                    padding_mode=pw_settings.default_img_padding_mode
                    )    
                
                hp_apply = lambda s : self.step(
                     patch_batch, #fix 
                     s,
                     coarsest_level-1, #fix 
                     outputs, 
                     #padding_mode=padding_mode_img, #fix 
                     padding_mode=pw_settings.default_patch_forwarding_padding_mode
                     )
                
                hp_step = lambda i,state: self.heuristic_patching_step(
                                        i,
                                        state,
                                        hp_patch_step, #fix 
                                        hp_saliceny, #fix 
                                        hp_copy, #fix 
                                        hp_apply, #fix 
                                        profile, #fix 
                                        #total_patching_time,
                                        #total_network_time,
                                        #total_scatter_time, 
                                        outputs, 
                                        callback, #fix 
                                        progress,
                                        progress_max, #fix 
                                        scale_facts, #fix 
                                        shape_out_vox, #fix 
                                        result,
                                        result_counts, 
                                        n_labels, #fix
                                        target_device, #fix 
                                        patch_batch, #fix 
                                        batch_id, #fix 
                                        num_samples, #fix
                                        activation_func, #fix
                                        classify, #fix
                                        )
               
               # std_=None
               # m_=None
               #for i in range(base_scale_indx,coarsest_level):
                    
                #    state = hp_step(i,state)
                #self.heuristic_patching_search(base_scale_indx,coarsest_level,split,hp_step,states,patch_batch,outputs)
                self.heuristic_patching_search(base_scale_indx,coarsest_level,split,hp_step,[None],patch_batch,outputs,full_backup_tree)
                    
                #return outputs
                
                

            pw_img.img = pw_img.img.to(device=img_device_org)
            

            if profile:            
                total_time =  (time.time() - self.profiling["total_start_time"])
                if profile:
                    print("\n")
                    total_n_patches = num_samples * runs
                    print("img_device : {} ".format(img_device))
                    print("target_device : {} ".format(target_device))
                    print("aug_device : {} ".format(aug_device))
                    print("patch_device : {} ".format(patch_device))
                    print("total_n_patches : {} (x {} scales)".format(total_n_patches,scale_facts.shape[1]))
                    print("total_patching_time {} / per patch {}".format(self.profiling["total_patching_time"],1.0*self.profiling["total_patching_time"]/total_n_patches))
                    print("total_scatter_time {} / per patch {}".format(self.profiling["total_scatter_time"],1.0*self.profiling["total_scatter_time"]/total_n_patches))
                    print("total_network_time {} / per patch {}".format(self.profiling["total_network_time"],1.0*self.profiling["total_network_time"]/total_n_patches))
                    print("total_time {} / per patch {}".format(total_time,1.0*total_time/total_n_patches))
            return result,result_counts,outputs
        

    def apply(self,
                pw_img,
                #reruns = 1,
                settings = {},
                scale_weight_base = 10,
                ):
        #saliency_map = None
        result = None
        result_counts = None
        #for rerun in range(reruns):
        result,result_counts,outputs = self.heuristic_patching(pw_img,
                                                             result = result,
                                                             result_counts = result_counts,
                                                             #saliency_map = saliency_map,
                                                             **settings,
                                                            )   
        
        self.outputs_ = outputs
        #b = len(result.keys())
        n_scales = len(result.keys())
        sk = list(result.keys())
        sk.sort()
        
        for indx,key in zip(range(n_scales),sk):
            if indx == 0:        
               shape_out_vox = tls.tt(pw_img.img.shape)
               shape_out_vox[0] = n_scales 
               shape_out_vox[1] = self.n_labels 
               results_all = torch.zeros(shape_out_vox.tolist())
               shape_out_vox[1] = 1 
               all_counts = torch.zeros(shape_out_vox.tolist())
               shape_out_vox[0] = 1

            
            tmp = result[key].cpu().clone()
      
            
            results_all[indx,...] = torch.nn.functional.interpolate(tmp, size=shape_out_vox[2:].tolist(), mode='bilinear' if self.dim == 2 else "trilinear")
            all_counts[indx,...] = torch.nn.functional.interpolate(result_counts[key].cpu(), size=shape_out_vox[2:].tolist(), mode='bilinear' if self.dim == 2 else "trilinear")


        s_weights = torch.tensor(n_scales-np.arange(self.depth)-1) + 1
        #print("weights: ",s_weights)
        s_weights = tls.tt(scale_weight_base)**s_weights
        s_weights = s_weights.reshape([self.depth,1,1,1] if self.dim==2 else [self.depth,1,1,1,1])
      #  print("weights: ",s_weights)
        
        
        counts_merged = (s_weights*all_counts).sum(dim=0)
        valid = counts_merged > 0
        
        
        results_merged = (s_weights*results_all).sum(dim=0)
        results_merged[:,valid[0,...]] /= counts_merged[:,valid[0,...]]

        return results_merged, counts_merged, results_all, all_counts, result

        
    def get_device(self):
        #print(self.parameters())
        try:
            return "cuda" if next(self.parameters()).is_cuda else "cpu"
        except:
            return  "cpu"