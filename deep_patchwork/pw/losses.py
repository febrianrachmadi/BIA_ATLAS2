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


class patchwork_bce(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.criterion_bce = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, logits,pred,label,scale):
        return self.criterion_bce(logits,label)





class patchwork_MSE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.criterion_mse = nn.MSELoss()

    def forward(self, logits,pred,label,scale):
        return self.criterion_mse(pred,label)


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super().__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])#.cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        #print(inputs.shape)
        #print(targets.shape)
        #print("amax : {}".format(targets.amax()))
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').view(-1)
        
        targets = targets.type(torch.long)
        if inputs.device != self.alpha.device:
            self.alpha = self.alpha.to(device=inputs.device)
       # print("test1: {}".format(self.alpha.device))
       # print("test2: {}".format(targets.device))
        
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        #print("{} {} {} {}".format(at.shape,pt.shape,self.gamma,BCE_loss.shape))
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
#https://amaarora.github.io/2020/06/29/FocalLoss.html    
    
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets,  smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss =  self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss    
#from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch

class DiceLoss_old(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,squared=False):
        super().__init__()
        self.squared = squared

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()    
        if self.squared:                            
            return - (2.*intersection + smooth)/((inputs**2.0).sum() + (targets**2.0).sum() + smooth)  
        else:
            return - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
# ... in the case of α=β=0.5 the Tversky index simplifies to 
# be the same as the Dice coefficient, which is also equal to the F1 score. 
# With α=β=1, Equation 2 produces Tanimoto coefficient, and setting α+β=1
# produces the set of Fβ scores. Larger βs weigh recall higher than 
# precision (by placing more emphasis on false negatives).
#
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky



def patchwork_loss( prediction, 
                    labels,
                    patch_imgs,
                    criterion,
                    activation,
                    scale = -1,
                    writer=None,
                    balance_classes=False,
                    prefix="",
                    return_details=False,
                    loss_weight_scale_fun=None,#lambda x : 1,
                    track_scales = -1,
                    prior=None,
                    ignore_output_channels = [],
                    task="segmentation",
                    label_is_weight = None):
    
  #  print("pred shape {}".format(labels.patchlist[0].tensor.shape))
   # print("labels shape {}".format(prediction["0"].shape))
    loss = 0
    correct = 0
    total = 0
    if return_details:
        losses = {}
        accuracies = {}
    for s in range(len(prediction)):
        if return_details:
            losses[s] = {}
            accuracies[s]= {}
        if (scale == -1 or s == scale):
            num_labels = labels.patchlist[s].tensor.shape[1]
            
            if label_is_weight is not None:
                weight_img = labels.patchlist[s].tensor[:,None,label_is_weight,...]
            for l in range(num_labels):
                if l not in ignore_output_channels:
                    #print("labels shape {}".format(l))
                    if task == "segmentation":
                        label = torch.clamp(labels.patchlist[s].tensor[:,None,l,...],min=0)
                    else:
                        label = labels.patchlist[s].tensor[:,None,l,...]
                        
                    logits = prediction[str(s)][:,None,l,...]
                    pred = activation(logits,l)
                    
                    label = tls.crop2shape(label, pred.shape)
                    
                    if task == "segmentation":
                        label_05 = (label>0.5) 
                        pred_05 = (pred>0.5)
                        correct_ = (label_05 == pred_05).sum()
                        total_ = label.numel()
                        
                        accuracy = 100 * correct_.float().div(total_)
                        
                        correct += correct_
                        total += total_
                        
                        if balance_classes:
                            total_num = torch.tensor(label.shape).prod()
                            n_fg = label.sum()
                            n_bg = total_num - n_fg
                            w_fg = n_bg / (total_num)
                            w_bg = n_fg / (total_num)
                            #w = w_fg + w_bg 
                            
                            weight = (label < 0.5) * w_bg + (label > 0.5) * w_fg 
                            weight *= total_num/weight.sum()
                        else:
                            weight = None  
                    
                    else:
                        weight = None  
                        
                        
                    #if label_is_weight is not None:
                    #    criterion_ = criterion(label_is_weight=label_is_weight)
                    #else:
                    criterion_ = criterion(weight=weight) if  type(criterion)!=list else criterion[s](weight=weight)
                        
                    if label_is_weight is not None:
                        loss_ = criterion_(logits,pred,label,s,weight_img,l)
                    else:
                        loss_ = criterion_(logits,pred,label,s)
                        
                    if prior is not None:
                        loss_ += prior(pred)
    
                    
                    if writer is not None and track_scales in [s,-1]:
                        writer[0].add_scalar(prefix+'Loss/train_'+str(s)+"_"+str(l), loss_.item(), writer[1])
    
                        if task == "segmentation":
                            writer[0].add_scalar(prefix+'accuracy/train_'+str(s)+"_"+str(l),accuracy , writer[1])
        
                            label_05_sum = label_05.sum().float()
                            TP_ = (label_05*pred_05).sum().float()
                            TP_rate = (TP_+0.0000001)/(label_05_sum+0.0000001)
                            FP_ = (torch.logical_not(label_05)*pred_05).sum().float()
                            FP_rate = (FP_+0.0000001)/((torch.logical_not(label_05)).sum().float()+0.0000001)
                            FN_ = (label_05_sum - TP_).float()
                            FN_rate = (FN_+0.0000001) /  (label_05_sum+0.0000001)
                            
                            #print("TP: {} {}".format((label_05*pred_05).sum(),(label_05).sum()))
                            #print("FP: {} {}".format((torch.logical_not(label_05)*pred_05).sum(),torch.logical_not(label_05).sum()))
                            F1_ = (TP_+0.5*(FN_+FP_))
                            F1 = (TP_ / F1_) if F1_ > 0 else 0 
                            
                            writer[0].add_scalar(prefix+'stats/F1_'+str(s)+"_"+str(l),F1 , writer[1])
                            writer[0].add_scalar(prefix+'stats/TP_'+str(s)+"_"+str(l),TP_rate , writer[1])
                            writer[0].add_scalar(prefix+'stats/FP_'+str(s)+"_"+str(l),FP_rate , writer[1])
                            writer[0].add_scalar(prefix+'stats/FN_'+str(s)+"_"+str(l),FN_rate , writer[1])
    
                    if return_details:
                        losses[s][l] = loss_
                        if task == "segmentation":
                            accuracies[s][l] = accuracy
                    
                    if loss_weight_scale_fun is not None:
                        loss += loss_  * loss_weight_scale_fun(s) #(2**(loss_weight_scale_fact*s))
                    else:
                        loss += loss_  
                
    if task == "segmentation":                
        accuracy = 100 * correct.float().div(total)   
    else:
        accuracy  = None
        
    if writer is not None:
            writer[0].add_scalar(prefix+'Loss/train_all', loss.item(), writer[1])
            if task == "segmentation":
                writer[0].add_scalar(prefix+'accuracy/train_all', accuracy, writer[1])
    if return_details:
        return loss,accuracy,losses,accuracies
    return loss,accuracy    
    
def multiscale_loss(prediction, 
                    labels,
                    criterion,
                    activation,
                    scale = -1,
                    writer=None,
                    balance_classes=False,
                    prefix="",
                    return_details=False,
                    fgbg_pred=False,
                    loss_weight_scale_fact=0,
                    loss_with_logits=True,
                    track_scales = 0,
                    clamp_label = 0):
    loss = 0
    correct = 0
    total = 0
    if return_details:
        losses = {}
        accuracies = {}
    for s in range(len(prediction)):
        if return_details:
            losses[s] = {}
            accuracies[s]= {}
        if scale == -1 or s == scale:
            num_labels = labels.patchlist[s].tensor.shape[1]
            for l in range(num_labels):
                #label = labels.patchlist[s].tensor[None,:,l,::]
                #pred = activation(prediction[str(s)][None,:,l,::])
                
                if clamp_label is not None:
                    label = torch.clamp(labels.patchlist[s].tensor[:,None,l,...],min=clamp_label)
                else:
                    label = labels.patchlist[s].tensor[:,None,l,...]
                    
                logits = prediction[str(s)][:,None,l,...]
                pred = activation(logits,l)
                
                #print("label : {}".format(label.shape))
                #print("pred : {}".format(pred.shape))
                label = tls.crop2shape(label, pred.shape)
                label_05 = (label>0.5) 
                pred_05 = (pred>0.5)
                correct_ = (label_05 == pred_05).sum()
                total_ = label.numel()
                #print("correct_ : {}".format(correct_))
                #print("total_ : {}".format(total_))
                #print("(pred>0.5) : {}".format((pred>0.5).sum()))
                #print("(pred>0.5): {}".format((pred>0.5).sum()))
                
                correct += correct_
                total += total_
                
                #weight = (label == 0)  + (label > 0) * label_weight
                #weight =  (label > 0) * (label_weight-1) + 1
                if balance_classes:
                    total_num = torch.tensor(label.shape).prod()
                    n_fg = label.sum()
                    n_bg = total_num - n_fg
                    w_fg = n_bg / (total_num)
                    w_bg = n_fg / (total_num)
                    #w = w_fg + w_bg 
                    
                    weight = (label < 0.5) * w_bg + (label > 0.5) * w_fg 
                    weight *= total_num/weight.sum()
                else:
                    weight = None  
                criterion_ = criterion(weight=weight) if  type(criterion)!=list else criterion[s](weight=weight)
                
                    
                #torch.nn.BCELoss(weight=weight)
                #loss_ = criterion_(pred,label)
                if loss_with_logits:
                    loss_ = criterion_(logits,label)
                else:
                    loss_ = criterion_(pred,label)
                    
                accuracy = 100 * correct_.float().div(total_)
                if writer is not None and track_scales in [s,-1]:
                    writer[0].add_scalar(prefix+'Loss/train_'+str(s)+"_"+str(l), loss_.item(), writer[1])

                    writer[0].add_scalar(prefix+'accuracy/train_'+str(s)+"_"+str(l),accuracy , writer[1])

                    label_05_sum = label_05.sum().float()
                    TP_ = (label_05*pred_05).sum().float()
                    TP_rate = (TP_+0.0000001)/(label_05_sum+0.0000001)
                    FP_ = (torch.logical_not(label_05)*pred_05).sum().float()
                    FP_rate = (FP_+0.0000001)/((torch.logical_not(label_05)).sum().float()+0.0000001)
                    FN_ = (label_05_sum - TP_).float()
                    FN_rate = (FN_+0.0000001) /  (label_05_sum+0.0000001)
                    
                    #print("TP: {} {}".format((label_05*pred_05).sum(),(label_05).sum()))
                    #print("FP: {} {}".format((torch.logical_not(label_05)*pred_05).sum(),torch.logical_not(label_05).sum()))
                    F1_ = (TP_+0.5*(FN_+FP_))
                    F1 = (TP_ / F1_) if F1_ > 0 else 0 
                    
                    writer[0].add_scalar(prefix+'stats/F1_'+str(s)+"_"+str(l),F1 , writer[1])
                    writer[0].add_scalar(prefix+'stats/TP_'+str(s)+"_"+str(l),TP_rate , writer[1])
                    writer[0].add_scalar(prefix+'stats/FP_'+str(s)+"_"+str(l),FP_rate , writer[1])
                    writer[0].add_scalar(prefix+'stats/FN_'+str(s)+"_"+str(l),FN_rate , writer[1])

                if return_details:
                    losses[s][l] = loss_
                    accuracies[s][l] = accuracy
                
                #loss += loss_ /2**s
                #loss += loss_ *(2**s)
                loss += loss_  * (2**(loss_weight_scale_fact*s))
            
            if fgbg_pred:
                label = torch.max(labels.patchlist[s].tensor[:,None,...],dim=2)
                pred = torch.max(activation(prediction[str(s)][:,None,...]),dim=2)
                label = tls.crop2shape(label, pred.shape)
                weight =  (label > 0) * (label_weight-1) + 1
                correct_ = ((label>0.5) == (pred>0.5)).sum()
                total_ = label.numel()
                criterion_ = criterion(weight=weight)
                loss_ = criterion_(pred,label)
                accuracy = 100 * correct_.float().div(total_)

                writer[0].add_scalar(prefix+'Loss/train_fgbg_'+str(s), loss_.item(), writer[1])
                writer[0].add_scalar(prefix+'accuracy/train_fgbg_'+str(s),accuracy, writer[1])
   
                loss += loss_ 
                if return_details:
                    losses[s][num_labels] = loss_
                    accuracies[s][num_labels] = accuracy
                
    accuracy = 100 * correct.float().div(total)   
        #print("accurcay: {}".format(accuracy))
    if writer is not None:
            writer[0].add_scalar(prefix+'Loss/train_all', loss.item(), writer[1])
            writer[0].add_scalar(prefix+'accuracy/train_all', accuracy, writer[1])
    if return_details:
        return loss,accuracy,losses,accuracies
    return loss,accuracy
    