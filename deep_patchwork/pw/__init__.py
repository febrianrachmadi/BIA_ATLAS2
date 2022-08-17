


__all__ = ['data','patchwork','vis','tools']


import pw.patchwork
import pw.vis
import pw.data
import torch
#torch.multiprocessing.set_start_method('spawn', force=True)



def g_kernel(dim=2,w_size=3,sigma=1):
    if dim==1:
        dx = torch.arange(0,w_size)
        XX = torch.meshgrid(dx, indexing="ij")
        XX = (XX[0] - (w_size//2)).float()
        g_filt  = torch.exp(-(XX**2)/sigma**2)
        g_filt  /= torch.sum(g_filt)
        return g_filt
    if dim==2:
        
        dx = torch.arange(0,w_size)
        YY, XX = torch.meshgrid(dx, dx, indexing="ij")
        XX = (XX - (w_size//2)).float()
        YY = (YY - (w_size//2)).float()
        
        g_filt  = torch.exp(-(XX**2+YY**2)/sigma**2)
        g_filt  /= torch.sum(g_filt)
        return g_filt
    if dim==3:
        dx = torch.arange(0,w_size)
        ZZ, YY, XX = torch.meshgrid(dx, dx, dx, indexing="ij")
        XX = (XX - (w_size//2)).float()
        YY = (YY - (w_size//2)).float()
        ZZ = (ZZ - (w_size//2)).float()
        
        g_filt  = torch.exp(-(XX**2+YY**2+ZZ**2)/sigma**2)
        g_filt  /= torch.sum(g_filt)
        return g_filt





g_kernels = {}
for d in range(1,4):
    g_kernels[d] = {}
    for w in range(3,8):
        g_kernels[d][w] = {}
        for s in range(1,w//2+1):
                 g_kernels[d][w][s] = g_kernel(dim=d,w_size=w,sigma=s)
                 

#non_lin_def_smooth2D = torch.cat((pw.g_kernels[2][5][2][None,None,:,:],pw.g_kernels[2][5][2][None,None,:,:]),dim=0)
#non_lin_def_smooth3D = torch.cat((pw.g_kernels[3][5][2][None,None,:,:],pw.g_kernels[3][5][2][None,None,:,:],pw.g_kernels[3][5][2][None,None,:,:]),dim=0)