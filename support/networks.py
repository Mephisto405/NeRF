import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NeRF(nn.Module):
    
    def __init__(self, inc_x=3, inc_d=3, width=256):
        super(NeRF, self).__init__()
        # Implementation according to the paper (Fig. 7)
        """
        d = 0: inc_x       -> 256
        d = 1: 256         -> 256
        d = 2: 256         -> 256
        d = 3: 256         -> 256
        d = 4: 256         -> 256
        d = 5: 256 + inc_x -> 256
        d = 6: 256         -> 256
        d = 7: 256         -> 256
        d = 8: 256         -> 256 + 1 (no act except alpha)
        d = 9: 256 + inc_d -> 128
        d =10: 128         -> 3 (sigmoid)
        """
        dims = [[inc_x, width], 
                [width, width], 
                [width, width], 
                [width, width], 
                [width, width], 
                [width + inc_x, width], 
                [width, width], 
                [width, width], 
                [width, width + 1], # no act except alpha
                [width + inc_d, width//2], 
                [width//2, 3]] # sigmoid
        
        layers = []
        for i in range(0,5):
            _inc, _outc = dims[i]
            layers.append(nn.Conv1d(_inc, _outc, 1, bias=True, groups=1))
            layers.append(nn.ReLU())
        self.embed_layers = nn.Sequential(*layers)
        
        layers = []
        for i in range(5,9):
            _inc, _outc = dims[i]
            layers.append(nn.Conv1d(_inc, _outc, 1, bias=True, groups=1))
            if i != 8:
                layers.append(nn.ReLU())
        self.feat_alpha_layers = nn.Sequential(*layers)
        
        layers = []
        for i in range(9,len(dims)):
            _inc, _outc = dims[i]
            layers.append(nn.Conv1d(_inc, _outc, 1, bias=True, groups=1))
            if i == len(dims) - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        self.rgb_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, C, H=64, W=64)
        pos = x[:,:3,...]
        drc = x[:,3:,...]
        feat = self.embed_layers(pos)
        
        feat = torch.cat([feat, pos], 1) # skip-connection
        feat = self.feat_alpha_layers(feat)
        alpha = feat[:,:1,...]
        
        feat = torch.cat([feat[:,1:,...], drc], 1)
        color = self.rgb_layers(feat)
        
        return color, F.relu(alpha)


class NeRFInterface():
    
    def __init__(self, model, optim, loss_func, near, far):
        """
        Args:
            images: numpy array of shape (# of tr&val&te images, H, W, RGBA)
            poses: numpy camera matrices of shape (# of tr&val&te images, 4, 4)
            render_poses: test-time random camera matrices (40, 4, 4)
            hwf: [800, 800, 1111.1110311937682] for the lego scene
            i_split: indices of datasets (i.e. [[0,...,99], [100,...,199], [200,...,399]])
        """
        self.model = model
        self.optim = optim
        self.loss_func = loss_func
        
        self.near = near
        self.far = far
        
        # training tracking
        self.iters = 0
        self.val_iters = 0
        self.m_tr_loss = None
        self.m_val_loss = None
    
    def get_iters(self):
        return self.iters
    
    def to_train_mode(self):
        self.model.train()
        self.iters += 1
    
    def to_eval_mode(self):
        self.model.eval()
        self.val_iters += 1
    
    def forward(self, batch):
        """
        Args:
            batch (tensor): (R, 2=o+d, 3)
        Returns:
            out (tensor): (R, 3)
        """
        self.model.zero_grad()
        
        # binning
        N_samples = 64
        #ts = torch.linspace(0.0, 1.0, N_samples+1, device=batch.device)[:-1]
        ts = torch.linspace(0.0, 1.0, N_samples+1, device=batch.device) # (S+1)
        ts = self.near + ts * (self.far - self.near)
        
        # jittering
        deltas = torch.rand(batch.shape[0], len(ts), device=batch.device)
        ts = ts.expand(batch.shape[0], -1)
        ts = ts + deltas # (R, S+1)
        
        # position sampling for stochastic quadrature
        o, d = batch[:,0,:], batch[:,1,:] # (R, 3)
        x = o[...,None] + ts[:,None,:] * d[...,None] # (R, 3, S+1)
        d = d[...,None].expand(d.shape[0], d.shape[1], x.shape[-1]) # (R, 3, S+1)
        
        # evaluating the neural radiance field
        ipt = torch.cat([x, d], 1) # (R, 6, S+1)
        color, alpha = self.model(ipt) # (R, 3, S+1), (R, 1, S+1)
        
        # rendering
        deltas = ts[:,1:] - ts[:,:-1] # (R, S)
        weights = deltas * alpha[:,0,:-1]
        weights = torch.cumsum(weights, dim=1)
        weights = torch.cat([torch.zeros(ts.shape[0],1,device=weights.device), weights], 1) # (R, S+1)
        weights = torch.exp(-weights)
        weights = weights[:,:-1] - weights[:,1:] # (R, S)
        out = torch.sum(weights[:,None,:] * color[...,:-1], -1) # (R, 3)
        
        # white background
        out += (1 - torch.sum(weights, -1)[...,None]) 
        
        return out
    
    def backward(self, out, gt):
        """
        Args:
            out (tensor): (R, 3)
            gt (tensor): (R, 3)
        """
        # back. prop.
        loss = self.loss_func(out, gt)
        loss.backward()
        
        # error handling        
        loss = loss.detach()
        if not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite loss at train time: %f"%(loss))
        
        clip = 1000
        actual = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
        if actual > clip:
            print("Clipped gradients %f -> %f"%(clip, actual))
        
        # logging
        self.m_tr_loss = loss if self.m_tr_loss is None else (self.m_tr_loss * (self.iters - 1) + loss) / self.iters
        
        # optimization
        self.optim.step()
    
    def update_val_summary(self, out, gt):
        loss = self.loss_func(out, gt)
        loss = loss.detach()
        self.m_val_loss = loss if self.m_val_loss is None else (self.m_val_loss * (self.val_iters - 1) + loss) / self.val_iters
    
    def flush_val_summary(self):
        self.val_iters = 0
        return self.m_val_loss
    
if __name__ == "__main__":
    import time
    model = NeRF().cuda()
    batch = torch.ones((64, 6, 64)).cuda()
    
    start_time = time.time()
    color, alpha = model(batch)
    print('Elapsed time: ', time.time() - start_time)
    print(color.shape)
    print(alpha.shape)
    
    import sys
    sys.path.insert(1, '/home/ubuntu/pid15-nerf/')
    from load_blender import load_blender_data
    images, poses, render_poses, hwf, i_split, near, far = load_blender_data()
    
    nerf_itf = NeRFInterface(model, None, None, images, poses, render_poses, hwf, i_split, near, far)