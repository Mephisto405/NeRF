# Python
import os
import sys
import time
import random
import imageio
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from load_blender import load_blender_data
from support.utils import MSE, L1
from support.datasets import NeRFDataset
from support.networks import NeRF, NeRFInterface


if __name__ == "__main__":
    random.seed("In-Young Cho @ KAIST")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # model initialization
    print('Model, dataset, optimizer, and scheduler initialization...')
    model = NeRF().cuda()
    device = next(model.parameters()).device
    
    # data initialization
    images, poses, render_poses, hwf, i_split, near, far = load_blender_data()
    
    tr_dataset = NeRFDataset(images, poses, render_poses, hwf, i_split, mode='train', device=device)
    val_dataset = NeRFDataset(images, poses, render_poses, hwf, i_split, mode='val', device=device)
    te_dataset = NeRFDataset(images, poses, render_poses, hwf, i_split, mode='test', device=device)
    
    BS = 64*64
    tr_dataloader = DataLoader(tr_dataset, batch_size=BS, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, num_workers=0)
    te_dataloader = DataLoader(te_dataset, batch_size=BS, num_workers=0)
    
    # optimizer and scheduler initialization
    lrate = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999))
    lrate_decay = 250
    decay_rate = 0.1
    
    # interface initialization
    nerf_itf = NeRFInterface(model, optimizer, MSE, near, far)
    
    # train
    print('Start NeRF training...')
    N_iters = 200e3
    best_err = 1e10
    start_time = time.time()
    
    while (nerf_itf.get_iters() < N_iters):
        # random permutation 
        # (Using shuffle=True option makes the training slower.)
        rand_idx = torch.randperm(tr_dataloader.dataset.ipt.shape[0])
        tr_dataloader.dataset.ipt = tr_dataloader.dataset.ipt[rand_idx]
        tr_dataloader.dataset.gt = tr_dataloader.dataset.gt[rand_idx]
        
        # single-epoch training
        for ipt, gt in tqdm(tr_dataloader, leave=False, ncols=70):
            nerf_itf.to_train_mode()
            out = nerf_itf.forward(ipt) # (R, 3)
            nerf_itf.backward(out, gt)
            
            # learning rate scheduling
            decay_steps = lrate_decay * 1000
            new_lrate = lrate * (decay_rate ** (nerf_itf.get_iters() / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            # validation
            if nerf_itf.get_iters() % 25000 == 0:
                #print('Elapsed time for 1 epoch: %.2fs'%(time.time() - start_time))
                with torch.no_grad():
                    for ipt, gt in tqdm(val_dataloader, leave=False, ncols=70):
                        nerf_itf.to_eval_mode()
                        out = nerf_itf.forward(ipt) # (R, 3)
                        nerf_itf.update_val_summary(out, gt)

                    m_val_err = nerf_itf.flush_val_summary()
                    if m_val_err < best_err:
                        best_err = m_val_err
                        print('\tBest error: %.3e'%(best_err))

            # test
            if nerf_itf.get_iters() % 5000 == 0:
                movie_base = os.path.join('./test_results',
                                           'lego_spiral_{:06d}'.format(nerf_itf.get_iters()))
                os.makedirs(movie_base, exist_ok=True)

                result = torch.zeros((len(te_dataset.ipt), 3), device=device)
                with torch.no_grad():
                    for i, ipt in tqdm(enumerate(te_dataloader), leave=False, ncols=70):
                        nerf_itf.to_eval_mode()
                        out = nerf_itf.forward(ipt) # (R, 3)
                        result[i*BS:min(i*BS+BS, len(result))] = out
                result = result.cpu().numpy()
                result = result.reshape(-1, hwf[0], hwf[1], 3)
                result = (255*np.clip(result,0,1)).astype(np.uint8)
                imageio.mimwrite(os.path.join(movie_base, 'rgb.mp4'), result, fps=30, quality=8)
                print('Saved test set.')