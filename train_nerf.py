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
from support.networks import NeRF, NeRFInterface
from support.datasets import NeRFDataset, NeRFCropDataset


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


if __name__ == "__main__":
    """
    python train_nerf.py --config configs/lego.txt
    """
    
    random.seed("In-Young Cho @ KAIST")
    np.random.seed(0)
    torch.manual_seed(0)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    
    # model initialization
    print('Model, dataset, optimizer, and scheduler initialization...')
    model = NeRF(embed=True).cuda()
    #model_fine = NeRF(embed=True).cuda()
    device = next(model.parameters()).device
    
    # data initialization
    precrop_iters = 500
    BS = 1024
    
    images, poses, render_poses, hwf, i_split, near, far = load_blender_data(half_res=True, testskip=8)
    tr_dataset = NeRFDataset(images, poses, render_poses, hwf, i_split, mode='train', device=device)
    tr_crop_dataset = NeRFCropDataset(images, poses, render_poses, hwf, i_split, mode='train', device=device,
                                     num_data=precrop_iters*BS)
    val_dataset = NeRFDataset(images, poses, render_poses, hwf, i_split, mode='val', device=device)
    te_dataset = NeRFDataset(images, poses, render_poses, hwf, i_split, mode='test', device=device)
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=BS, num_workers=0)
    tr_crop_dataloader = DataLoader(tr_crop_dataset, batch_size=BS, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, num_workers=0)
    te_dataloader = DataLoader(te_dataset, batch_size=BS, num_workers=0)
    
    # optimizer and scheduler initialization
    lrate = 5e-4
    #optimizer = optim.Adam(list(model.parameters() + model_fine.parameters()), lr=lrate, betas=(0.9, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999))
    lrate_decay = 500
    decay_rate = 0.1
    
    # interface initialization
    #nerf_itf = NeRFInterface(model, optimizer, MSE, near, far, model_fine)
    nerf_itf = NeRFInterface(model, optimizer, MSE, near, far)
    
    # pre-processing
    print('Pre-processing...')
    i_train = i_split[0]
    H, W, focal = hwf
    for i in range(500):
        img_i = np.random.choice(i_train)
        target = torch.from_numpy(images[img_i])
        pose = poses[img_i,:3,:4]
        
        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

        dH = int(H//2 * 0.5)
        dW = int(W//2 * 0.5)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            ), -1)
        if i == 0:
            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {500}")

        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[BS], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 3).to(device)  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 3).to(device)  # (N_rand, 3)
        gt = target[select_coords[:, 0], select_coords[:, 1]].reshape(-1, 3).to(device)  # (N_rand, 3)
        
        view_dirs = rays_d
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        
        nerf_itf.to_train_mode()
        out = nerf_itf.forward(rays_o, rays_d, view_dirs)
        nerf_itf.backward(out, gt)
    """
    for ipt, gt in tqdm(tr_crop_dataloader, leave=False, ncols=70):
        nerf_itf.to_train_mode()
        out = nerf_itf.forward(ipt) # (R, 3)
        nerf_itf.backward(out, gt)
    """
    
    print('Initial training error: %.3e'%(nerf_itf.m_tr_loss))
    
    # video after pre-processing
    movie_base = os.path.join('./test_results',
                               'lego_spiral_pp')
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
    
    # train
    print('Start NeRF training...')
    N_iters = 200e3
    best_err = 1e10
    start_time = time.time()
    epoch = 0
    
    while (nerf_itf.get_iters() < N_iters):
        # random permutation 
        # (Using shuffle=True option makes the training slower.)
        rand_idx = torch.randperm(tr_dataloader.dataset.ipt.shape[0])
        tr_dataloader.dataset.ipt = tr_dataloader.dataset.ipt[rand_idx]
        tr_dataloader.dataset.gt = tr_dataloader.dataset.gt[rand_idx]
        epoch += 1
        
        # single-epoch training
        print(f'<-- Epoch {epoch} -->')
        for ipt, gt in tqdm(tr_dataloader, leave=False, ncols=70):
            nerf_itf.to_train_mode()
            out = nerf_itf.forward(ipt) # (R, 3)
            nerf_itf.backward(out, gt)
            
            # learning rate scheduling
            decay_steps = lrate_decay * 1000
            new_lrate = lrate * (decay_rate ** (nerf_itf.get_iters() / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            # training
            if nerf_itf.get_iters() % 500 == 0:
                print('Training error: %.3e'%(nerf_itf.m_tr_loss))
                
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
            if nerf_itf.get_iters() % 2500 == 0:
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
        
        print('Training error: %.3e'%(nerf_itf.m_tr_loss))