import torch
import numpy as np
from torch.utils.data import Dataset

class NeRFDataset(Dataset):
    
    def __init__(self, images, poses, render_poses, hwf, i_split, mode='train', device='cuda:0'):
        i_tr, i_val, i_te = i_split
        tr_images = np.stack([images[i] for i in i_tr])
        val_images = np.stack([images[i] for i in i_val])
        tr_poses = np.stack([poses[i] for i in i_tr])
        val_poses = np.stack([poses[i] for i in i_tr])
        self.H, self.W, self.focal = hwf
        self.mode = mode
        
        self.ipt = None
        self.gt = None
        if mode == 'train':
            self.ipt = self.init_ray_from_dataset(tr_poses).reshape(-1, 2, 3)
            self.ipt = torch.Tensor(self.ipt).to(device)
            self.gt = torch.Tensor(tr_images.reshape(-1,3)).to(device)
            assert len(self.ipt) == len(self.gt), 'input dataset size: %d, gt dataset size: %d'%(self.ipt, self.gt)
        elif mode == 'val':
            self.ipt = self.init_ray_from_dataset(val_poses).reshape(-1, 2, 3)
            self.ipt = torch.Tensor(self.ipt).to(device)
            self.gt = torch.Tensor(val_images.reshape(-1,3)).to(device)
            assert len(self.ipt) == len(self.gt), 'input dataset size: %d, gt dataset size: %d'%(self.ipt, self.gt)
        elif mode == 'test':
            self.ipt = self.init_ray_from_dataset(render_poses).reshape(-1, 2, 3)
            self.ipt = torch.Tensor(self.ipt).to(device)
        else:
            raise RuntimeError('Unknown mode.')
        
    def init_ray_from_dataset(self, poses):
        """
        Args:
            poses (numpy.array): (# of images, 4, 4)
        """
        # (set of images, set of camera parameters) -> sample R number of pixels (rays) -> o, d (unit vector)
        # self.init_ray_from_dataset(self.tr_poses)
        i, j = np.meshgrid(np.arange(self.W, dtype=np.float32), 
                           np.arange(self.H, dtype=np.float32), 
                           indexing='xy')
        d = np.stack([(i-0.5*self.W)/self.focal, (0.5*self.H-j)/self.focal, -np.ones(i.shape)], -1)
        rays = np.stack([self.init_ray_from_image(p, d) for p in poses], 0) # (N, 2, H, W, 3)
        rays = np.transpose(rays, [0,2,3,1,4]) # (N, H, W, 2=o+d, 3)
        return rays
    
    def init_ray_from_image(self, pose, d):
        ray_dirs = d.reshape(-1, 3) @ pose[:3,:3] # camera coord. to world coord.
        ray_dirs = ray_dirs.reshape(self.W, self.H, 3)
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
        ray_origins = np.broadcast_to(pose[:3,3], np.shape(ray_dirs))
        return ray_origins, ray_dirs
    
    def __len__(self):
        """if self.mode != 'train':"""
        return self.ipt.shape[0]
        """else:
            return self.ipt.shape[0] * self.ipt.shape[1] * self.ipt.shape[2]"""

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.ipt[idx]
        else:
            return self.ipt[idx], self.gt[idx]
        """
        elif self.mode == 'train':
            
            hh = self.H//4
            hw = self.W//4
            
            img_ipt = self.ipt[idx//(self.H*self.W),hh:-hh,hw:-hw,...] # (N, H, W, 2=o+d, 3)
            img_gt = self.gt[idx//(self.H*self.W),hh:-hh,hw:-hw,...] # (N, H, W, 3)
            
            img_ipt = img_ipt.reshape(-1, 2, 3)
            img_gt = img_gt.reshape(-1, 3)
            cropped_idx = idx % ((self.H-2*hh)*(self.W-2*hw))
            
            return img_ipt[cropped_idx], img_gt[cropped_idx]
            
            return self.ipt[idx//(self.H*self.W),idx%(self.H*self.W)//self.H,idx%(self.H*self.W)%self.H,...], self.gt[idx//(self.H*self.W),idx%(self.H*self.W)//self.H,idx%(self.H*self.W)%self.H,...]"""


class NeRFCropDataset(Dataset):
    
    def __init__(self, images, poses, render_poses, hwf, i_split, mode='train', device='cuda:0', num_data=500*32*32):
        i_tr, _, _ = i_split
        tr_images = np.stack([images[i] for i in i_tr])
        tr_poses = np.stack([poses[i] for i in i_tr])
        self.H, self.W, self.focal = hwf
        self.mode = mode
        
        self.ipt = None
        self.gt = None
        hh = self.H//4
        hw = self.W//4
        if mode == 'train':
            self.ipt = self.init_ray_from_dataset(tr_poses)
            self.ipt = self.ipt[:,hh:-hh,hw:-hw,...].reshape(-1, 2, 3)
            self.ipt = torch.Tensor(self.ipt).to(device)
            self.gt = torch.Tensor(tr_images[:,hh:-hh,hw:-hw,...].reshape(-1,3)).to(device)
            
            rand_idx = torch.randperm(len(self.gt))
            self.ipt = self.ipt[rand_idx][:num_data]
            self.gt = self.gt[rand_idx][:num_data]
            assert len(self.ipt) == len(self.gt), 'input dataset size: %d, gt dataset size: %d'%(self.ipt, self.gt)
        else:
            raise RuntimeError('Unknown mode.')
    
    def __len__(self):
        return len(self.gt)
        
    def init_ray_from_dataset(self, poses):
        """
        Args:
            poses (numpy.array): (# of images, 4, 4)
        """
        # (set of images, set of camera parameters) -> sample R number of pixels (rays) -> o, d (unit vector)
        # self.init_ray_from_dataset(self.tr_poses)
        i, j = np.meshgrid(np.arange(self.W, dtype=np.float32), 
                           np.arange(self.H, dtype=np.float32), 
                           indexing='xy')
        d = np.stack([(i-0.5*self.W)/self.focal, (0.5*self.H-j)/self.focal, -np.ones(i.shape)], -1)
        rays = np.stack([self.init_ray_from_image(p, d) for p in poses], 0) # (N, 2, H, W, 3)
        rays = np.transpose(rays, [0,2,3,1,4]) # (N, H, W, 2=o+d, 3)
        return rays
    
    def init_ray_from_image(self, pose, d):
        ray_dirs = d.reshape(-1, 3) @ pose[:3,:3] # camera coord. to world coord.
        ray_dirs = ray_dirs.reshape(self.W, self.H, 3)
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
        ray_origins = np.broadcast_to(pose[:3,3], np.shape(ray_dirs))
        return ray_origins, ray_dirs

    def __getitem__(self, idx):
        return self.ipt[idx], self.gt[idx]