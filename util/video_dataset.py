import torch
import h5py
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import os
import weakref
import cv2
import numpy as np
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange, repeat

class Sequence_video(Dataset):
    def __init__(self, cfg, seq_name):
        
        self.seq_name = seq_name
        self.seq = seq_name[:seq_name.rfind('_',0,seq_name.rfind('_'))]
        self.cfg = cfg
        self.mode = cfg.mode

        self.gt_fps = cfg.gt_fps

        self.blur_length = int(seq_name.split('_')[-1])
        self.rs_delay_length= int(seq_name.split('_')[-2])
        self.rs_length = self.rs_delay_length + self.blur_length - 1
        self.interval_length = cfg.interval_length

        self.delay_time = int(self.rs_delay_length * 1e6/self.gt_fps)
        self.whole_time = int((self.rs_length -1) * 1e6/self.gt_fps)
        self.exposure_time = int((self.blur_length -1) * 1e6/self.gt_fps)
        self.interval_time = int(self.interval_length * 1e6/self.gt_fps)

        self.img_folder = os.path.join(cfg.img_root, seq_name, 'rs_blur')
        self.mid_gt_folder = os.path.join(cfg.img_root, seq_name, 'gt')
        self.rs_sharp_folder = os.path.join(cfg.img_root, seq_name, 'rs_sharp')
        
        self.all_gt_folder = os.path.join(cfg.gt_root, self.seq)

        self.event_file = os.path.join(cfg.event_root, self.seq, f'{self.seq}.h5')

        self.gt_list = sorted(os.listdir(self.all_gt_folder))
        self.num_input= len(os.listdir(self.img_folder))

        self.crop_size = cfg.crop_size
        im0 = cv2.imread(os.path.join(self.img_folder, '00000.png'))
        self.height,self.width,_ = im0.shape
        self.outsize = self.crop_size if self.mode == 'train' else (self.width, self.height)
        
        self.voxel_grid_channel = cfg.voxel_grid_channel

        self.h5_file = None
        with h5py.File(self.event_file, 'r') as f:
            img_to_idx = f['img_to_idx']
            self.ev_idx = np.stack([img_to_idx[::self.interval_length][:self.num_input], img_to_idx[self.rs_length -1::self.interval_length]], axis=1)


        self._finalizer = weakref.finalize(self, self.close_callback, self.h5_file)

    def change_mode(self, mode):
        self.mode = mode
        self.outsize = self.crop_size if self.mode == 'train' else (self.width, self.height)

    def events_to_voxel_grid_numpy(self, event):
        width, height = self.outsize

        ch = (event[:,0].astype(np.float32) / self.exposure * self.voxel_grid_channel).astype(np.int32)
        ex = event[:,1]
        ey = event[:,2]
        ep = event[:,3].astype(np.int32)
        ep[ep == 0] = -1
        voxel_grid = np.zeros((self.voxel_grid_channel, height, width), dtype=np.int32)
        np.add.at(voxel_grid, (ch,ey,ex), ep)

        return voxel_grid

    def events_to_voxel_grid(self, event):
        width, height = self.outsize

        ch = (event[:,0].to(torch.float32) / self.exposure * self.voxel_grid_channel).long()
        torch.clamp_(ch, 0, self.voxel_grid_channel-1)
        ex = event[:,1].long()
        ey = event[:,2].long()
        ep = event[:,3].to(torch.float32)
        ep[ep == 0] = -1
        
        voxel_grid = torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        voxel_grid.index_put_((ch,ey,ex), ep, accumulate=True)
        
        return voxel_grid

    def events_to_rs_voxel_grid(self, sample):
        width, height = self.outsize
        event = sample['event']

        delay = float(self.delay_time) / self.height
        et = event[:,0].to(torch.float32)
        ex = event[:,1].long()
        ey = event[:,2].long()
        ep = event[:,3].to(torch.float32)
        ep[ep == 0] = -1
        
        gs_ch = (et/self.whole_time * self.voxel_grid_channel).long()
        rs_ch = ((et-ey*delay)/self.exposure_time * self.voxel_grid_channel).long()
        
        gs_events = torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        rs_events = torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        
        gs_valid = (gs_ch >= 0) & (gs_ch < self.voxel_grid_channel)
        rs_valid = (rs_ch >= 0) & (rs_ch < self.voxel_grid_channel)

        gs_events.index_put_((gs_ch[gs_valid],ey[gs_valid],ex[gs_valid]), ep[gs_valid], accumulate=True)
        rs_events.index_put_((rs_ch[rs_valid],ey[rs_valid],ex[rs_valid]), ep[rs_valid], accumulate=True)
        
        sample['gs_events'] = gs_events
        sample['rs_events'] = rs_events

        split_channel = self.voxel_grid_channel//2
        events_split_l = torch.zeros((split_channel, height, width), dtype=torch.float32)
        events_split_r = torch.zeros((split_channel, height, width), dtype=torch.float32)
        left_time = self.whole_time * sample['timestamp'] 
        right_time = self.whole_time * (1 - sample['timestamp'])
        left_idx = et < left_time
        right_idx = et >= left_time
        ch_l = (et[left_idx]/left_time * split_channel).long()
        ch_r = ((et[right_idx] - left_time)/right_time * split_channel).long()
        ch_l = torch.clamp(ch_l, 0, split_channel-1)
        ch_r = torch.clamp(ch_r, 0, split_channel-1)
        events_split_l.index_put_((ch_l,ey[left_idx],ex[left_idx]), ep[left_idx], accumulate=True)
        events_split_r.index_put_((ch_r,ey[right_idx],ex[right_idx]), ep[right_idx], accumulate=True)
        
        sample['events_split'] = torch.cat([events_split_l, events_split_r], dim=0)
        return sample

    @staticmethod
    def close_callback(h5f):
        if h5f is not None:
            h5f.close()

    def __len__(self):
        return self.num_input

    def augment(self, sample):
        ## crop
        h, w, _ = sample['image'].shape
        x = np.random.randint(0, w - self.crop_size[0])
        y = np.random.randint(0, h - self.crop_size[1])
        sample['image'] = sample['image'][y:y+self.crop_size[1], x:x+self.crop_size[0],:]
        sample['gt'] = sample['gt'][y:y+self.crop_size[1], x:x+self.crop_size[0],:]
        sample['rs_sharp'] = sample['rs_sharp'][y:y+self.crop_size[1], x:x+self.crop_size[0],:]
        sample['time_map'] = sample['time_map'][y:y+self.crop_size[1], x:x+self.crop_size[0],:]

        event_mask = (sample['event'][:,1] >= x) & (sample['event'][:,1] < x+self.crop_size[0]) & (sample['event'][:,2] >= y) & (sample['event'][:,2] < y+self.crop_size[1])
        sample['event'] = sample['event'][event_mask,:]
        sample['event'][:,1] = sample['event'][:,1] - x
        sample['event'][:,2] = sample['event'][:,2] - y
        # assert np.max(sample['event'][:,1]) < 1280
        # assert np.max(sample['event'][:,2]) < 720

        ## w flip
        if np.random.rand() > 0.5:
            sample['image'] = sample['image'][:, ::-1, :]
            sample['event'][:,1] = self.crop_size[0] - 1 - sample['event'][:,1]
            sample['gt'] = sample['gt'][:, ::-1, :]
            sample['rs_sharp'] = sample['rs_sharp'][:, ::-1, :]
            #sample['time_map'] = sample['time_map'][:, ::-1, :]
            sample['time_map'] = torch.flip(sample['time_map'], dims=[1])
        ## h flip
        if np.random.rand() > 0.5:
            sample['image'] = sample['image'][::-1, :, :]
            sample['event'][:,2] = self.crop_size[1] - 1 - sample['event'][:,2]
            sample['gt'] = sample['gt'][::-1, :, :]
            sample['rs_sharp'] = sample['rs_sharp'][::-1, :, :]
            #sample['time_map'] = sample['time_map'][::-1, :, :]
            sample['time_map'] = torch.flip(sample['time_map'], dims=[0])
        ## transpose
        if self.crop_size[0] == self.crop_size[1]:
            if np.random.rand() > 0.5:
                sample['image'] = sample['image'].transpose((1,0,2))
                sample['event'][:,[1,2]] = sample['event'][:,[2,1]]
                sample['gt'] = sample['gt'].transpose((1,0,2))
                sample['rs_sharp'] = sample['rs_sharp'].transpose((1,0,2))
                sample['time_map'] = sample['time_map'].permute(1,0,2)
        return sample

    def get_timemap(self, sample):
        row_stamp = torch.arange(self.height, dtype=torch.float32)/(self.height - 1)*self.rs_delay_length/self.rs_length + self.blur_length /(2 * self.rs_length)
        target_dis = row_stamp - sample['timestamp']
        
        time_map = torch.stack([row_stamp, target_dis], dim=1)
        time_map = repeat(time_map, 'h c-> h w c', w = self.width)
        
        sample['time_map'] = time_map
        return sample

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.event_file, 'r')
            self.events = self.h5_file['events']
            self.img_to_idx = self.h5_file['img_to_idx']

        img_input = cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, f'{index:05d}.png')), cv2.COLOR_BGR2RGB)
        rs_sharp_img = cv2.cvtColor(cv2.imread(os.path.join(self.rs_sharp_folder, f'{index:05d}.png')), cv2.COLOR_BGR2RGB)
        event_input = self.events[self.ev_idx[index,0]:self.ev_idx[index,1],:].astype(np.int32)
        event_input[:,0] = event_input[:,0] - self.interval_time * index
        #event_input = np.load(os.path.join(self.event_folder, f'{index:05d}.npz'))['event']

        h,w,_ = img_input.shape
        if self.mode == 'train':
            timestamp = np.random.randint(1,self.voxel_grid_channel)/self.voxel_grid_channel
        else:
            timestamp = 0.5
        gt_idx = index * self.interval_length + int(timestamp * self.rs_length)
        gt_img = cv2.cvtColor(cv2.imread(os.path.join(self.all_gt_folder, self.gt_list[gt_idx])), cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (w,h), interpolation=cv2.INTER_LINEAR)
        
        sample = {
            'image': img_input,
            'rs_sharp': rs_sharp_img,
            'gt': gt_img,
            'event': event_input,
            'timestamp': timestamp,
            'seq_name':self.seq_name,
            'frame_id':index, 
        }
        
        sample = self.get_timemap(sample)
        if self.mode == 'train':
            sample = self.augment(sample)
        
        sample['event'] = torch.from_numpy(sample['event'])
        sample['image'] = torch.from_numpy(sample['image'].copy()).permute(2,0,1).float()/255
        sample['gt'] = torch.from_numpy(sample['gt'].copy()).permute(2,0,1).float()/255
        sample['rs_sharp'] = torch.from_numpy(sample['rs_sharp'].copy()).permute(2,0,1).float()/255
        sample['time_map'] = sample['time_map'].permute(2,0,1)

        sample = self.events_to_rs_voxel_grid(sample)
        del sample['event']
        return sample


def get_video_dataset(cfg):
    all_seqs = os.listdir(cfg.img_root)
    all_seqs.sort()
    
    seq_dataset_list = []

    for seq in all_seqs:
        seq_dataset_list.append(Sequence_video(cfg, seq))
    return ConcatDataset(seq_dataset_list)

