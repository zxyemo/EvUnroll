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

class Sequence_Real(Dataset):
    def __init__(self, cfg, seq_name):
        
        self.seq_name = seq_name
        self.seq = seq_name.split('_')[0]
        self.cfg = cfg

        self.fps = cfg.fps
        

        self.exposure_time = int(seq_name.split('_')[-2])
        self.delay_time= int(seq_name.split('_')[-1])
        self.whole_time = self.exposure_time + self.delay_time 
        self.interval_time = int(1e6/self.fps)

        self.img_folder = os.path.join(cfg.data_root, seq_name, 'image')
        self.img_list = sorted(os.listdir(self.img_folder))
        self.event_file = os.path.join(cfg.data_root, seq_name, f'{self.seq}.npz')
        
        self.num_input= len(self.img_list)

        im0 = cv2.imread(os.path.join(self.img_folder, self.img_list[0]))
        self.height,self.width,_ = im0.shape
        
        self.voxel_grid_channel = cfg.voxel_grid_channel

        self.ev_idx = None
        self.events = None

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
        width, height = self.width, self.height
        event = sample['event']

        delay = float(self.delay_time) / (self.height -1)
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
        
        sample['gs_events'] = -gs_events
        sample['rs_events'] = -rs_events

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
        sample['events_split'] = -sample['events_split']
        return sample


    def __len__(self):
        return self.num_input


    def get_timemap(self, sample):
        row_stamp = torch.arange(self.height, dtype=torch.float32)/(self.height - 1)*self.delay_time/self.whole_time + self.exposure_time /(2 * self.whole_time)
        target_dis = row_stamp - sample['timestamp']
        
        time_map = torch.stack([row_stamp, target_dis], dim=1)
        time_map = repeat(time_map, 'h c-> h w c', w = self.width)
        
        sample['time_map'] = time_map
        return sample

    def get_event(self, idx):
        if self.ev_idx is None:
            if self.events.ndim == 1:
                et = self.events['t']
                ex = self.events['x']
                ey = self.events['y']
                ep = self.events['p']
                self.events = np.stack([et,ex,ey,ep], axis=1)
            self.ev_idx = []
            ev_start_idx = 0
            ev_end_idx = 0
            for i in range(self.num_input):
                start_t = self.interval_time * i
                end_t = start_t + self.whole_time
                
                ev_start_idx = ev_end_idx
                while self.events[ev_start_idx,0] < start_t:
                    ev_start_idx += 1
                ev_end_idx = ev_start_idx
                while self.events[ev_end_idx,0] < end_t:
                    ev_end_idx += 1
                self.ev_idx.append((ev_start_idx, ev_end_idx))
        
        start_idx, end_idx = self.ev_idx[idx]
        event = self.events[start_idx:end_idx]
        event[:,0] = event[:,0] - self.interval_time * idx
        return event
    
    def __getitem__(self, index):
        
        if self.events is None:
            self.events = np.load(self.event_file)['event']

        img_input = cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, self.img_list[index])), cv2.COLOR_BGR2RGB)
        
        event_input = self.get_event(index)
        # event_input = self.events[ev_start_idx:ev_end_idx,:].astype(np.int32)
        # event_input[:,0] = event_input[:,0] - self.interval_time * index

        h,w,_ = img_input.shape

        timestamp = 0.5

        sample = {
            'image': img_input,
            'event': event_input,
            'timestamp': timestamp,
            'seq_name':self.seq_name,
            'frame_id':self.img_list[index].split('.')[0], 
        }
        
        sample = self.get_timemap(sample)
        
        sample['event'] = torch.from_numpy(sample['event'])
        sample['image'] = torch.from_numpy(sample['image'].copy()).permute(2,0,1).float()/255
        sample['time_map'] = sample['time_map'].permute(2,0,1)

        sample = self.events_to_rs_voxel_grid(sample)
        del sample['event']
        return sample


def get_dataset(cfg):
    all_seqs = os.listdir(cfg.data_root)
    all_seqs.sort()
    
    seq_dataset_list = []

    for seq in all_seqs:
        if os.path.isdir(os.path.join(cfg.data_root, seq)):
            seq_dataset_list.append(Sequence_Real(cfg, seq))
    return ConcatDataset(seq_dataset_list)

