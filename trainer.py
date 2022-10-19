import torch
from torch.utils.data import DataLoader, random_split
from util.loss import LapLoss, CharbonnierLoss, EdgeSmooth_loss
from model.warplayer import bwarp
from model.EvUnrollNet import EvUnrollNet
from util.dataset import get_dataset
import lpips
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import os
import datetime
from util.viz_util import viz_event, flow_to_color
from skimage.metrics import peak_signal_noise_ratio as PSNR

device = torch.device("cuda")

class Trainer:
    def __init__(self, cfg):

        self.model = EvUnrollNet(cfg.model).to(device)
        self.model.weight_init()
        self.model.freeze()

        train_cfg = cfg.train
        self.train_cfg = train_cfg
        
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=train_cfg.lr)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=train_cfg.T0, T_mult=train_cfg.Tmult, eta_min=1e-6)
        
        self.lap_Loss = LapLoss()
        self.L1_Loss = CharbonnierLoss()
        self.Smooth_Loss = EdgeSmooth_loss()
        self.perc_Loss = lpips.LPIPS().to(device)

        train_dataset_all = get_dataset(cfg.train_dataset)

        train_dataset, val_dataset = random_split(train_dataset_all, [int(len(train_dataset_all)*0.9), len(train_dataset_all)-int(len(train_dataset_all)*0.9)], generator=torch.Generator().manual_seed(42))
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.val_batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        self.train_cfg.steps_per_epoch = len(self.train_loader)
        self.writer = SummaryWriter(os.path.join(train_cfg.log_path, 'train/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
        self.writer_val = SummaryWriter(os.path.join(train_cfg.log_path, 'val/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))

        self.load_checkpoint()
        self.ckp_path = os.path.join(train_cfg.ckp_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.ckp_path, exist_ok=True)
        
    def train(self,):
        #self.evaluate(step=0)
        for epoch in range(self.start_epoch, self.train_cfg.max_epoch):
            self.model.train()
            total_loss = 0
            with tqdm(total=len(self.train_loader), desc=f'Train Epoch {epoch}') as pbar:
                for index, batch in enumerate(self.train_loader):
                    batch['image'] = batch['image'].to(device, non_blocking=True)
                    batch['rs_events'] = batch['rs_events'].to(device, non_blocking=True)
                    batch['gs_events'] = batch['gs_events'].to(device, non_blocking=True)
                    batch['events_split'] = batch['events_split'].to(device, non_blocking=True)
                    batch['rs_sharp'] = batch['rs_sharp'].to(device, non_blocking=True)
                    batch['gt'] = batch['gt'].to(device, non_blocking=True)
                    batch['time_map'] = batch['time_map'].to(device, non_blocking=True)
                    # batch['timestamp'] = batch['timestamp'].to(device)

                    output = self.model(batch)

                    self.get_learning_rate()
                    self.optimizer.zero_grad() 
                    loss_G = self.loss_fn(output, batch, self.step)
                    loss_G.backward()
                    self.optimizer.step()
                    
                    total_loss += loss_G.item()
                    pbar.set_postfix({'loss': loss_G.item(), 'total_loss': total_loss/ (index + 1)})
                    pbar.update(1)
                    self.step += 1
                    
            # self.scheduler.step()
            if epoch % self.train_cfg.checkpoint_freq == 0:
                self.save_checkpoint(epoch)
            if epoch % self.train_cfg.val_freq == 0:
                self.evaluate(epoch=epoch)
            
    
    def evaluate(self, epoch):
        self.model.eval()  
        psnr_list = []
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'Validate') as pbar:
                for index, batch in enumerate(self.val_loader):
                    batch['image'] = batch['image'].to(device, non_blocking=True)
                    batch['rs_events'] = batch['rs_events'].to(device, non_blocking=True)
                    batch['gs_events'] = batch['gs_events'].to(device, non_blocking=True)
                    batch['events_split'] = batch['events_split'].to(device, non_blocking=True)
                    batch['rs_sharp'] = batch['rs_sharp'].to(device, non_blocking=True)
                    batch['gt'] = batch['gt'].to(device, non_blocking=True)
                    batch['time_map'] = batch['time_map'].to(device, non_blocking=True)
                    #batch['timestamp'] = batch['timestamp'].to(device)

                    outputs = self.model(batch)
                    

                    batch_size = batch['image'].shape[0]
                    
                    rs_blur = batch['image'].permute(0,2,3,1).detach().cpu().numpy()
                    gt = batch['gt'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    pred = torch.clamp(outputs['pred'], 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
                    pred_syn = outputs['pred_syn'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    pred_warped = outputs['pred_warped'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    event_frame = batch['gs_events'].detach().cpu().numpy()
                    deblur = torch.clamp(outputs['deblur'], 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow = outputs['gs2rs_flow'].permute(0, 2, 3, 1).detach().cpu().numpy()

                    for j in range(batch_size):
                        psnr =  PSNR(gt[j], pred[j], data_range=1)                                                                            
                        #psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
                        psnr_list.append(psnr)

                    pbar.set_postfix({'psnr': np.array(psnr_list[-batch_size:]).mean()})
                    pbar.update(1)
                    
                    if index % self.train_cfg.log_freq == 1:
                        rs_blur_img =(rs_blur[0]* 255).astype('uint8')
                        gt_img = (gt[0]* 255).astype('uint8')
                        pred_img = (pred[0]*255).astype('uint8')
                        pred_syn_img = (pred_syn[0]*255).astype('uint8')
                        pred_warped_img = (pred_warped[0]*255).astype('uint8')
                        deblur_img = (deblur[0]*255).astype('uint8')
                        flow_viz_img = flow_to_color(flow[0])
                        ev_img = (viz_event(event_frame[0,8]) * 255).astype('uint8')
                        imgs = np.concatenate([rs_blur_img, gt_img, pred_img, pred_syn_img, pred_warped_img, flow_viz_img, deblur_img, ev_img], axis=1)
                        self.writer_val.add_image(f'val_image_epoch{epoch}', imgs, index, dataformats='HWC')
        self.writer_val.add_scalar('psnr', np.array(psnr_list).mean(), epoch)
                        

    def loss_fn(self, outputs, inputs, step):

        l1_loss = self.L1_Loss(outputs['pred'], inputs['gt'])
        perceptual_loss = self.perc_Loss(outputs['pred'], inputs['gt'], normalize=True).mean()
        syn_l1_loss = self.L1_Loss(outputs['pred_syn'], inputs['gt'])
        flow_l1_loss = self.L1_Loss(outputs['pred_warped'], inputs['gt'])
        loss_G =  l1_loss + self.train_cfg.perceptual_loss_weight * perceptual_loss + 0.5*(syn_l1_loss + flow_l1_loss)
                    
        if step % self.train_cfg.log_freq == 1:
            self.writer.add_scalar('loss', loss_G.item(), step)
            self.writer.add_scalar('l1_loss', l1_loss.item(), step)
            self.writer.add_scalar('perceptual_loss', perceptual_loss.item(), step)
            self.writer.add_scalar('syn_l1_loss', syn_l1_loss.item(), step)
            self.writer.add_scalar('flow_l1_loss', flow_l1_loss.item(), step)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], step) 

        return loss_G

    def loss_fn_flow(self, output, inputs, step):
        l1_loss = self.L1_Loss(output['pred_warped'], inputs['gt'])
        smooth_loss = self.Smooth_Loss(output['gs2rs_flow'], inputs['gt'])

        inputs['gt_0'] = inputs['gt_0'].to(device, non_blocking=True)
        inputs['gt_1'] = inputs['gt_1'].to(device, non_blocking=True)

        warped_im_0 = bwarp(inputs['gt_0'], output['gs_flow'][:,:2,:,:])
        warped_im_1 = bwarp(inputs['gt_1'], output['gs_flow'][:,2:,:,:])

        gs_flow_loss = self.L1_Loss(warped_im_0, inputs['gt']) + self.L1_Loss(warped_im_1, inputs['gt'])
        gs_flow_smooth_loss = self.Smooth_Loss(output['gs_flow'], inputs['gt'], rs=False)

        #loss_G =  l1_loss + self.train_cfg.flow_loss_weight * smooth_loss + 0.5 * (gs_flow_loss + self.train_cfg.flow_loss_weight * gs_flow_smooth_loss)
        if step > 1000:
            loss_G =  l1_loss + self.train_cfg.flow_loss_weight * smooth_loss + 0.5 * (gs_flow_loss + self.train_cfg.flow_loss_weight * gs_flow_smooth_loss)
        else:
            loss_G =  l1_loss + 0.5 * gs_flow_loss
        if step % self.train_cfg.log_freq == 1:
            self.writer.add_scalar('loss', loss_G.item(), step)
            self.writer.add_scalar('l1_loss', l1_loss.item(), step)
            self.writer.add_scalar('smooth_loss', smooth_loss.item(), step)
            self.writer.add_scalar('gs_flow_loss', gs_flow_loss.item(), step)
            self.writer.add_scalar('gs_flow_smooth_loss', gs_flow_smooth_loss.item(), step)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], step) 
        return loss_G

    def load_checkpoint(self):
        if self.train_cfg.resume_path is not None:
            checkpoint = torch.load(self.train_cfg.resume_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.step = checkpoint['step']

            print(f'Loaded checkpoint {self.train_cfg.resume_path} (epoch {self.start_epoch})')
        else:
            # checkpoint = torch.load('checkpoints/fusion_final/2022-10-17_09-38-18/048.pth')
            # self.model.load_state_dict(checkpoint['model'], strict=False)
            self.start_epoch = 0
            self.step = 0
        return

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'step': self.step,
        }
        torch.save(checkpoint, os.path.join(self.ckp_path, f'{epoch:03d}.pth'))
        return
    
    def get_learning_rate(self):
        
        mul = np.cos(self.step / (self.train_cfg.max_epoch * self.train_cfg.steps_per_epoch) * math.pi) * 0.5 + 0.5
        learning_rate = (self.train_cfg.lr - 2e-6) * mul + 2e-6
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        return 

    