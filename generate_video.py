import torch
import random
import numpy as np
from util.config import cfg
import torch
from torch.utils.data import DataLoader
from model.EvUnrollNet import EvUnrollNet
from util.dataset import get_dataset
import lpips
from tqdm import tqdm
import numpy as np
import os
from util.viz_util import viz_event, flow_to_color
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import cv2

def main():
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    test_cfg = cfg.test
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_cfg.device = device
    
    model = EvUnrollNet(cfg.model).to(device)
    model.load_state_dict(torch.load(test_cfg.model_path, map_location=device)['model'])

    model.eval()

    test_dataset = get_dataset(cfg.test_dataset, is_video= True)

    test_loader = DataLoader(
            test_dataset,
            batch_size=test_cfg.batch_size,
            num_workers=test_cfg.workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
    
    result_path = test_cfg.result_path
    os.makedirs(result_path, exist_ok=True)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    lpips_fn = lpips.LPIPS().to(device)

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f'Validate') as pbar:
            for index, batch in enumerate(test_loader):
                batch['image'] = batch['image'].to(device, non_blocking=True)
                batch['rs_events'] = batch['rs_events'].to(device, non_blocking=True)
                batch['gs_events'] = batch['gs_events'].to(device, non_blocking=True)
                batch['events_split'] = batch['events_split'].to(device, non_blocking=True)
                batch['rs_sharp'] = batch['rs_sharp'].to(device, non_blocking=True)
                batch['gt'] = batch['gt'].to(device, non_blocking=True)
                batch['time_map'] = batch['time_map'].to(device, non_blocking=True)
                #batch['timestamp'] = batch['timestamp'].to(device)

                video_length = batch['timestamp'].shape[1]
                for i in range(video_length):
                    inputs_sample = {
                        'image': batch['image'],
                        'rs_events': batch['rs_events'],
                        'gs_events': batch['gs_events'],
                        'events_split': batch['events_split'][:, i, ...],
                        'time_map': batch['time_map'][:, i, ...],
                        'timestamp': batch['timestamp'][:, i],
                    }
                    outputs = model(inputs_sample)

                    batch_size = batch['image'].shape[0]
                    
                    rs_blur = batch['image'].permute(0,2,3,1).detach().cpu().numpy()
                    gt = batch['gt'][:,i,...].permute(0, 2, 3, 1).detach().cpu().numpy()
                    pred = torch.clamp(outputs['pred'], 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()

                    for j in range(batch_size):
                        psnr =  PSNR(gt[j], pred[j], data_range=1) 
                        ssim =  SSIM(gt[j], pred[j], data_range=1, multichannel=True)
                        lpips_score = lpips_fn(outputs['pred'][j:j+1,...], batch['gt'][j:j+1,i,...], normalize=True).item()
                        psnr_list.append(psnr)
                        ssim_list.append(ssim)
                        lpips_list.append(lpips_score)

                    
                    for j in range(batch_size):
                        rs_blur_img =(rs_blur[j]* 255).astype('uint8')
                        gt_img = (gt[j]* 255).astype('uint8')
                        pred_img = (pred[j]*255).astype('uint8')
                        imgs = np.concatenate([rs_blur_img, gt_img, pred_img], axis=1)
                        
                        seq_name = batch['seq_name'][j]
                        frame_id = batch['frame_id'][j]
                        cv2.imwrite(os.path.join(result_path, f'{seq_name}_{frame_id:05d}_{i:02d}.png'), imgs[:,:,::-1])
                pbar.set_postfix({'psnr': np.array(psnr_list[-batch_size*video_length:]).mean(), 'ssim': np.array(ssim_list[-batch_size*video_length:]).mean(), 'lpips': np.array(lpips_list[-batch_size*video_length:]).mean()})
                pbar.update(1)           
    
    print(f'psnr: {np.array(psnr_list).mean()}, ssim: {np.array(ssim_list).mean()}, lpips: {np.array(lpips_list).mean()}')
    

    
if __name__ == '__main__':
    main()
