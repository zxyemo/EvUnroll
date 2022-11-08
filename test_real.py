import torch
import random
import numpy as np
from util.real_config import cfg
import torch
from torch.utils.data import DataLoader
from model.EvUnrollNet import EvUnrollNet
from util.real_dataset import get_dataset

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

    test_dataset = get_dataset(cfg.test_dataset)

    test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            #num_workers=test_cfg.workers,
            shuffle=False,
            pin_memory=True,
            #persistent_workers=True,
        )
    
    result_path = test_cfg.result_path
    os.makedirs(result_path, exist_ok=True)

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f'Test', dynamic_ncols=True) as pbar:
            for index, batch in enumerate(test_loader):
                batch['image'] = batch['image'].to(device, non_blocking=True)
                batch['rs_events'] = batch['rs_events'].to(device, non_blocking=True)
                batch['gs_events'] = batch['gs_events'].to(device, non_blocking=True)
                batch['events_split'] = batch['events_split'].to(device, non_blocking=True)
                batch['time_map'] = batch['time_map'].to(device, non_blocking=True)
                #batch['timestamp'] = batch['timestamp'].to(device)

                outputs = model(batch,deblur_first = False)

                batch_size = batch['image'].shape[0]
                
                rs_blur = batch['image'].permute(0,2,3,1).detach().cpu().numpy()
                pred = torch.clamp(outputs['pred'], 0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()

                pbar.update(1)
                
                for j in range(batch_size):
                    rs_blur_img =(rs_blur[j]* 255).astype('uint8')
                    pred_img = (pred[j]*255).astype('uint8')
                    imgs = np.concatenate([rs_blur_img, pred_img], axis=1)
                    
                    seq_name = batch['seq_name'][j]
                    frame_id = batch['frame_id'][j]
                    cv2.imwrite(os.path.join(result_path, f'{seq_name}_{frame_id}.png'), imgs[:,:,::-1])
                
    
if __name__ == '__main__':
    main()
