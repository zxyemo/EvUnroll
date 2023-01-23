import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_block import conv, deconv, ResBlock, downConv, upConv, upConv_CA

class DeblurNet(nn.Module):
    def __init__(self , cfg) -> None:
        super().__init__()
        self.depth = cfg.depth
        base_chs = cfg.base_chs
        
        self.ev_head = nn.Conv2d(cfg.voxel_grid_channel, base_chs, kernel_size=3, stride=1, padding=1)
        self.img_head = nn.Conv2d(3, base_chs, kernel_size=3, stride=1, padding=1)
        self.ev_down = nn.ModuleList()
        self.img_down = nn.ModuleList()
        self.up_path = nn.ModuleList()

        for i in range(self.depth):
            self.ev_down.append(downConv(base_chs*2**i, base_chs*2**(i+1)))
            self.img_down.append(downConv(base_chs*2**i, base_chs*2**(i+1)))
        
        self.bottom = nn.Sequential(
            conv(base_chs*2**(self.depth +1), base_chs*2**self.depth),
            ResBlock(base_chs*2**self.depth),
            ResBlock(base_chs*2**self.depth),
        )

        for i in range(1,self.depth+1):
            self.up_path.append(upConv(base_chs*2**i, base_chs*2**(i-1), base_chs*2**i))

        self.pred = nn.Conv2d(base_chs, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, image, event):

        ev = self.ev_head(event)
        img = self.img_head(image)
        ev_downs = []
        img_downs = []
        for i in range(self.depth):
            ev, ev_skip = self.ev_down[i](ev)
            img, img_skip = self.img_down[i](img)
            ev_downs.append(ev_skip)
            img_downs.append(img_skip)
        x = torch.cat([ev, img], dim=1)
        x = self.bottom(x)
        for i in range(self.depth-1, -1, -1):
            x = self.up_path[i](x, torch.cat([ev_downs[i], img_downs[i]], dim=1))
            #x = self.up_path[i](x, ev_downs[i], img_downs[i])
        
        res = self.pred(x)
        pred_img = image + res
        
        return pred_img