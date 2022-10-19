import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_block import conv, deconv, ResBlock, downConv, upConv, upConv_CA, ConvLSTM, CA_layer, Unet
from model.deblurNet import DeblurNet
from einops import rearrange,repeat
from model.warplayer import bwarp

class SynthesisNet(nn.Module):
    def __init__(self , cfg) -> None:
        super().__init__()
        self.depth = cfg.depth
        base_chs = cfg.base_chs
        self.ev_input_dim = cfg.voxel_grid_channel

        ## bi-directional convlstm part for better temporal information encoding
        self.ev_head = nn.Sequential(
            nn.Conv2d(1, base_chs, kernel_size=3, stride=1, padding=1),
            ResBlock(base_chs),
        )
        self.convlstm_f = ConvLSTM(input_dim=base_chs, hidden_dim=base_chs)
        self.convlstm_b = ConvLSTM(input_dim=base_chs, hidden_dim=base_chs)

        self.time_attention = CA_layer(base_chs*2, 2+3)
        self.ev_conv = conv(base_chs*2, base_chs, 3, 1, 1)
        
        ## unet with separate encoder for image and event
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

    def forward(self, inputs):
        event = inputs['gs_events']
        image = inputs['deblur'].clone().detach()
        
        ## extract features for each channel of event
        ev = rearrange(event, 'b t h w -> (b t) () h w')
        ev = self.ev_head(ev)
        ev = rearrange(ev, '(b t) c h w -> b t c h w', t=self.ev_input_dim)
        
        batch_size = ev.shape[0]
        chs = torch.round(inputs['timestamp']*self.ev_input_dim).long()
        
        ## bi-directional convlstm
        lstm_f_list = []
        lstm_b_list = []
        for i in range(batch_size):
            lstm_out_f, _ = self.convlstm_f(ev[i:i+1, :chs[i], :, :, :])
            lstm_out_b, _ = self.convlstm_b(torch.flip(ev[i:i+1, chs[i]:, :, :, :], dims=[1]))
            lstm_f_list.append(lstm_out_f[0][:,-1,...])
            lstm_b_list.append(lstm_out_b[0][:,-1,...])
        
        lstm_f = torch.concat(lstm_f_list, dim=0) 
        lstm_b = torch.concat(lstm_b_list, dim=0)
        lstm_bi = torch.cat([lstm_f, lstm_b], dim=1)
        
        ## attention with time_map to extract rolling shutter information
        ev = self.time_attention(lstm_bi, torch.cat([image, inputs['time_map']], dim=1))

        ## unet to predict residual image
        ev = self.ev_conv(ev)
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
            
        res = torch.tanh(self.pred(x))
        pred_img = image + res
        
        return pred_img

class FlowNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.depth = cfg.depth
        base_chs = cfg.base_chs
        self.gs2gs_net = Unet(in_ch=cfg.voxel_grid_channel, out_ch=4, base_chs=base_chs, depth=self.depth)
        self.gs2rs_net = Unet(in_ch=3*3+2*3, out_ch=2, base_chs=base_chs, depth=self.depth)
    def forward(self, inputs):
        image = inputs['deblur'].clone().detach()
        event = inputs['events_split']
        timestamp = inputs['timestamp'].to(image.device)
        time_map = inputs['time_map']

        ## predict flow from ts to 0 and 1
        gs_flow = self.gs2gs_net(event)

        gsflow_t0 = gs_flow[:,0:2,...]
        gsflow_t1 = gs_flow[:,2:4,...]

        target_distance = time_map[:,1:2,...]
        timestamp = repeat(timestamp, 'b -> b () () ()').to(torch.float32)

        ## approximate the unrolling flow map
        gs2rs_flow_t0 = gsflow_t0 * (- target_distance) / timestamp
        gs2rs_flow_t1 = gsflow_t1 * target_distance / (1 - timestamp)
        
        warped_im_t0 = bwarp(image, gs2rs_flow_t0)
        warped_im_t1 = bwarp(image, gs2rs_flow_t1)

        ## predict the refined unrolling flow
        gs2rs_flow = self.gs2rs_net(torch.cat([warped_im_t0, gs2rs_flow_t0, warped_im_t1, gs2rs_flow_t1,image,time_map], dim=1))

        warped_im = bwarp(image, gs2rs_flow)

        output = {
            'gs_flow': gs_flow,
            'gs2rs_flow': gs2rs_flow,
            'pred_warped': warped_im,
        }
        return output


class EvUnrollNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.depth = cfg.depth
        base_chs = cfg.base_chs

        self.deblut_net = DeblurNet(cfg)
        self.synthesis_net = SynthesisNet(cfg)
        self.flow_net = FlowNet(cfg)
        self.fusion_net = Unet(in_ch=cfg.voxel_grid_channel+3*3+2*2, out_ch=1+3, base_chs=base_chs, depth=self.depth)
    
    ## freeze part of the network
    ## modify the freeze() and foreard() function to different part the network
    def freeze(self):
        for param in self.deblut_net.parameters():
            param.requires_grad = False
        for param in self.synthesis_net.parameters():
            param.requires_grad = False
        for param in self.flow_net.parameters():
            param.requires_grad = False

    def forward(self, inputs, deblur_first = True):
        if deblur_first:
            deblur_img = torch.clamp(self.deblut_net(inputs), 0 ,1)
            inputs['deblur'] = deblur_img
        else:
            ## use the input image as the deblurred image
            inputs['deblur'] = inputs['image']
        
        pred_syn = self.synthesis_net(inputs)
        pred_syn = torch.clamp(pred_syn, 0, 1)
        output = {
            'pred_syn': pred_syn,
            'deblur': inputs['deblur'],
        }
        flow_output = self.flow_net(inputs)
        output.update(flow_output)

        fusion_input = torch.cat([inputs['gs_events'], output['deblur'],output['pred_syn'] ,output['pred_warped'], output['gs2rs_flow'],inputs['time_map']], dim=1)
        tmp = self.fusion_net(fusion_input)
        mask = torch.sigmoid(tmp[:,0:1,...])
        res = tmp[:,1:4,...]
        output['pred'] = mask * output['pred_syn'] + (1-mask) * output['pred_warped'] + res
        return output

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # conv init
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std = 0.02)  # linear init
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight.data)  # deconv init
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 1, 0.02)
                torch.nn.init.zeros_(m.bias)

    

        