import torch.nn as nn
import torch
import torch.nn.functional as F

def get_norm(norm='none'):
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'gelu':
        norm_layer = nn.GELU
    elif norm == 'none':
        norm_layer = nn.Identity
    else:
        print("=====Wrong norm type!======")
    return norm_layer

def conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm = 'none'):
    norm_layer = get_norm(norm)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,padding=padding),
        norm_layer(out_ch),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

def deconv(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm = 'none'):
    norm_layer = get_norm(norm)
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,stride=stride,padding=padding),
        norm_layer(out_ch),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

class ResBlock(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv(in_ch=in_ch, out_ch=in_ch, kernel_size=3),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,stride=1,padding=1),
        )
    def forward(self, x):
        res = self.conv(x)
        x = x + res
        return x

class downConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv1 = ResBlock(in_ch)
        self.conv2 = conv(in_ch, in_ch, kernel_size=1,stride=1,padding=0)
        self.down = conv(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x_skip = self.conv2(x)
        x = self.down(x)
        return x, x_skip

class upConv(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        self.deconv = deconv(in_ch, out_ch)
        self.conv_skip = conv(skip_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv1 = conv(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = ResBlock(out_ch)
        
    def forward(self, x, x_skip):
        x = self.deconv(x)
        x_skip = self.conv_skip(x_skip)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
    
class CA_layer(nn.Module):
    def __init__(self, in_ch, cross_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cross_ch, in_ch//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_ch//2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, cross):
        res = self.conv1(x)
        cross = self.conv2(cross)
        res = res * cross
        x = x + res
        return x

class upConv_CA(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch) -> None:
        super().__init__()
        skip_ch = skip_ch //2
        self.deconv = deconv(in_ch, out_ch)

        self.conv_skip_ev = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.conv_skip_im = nn.Conv2d(skip_ch, skip_ch, kernel_size=3, stride=1, padding=1)

        self.conv_skip = conv(skip_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv1 = conv(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = ResBlock(out_ch)
        
    def forward(self, x,  ev_skip, im_skip):
        x = self.deconv(x)
        
        im_skip_res = self.conv_skip_im(im_skip)
        ev_skip = self.conv_skip_ev(ev_skip)
        x_skip = im_skip + im_skip_res * ev_skip

        x_skip = self.conv_skip(x_skip)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

## Unet module
class Unet(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 3, base_chs = 32, depth = 3) -> None:
        super().__init__()
        self.depth = depth
        self.head = nn.Conv2d(in_ch, base_chs, kernel_size=3, stride=1, padding=1)

        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        for i in range(self.depth):
            self.down_path.append(downConv(base_chs*2**i, base_chs*2**(i+1)))
        
        self.bottom = nn.Sequential(
            ResBlock(base_chs*2**self.depth),
            ResBlock(base_chs*2**self.depth),
        )

        for i in range(1,self.depth+1):
            self.up_path.append(upConv(base_chs*2**i, base_chs*2**(i-1), base_chs*2**(i-1)))

        self.pred = nn.Conv2d(base_chs, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x_skip_list = []
        for i in range(self.depth):
            x, x_skip = self.down_path[i](x)
            x_skip_list.append(x_skip)
        x = self.bottom(x)
        for i in range(self.depth-1, -1, -1):
            x = self.up_path[i](x, x_skip_list[i])
        x = self.pred(x)
        return x


# From https://github.com/ndrplz/ConvLSTM_pytorch
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size = (3,3), num_layers = 1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param