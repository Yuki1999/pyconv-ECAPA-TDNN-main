import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class ASNorm(nn.Module):
    def __init__(self, num_impostors=300):
        super(ASNorm, self).__init__()
        self.num_impostors = num_impostors

    def forward(self, target_scores, impostor_scores):
        """
        target_scores: 目标说话人的得分 (batch_size,)
        impostor_scores: 冒名顶替者的得分 (batch_size, num_impostors)
        """
        # 计算冒名顶替者得分的均值和标准差
        impostor_mean = torch.mean(impostor_scores, dim=1, keepdim=True)
        impostor_std = torch.std(impostor_scores, dim=1, keepdim=True)

        # 归一化目标得分
        normalized_scores = (target_scores - impostor_mean) / impostor_std

        return normalized_scores

class PyConv1d(nn.Module):
    """PyConv1d with padding (general case). Applies a 1D PyConv over an input signal composed of several input planes.
    Args:
        in_channels (int): Number of channels in the input signal
        out_channels (list): Number of channels for each pyramid level produced by the convolution
        pyconv_kernels (list): Spatial size of the kernel for each pyramid level
        pyconv_groups (list): Number of blocked connections from input channels to output channels for each pyramid level
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
    """
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, pyconv_padding = False, stride=1, dilation=1, bias=False):
        super(PyConv1d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)
        if pyconv_padding == False:
            self.pyconv_levels = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                        stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                        dilation=dilation, bias=bias)
                for i in range(len(pyconv_kernels))
            ])
        else:
            self.pyconv_levels = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                        stride=stride, padding=pyconv_padding[i] // 2, groups=pyconv_groups[i],
                        dilation=dilation, bias=bias)
                for i in range(len(pyconv_kernels))
            ])

    def forward(self, x):
        return torch.cat([level(x) for level in self.pyconv_levels], 1)

class SEModule(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        # self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.conv1  = PyConv1d(inplanes, [width*scale//2, width*scale//2], pyconv_kernels=[1, 1], pyconv_groups=[1, 4])
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        # self.conv3  = PyConv1d(width*scale, [planes//2, planes//2], pyconv_kernels=[1, 1], pyconv_groups=[1, 4]),
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes, reduction=2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
            )

        self.specaug = FbankAug() # Spec augmentation

        # self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.conv1  = PyConv1d(80, [C//2, C//2], pyconv_kernels=[5, 5], pyconv_groups=[1, 4], pyconv_padding=[2,2])
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        # self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.layer4  = PyConv1d(3*C, [1536//2, 1536//2], pyconv_kernels=[1, 1], pyconv_groups=[1, 4])
        self.attention = nn.Sequential(
            # nn.Conv1d(4608, 256, kernel_size=1),
            PyConv1d(4608, [256//2, 256//2], pyconv_kernels=[1, 1], pyconv_groups=[1, 4]),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            # nn.Conv1d(256, 1536, kernel_size=1),
            PyConv1d(256, [1536//2, 1536//2], pyconv_kernels=[1, 1], pyconv_groups=[1, 4]),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

        # 添加 AS-Norm
        self.as_norm = ASNorm(num_impostors=300)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

    def score_with_as_norm(self, target_scores, impostor_scores):
        """
        使用 AS-Norm 计算归一化得分
        target_scores: 目标说话人的得分 (batch_size,)
        impostor_scores: 冒名顶替者的得分 (batch_size, num_impostors)
        """
        return self.as_norm(target_scores, impostor_scores)
