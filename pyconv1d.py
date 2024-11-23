import torch
from torch import nn

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
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv1d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                      stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                      dilation=dilation, bias=bias)
            for i in range(len(pyconv_kernels))
        ])

    def forward(self, x):
        return torch.cat([level(x) for level in self.pyconv_levels], 1)
    

# if __name__ == '__main__':
#     # PyConv with two pyramid levels, kernels: 3, 5
#     m1 = PyConv1d(in_channels=64, out_channels=[32, 32], pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
#     input1 = torch.randn(4, 64, 128)
#     output1 = m1(input1)
#     print(output1.shape)
    
#     # PyConv with three pyramid levels, kernels: 3, 5, 7
#     m2 = PyConv1d(in_channels=64, out_channels=[16, 16, 32], pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
#     input2 = torch.randn(4, 64, 128)
#     output2 = m2(input2)
#     print(output2.shape)
