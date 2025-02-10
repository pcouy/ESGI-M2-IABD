from torch import nn
import torch.nn.functional as F


class SpacioTemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode = "zeros",
        device = None,
        dtype = None,
        time_size = 2,
    ):
        super().__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if type(padding) is int:
            padding = (padding, padding)
        if type(stride) is int:
            stride = (stride, stride)
        if type(dilation) is int:
            dilation = (dilation, dilation)

        conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(*kernel_size, 1),
            stride=(*stride, 1),
            padding=(*padding, 0),
            dilation=(*dilation, 1),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(*kernel_size, time_size),
            stride=(*stride, 1),
            padding=(*padding, 0),
            dilation=(*dilation, 1),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.Sequential(
            conv1,
            conv2,
        )
        

    def forward(self, x):
        y = self.layers(x)
        return y.view(y.shape[:-1])


