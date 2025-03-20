from torch import nn
import torch.nn.functional as F


class SpacioTemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        time_size=2,
        activation=None,
    ):
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            device=self.device,
            dtype=dtype,
        )
        conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 1, time_size),
            stride=(*stride, 1),
            padding=(0, 0, 0),
            dilation=(*dilation, 1),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=self.device,
            dtype=dtype,
        )

        self.layers = nn.Sequential()
        self.layers.append(conv1)
        if activation is not None:
            self.layers.append(activation())
        self.layers.append(conv2)

    def forward(self, x):
        y = self.layers(x)
        return y.view(y.shape[:-1])
