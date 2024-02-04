import torch
import torch.nn as nn
from .spectral_norm import SpectralNorm

class ConvLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation='LeakyReLU', norm=None, sn=False):
        super(ConvLayer1D, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                                bias=bias)
        if sn:
            self.conv1d = SpectralNorm(self.conv1d)
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = getattr(torch.nn, activation, 'LeakyReLU')
                self.activation = self.activation()
            else:
                self.activation = getattr(torch, activation, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm1d(out_channels, momentum=0.01)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm1d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv1d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

# if __name__ == '__main__':
#     x = torch.rand(1, 4, 16).cuda()
#     model = ConvLayer1D(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1, activation='LeakyReLU', norm=None, sn=False).cuda()
#     pred = model(x)
#     print(pred.shape)

class ConvLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation='LeakyReLU', norm=None, sn=False):
        super(ConvLayer2D, self).__init__()
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                bias=bias)
        if sn:
            self.conv2d = SpectralNorm(self.conv2d)
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = getattr(torch.nn, activation, 'LeakyReLU')
                self.activation = self.activation()
            else:
                self.activation = getattr(torch, activation, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.01)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConvLayer3D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, 
                 stride:int=1, 
                 padding:int=0,
                 activation:str='LeakyReLU', 
                 norm:str=None, 
                 sn:bool=False):
        super(ConvLayer3D, self).__init__()
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        bias = False if norm == 'BN' else True
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding,bias=bias)

        if sn:
            self.conv3d=SpectralNorm(self.conv3d)
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = getattr(torch.nn, activation, 'LeakyReLU')
                self.activation = self.activation()
            else:
                self.activation = getattr(torch, activation, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm3d(out_channels, momentum=0.01)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)

    def forward(self, x):
        out=self.conv3d(x)

        if self.norm is not None:#in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ConvLayer3DDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation='LeakyReLU', norm=None, sn=False):
        super(ConvLayer3DDown, self).__init__()
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        bias = False if norm == 'BN' else True

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if sn:
            self.conv3d=SpectralNorm(self.conv3d)
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = getattr(torch.nn, activation, 'LeakyReLU')
                self.activation = self.activation()
            else:
                self.activation = getattr(torch, activation, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm3d(out_channels, momentum=0.01)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)

    def forward(self, x):
        out=self.conv3d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None, sn=False):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        if isinstance(stride, int): stride = (stride, stride)
        if sn:
            self.conv1 = SpectralNorm(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=bias))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        if sn:
            self.conv2 = SpectralNorm(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=bias))
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            ) if downsample is None and not (stride == 1 and in_channels==out_channels) else downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None, sn=False):
        super(ResidualBlock3D, self).__init__()
        bias = False if norm == 'BN' else True
        if isinstance(stride, int): stride = (stride,stride,stride)
        if sn:
            self.conv1 = SpectralNorm(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=bias))
        else:
            self.conv1 =  nn.Conv3d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm3d(out_channels)
            self.bn2 = nn.InstanceNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        if sn:
            self.conv2 = SpectralNorm(
                nn.Conv3d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=bias))
        else:
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            ) if downsample is None and not (stride == 1 and in_channels==out_channels) else downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out