"""
Dynamically created UNet with variable Width, Depth and activation
Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


logging.basicConfig()
logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


def depthwise_separable_conv(input_channels, output_channels):
    """Depthwise separable 3x3 convolution."""
    return nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels, bias=False),
        nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
    )


def conv3x3(input_channels, output_channels, convolution='regular'):
    if convolution == 'depthwise':
        return depthwise_separable_conv(input_channels, output_channels)
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)


def get_activation(activation='relu'):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    raise ValueError(f'Unknown activation: {activation}')


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, convolution='regular', activation='relu'):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            OrderedDict([
                ('conv1', conv3x3(input_channels, output_channels, convolution)),
                ('bn1', nn.BatchNorm2d(output_channels)),
                ('act1', get_activation(activation)),
                ('conv2', conv3x3(output_channels, output_channels, convolution)),
                ('bn2', nn.BatchNorm2d(output_channels)),
                ('act2', get_activation(activation)),
            ])
        )

    def forward(self, x):
        return self.block(x)


class UNetLext(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on depth and width."""

    def __init__(self, basic_width=32, depth=7, center_depth=2,
                 input_channels=3, output_channels=1,
                 convolution='regular', activation='relu',
                 pretrained=False, path_pretrained=None,
                 restore_weights=False, path_weights=None, **kwargs):
        """
        Parameters
        ----------
        basic_width:
            Basic width of the network, which is doubled at each layer.
        depth:
            Number of layers.
        center_depth:
            Depth of the central block in UNet.
        input_channels:
            Number of input channels.
        """
        super(UNetLext, self).__init__()

        self.basic_width = basic_width
        self.depth = depth
        self.center_depth = center_depth
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Encoder path
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = input_channels
        encoder_channels = []

        for d in range(depth):
            out_ch = basic_width * (2 ** min(d, 4))
            self.encoder.append(ConvBlock(in_ch, out_ch, convolution, activation))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            encoder_channels.append(out_ch)
            in_ch = out_ch

        # Center bottleneck
        center_blocks = OrderedDict()
        for c in range(center_depth):
            center_blocks[f'center_{c}'] = ConvBlock(in_ch, in_ch, convolution, activation)
        self.center = nn.Sequential(center_blocks)

        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for out_ch in reversed(encoder_channels):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(out_ch * 2, out_ch, convolution, activation))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)

        if restore_weights and path_weights is not None:
            self.load_state_dict(torch.load(path_weights, map_location='cpu'))
            logger.debug(f'Restored weights from {path_weights}')
        elif pretrained and path_pretrained is not None:
            self.load_state_dict(torch.load(path_pretrained, map_location='cpu'), strict=False)
            logger.debug(f'Loaded pretrained weights from {path_pretrained}')

    def forward(self, x):
        skips = []

        for enc, pool in zip(self.encoder, self.pools):
            x = enc(x)
            skips.append(x)
            # Guard against spatial dimension dropping below 1x1 when depth > 5 on 224x224 inputs
            if x.shape[-2] >= 2 and x.shape[-1] >= 2:
                x = pool(x)

        x = self.center(x)

        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            if x.shape[-2:] != skip.shape[-2:]:
                x = up(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final_conv(x)