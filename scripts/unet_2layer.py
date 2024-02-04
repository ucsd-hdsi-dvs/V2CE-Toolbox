import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from einops import rearrange
from .submodules import ConvLayer2D, ResidualBlock, ConvLayer3D, ResidualBlock3D


def num_trainable_parameters(module):
    """Return the number of trainable parameters in the module"""
    
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  module.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class UNet(nn.Module):
    def __init__(self, num_input_channels=160, num_output_channels=16,
                 skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2,
                 norm=None, sn=False, multi=False, ret_last_feature=False):
        super(UNet, self).__init__()

        self.sn = sn
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm
        self.ret_last_feature = ret_last_feature

        self.num_encoders = num_encoders

        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * \
            pow(2, self.num_encoders)

        assert (self.num_input_channels > 0)
        assert (self.num_output_channels > 0)

        self.activation_name = self.activation
        if self.activation is not None:
            self.activation = getattr(torch, self.activation, 'sigmoid')

        # Build layers
        # N x C x H x W -> N x 32 x H x W
        self.head = ResidualBlock(self.num_input_channels, self.base_num_channels,
                              stride=1, sn=False)
        self.multi = multi

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer()

        if multi:
            self.pred_layers = self.build_multiscale_prediction_layers()

        self.init_weights()

    def __str__(self):
        summary = "UNet Architecture\n"
        summary += "Total Parameters - {}\n".format(
            num_trainable_parameters(self))
        return summary

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, 10.)
                # nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_encoders(self):
        encoder_input_sizes = []
        for i in range(self.num_encoders):
            encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        encoder_output_sizes = [self.base_num_channels * pow(2, i + 1)
                                for i in range(self.num_encoders)]

        encoders = nn.ModuleList()
        for input_size, output_size in zip(encoder_input_sizes, encoder_output_sizes):
            encoders.append(
                ResidualBlock(input_size, output_size,
                                      stride=2, norm=self.norm, sn=False)
            )

        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for _ in range(self.num_residual_blocks):
            resblocks.append(ResidualBlock(self.max_num_channels,
                                           self.max_num_channels,
                                           norm=self.norm,
                                           sn=self.sn))
        return resblocks

    def build_prediction_layer(self):
        pred = ConvLayer2D(self.base_num_channels,
                         self.num_output_channels,
                         kernel_size=1,
                         padding=0,
                         norm=None,
                         sn=None,
                         activation=self.activation_name)
        return pred

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1)
                                             for i in range(self.num_encoders)]))

        decoders = nn.ModuleList()

        first_layer = True
        for input_size in decoder_input_sizes:
            layer_input = input_size if self.skip_type == 'sum' else int(
                1.5 * input_size)
            if not first_layer and self.multi:
                layer_input += self.num_output_channels

            decoders.append(ResidualBlock(layer_input,
                                      input_size // 2,
                                      stride=1, 
                                      norm=self.norm, sn=self.sn))
            first_layer = False
        return decoders

    def build_multiscale_prediction_layers(self):
        pred_sizes = list(reversed([self.base_num_channels * pow(2, i)
                                    for i in range(self.num_encoders)]))

        pred_layers = nn.ModuleList()
        for input_size in pred_sizes:
            pred_layers.append(ConvLayer2D(input_size,
                                         self.num_output_channels,
                                         kernel_size=1,
                                         padding=0,
                                         norm=None,
                                         sn=None,
                                         activation=self.activation_name))
        return pred_layers

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # Head
        x = self.head(x)

        skip_connections = []
        # Encoder
        for i, encoder in enumerate(self.encoders):
            skip_connections.append(x)
            x = encoder(x)

        # Residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # Reverse the skip connections
        skip_connections = list(reversed(skip_connections))

        # Decoder
        all_pred = []
        for i, (skip_connection, decoder) in enumerate(zip(skip_connections, self.decoders)):
            x = f.interpolate(x, size=(skip_connection.shape[2], skip_connection.shape[3]),
                              mode='nearest')
            x = self.apply_skip_connection(x, skip_connection)
            x = decoder(x)
            if self.multi:
                all_pred.append(self.pred_layers[i](x))
                x = self.apply_skip_connection(x, all_pred[-1])

        if self.multi:
            return all_pred
        
        # Final Output
        final_pred = self.pred(x)

        if self.ret_last_feature:
            return final_pred, x
        
        return [final_pred]


class UNet3D(nn.Module):
    def __init__(self, num_input_channels=160, num_output_channels=16,
                 skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2,
                 norm=None, sn=False, multi=False, ret_last_feature=False):
        super(UNet3D, self).__init__()

        self.sn = sn
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm
        self.ret_last_feature = ret_last_feature

        self.num_encoders = num_encoders

        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * \
            pow(2, self.num_encoders)

        assert (self.num_input_channels > 0)
        assert (self.num_output_channels > 0)

        self.activation_name = self.activation
        if self.activation is not None:
            self.activation = getattr(torch, self.activation, 'sigmoid')

        # Build layers
        # N x C x H x W -> N x 32 x H x W
        self.head = ConvLayer3D(self.num_input_channels, self.base_num_channels,
                              kernel_size=3, stride=1, padding=1, sn=False)
        self.multi = multi

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer()

        if multi:
            self.pred_layers = self.build_multiscale_prediction_layers()

        self.init_weights()

    def __str__(self):
        summary = "UNet Architecture\n"
        summary += "Total Parameters - {}\n".format(
            num_trainable_parameters(self))
        return summary

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, 10.)
                # nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_encoders(self):
        encoder_input_sizes = []
        for i in range(self.num_encoders):
            encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        encoder_output_sizes = [self.base_num_channels * pow(2, i + 1)
                                for i in range(self.num_encoders)]

        encoders = nn.ModuleList()
        for input_size, output_size in zip(encoder_input_sizes, encoder_output_sizes):
            encoders.append(ResidualBlock3D(input_size, output_size, 
                                      stride=(1,2,2), norm=self.norm, sn=False))

        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for _ in range(self.num_residual_blocks):
            resblocks.append(ResidualBlock3D(self.max_num_channels,
                                           self.max_num_channels,
                                           norm=self.norm,
                                           sn=self.sn))
        return resblocks

    def build_prediction_layer(self):
        pred = ConvLayer3D(self.base_num_channels,
                         self.num_output_channels,
                         kernel_size=1,
                         padding=0,
                         norm=None,
                         sn=None,
                         activation=self.activation_name)
        return pred

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1)
                                             for i in range(self.num_encoders)]))

        decoders = nn.ModuleList()

        first_layer = True
        for input_size in decoder_input_sizes:
            layer_input = input_size if self.skip_type == 'sum' else int(
                1.5 * input_size)
            if not first_layer and self.multi:
                layer_input += self.num_output_channels

            decoders.append(ResidualBlock3D(layer_input,
                                      input_size // 2,
                                      stride=1,
                                      norm=self.norm, sn=self.sn))
            first_layer = False
        return decoders

    def build_multiscale_prediction_layers(self):
        pred_sizes = list(reversed([self.base_num_channels * pow(2, i)
                                    for i in range(self.num_encoders)]))

        pred_layers = nn.ModuleList()
        for input_size in pred_sizes:
            pred_layers.append(ConvLayer3D(input_size,
                                         self.num_output_channels,
                                         kernel_size=1,
                                         padding=0,
                                         norm=None,
                                         sn=None,
                                         activation=self.activation_name))
        return pred_layers

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # Head        
        x = self.head(x)

        skip_connections = []
        # Encoder
        for i, encoder in enumerate(self.encoders):
            skip_connections.append(x)
            x = encoder(x)
           
        for resblock in self.resblocks:
            x = resblock(x)

        # Reverse the skip connections
        skip_connections = list(reversed(skip_connections))

        # Decoder
        all_pred = []
        for i, (skip_connection, decoder) in enumerate(zip(skip_connections, self.decoders)):
            B, C, *_ = x.shape
            x = rearrange(x, 'b c l h w -> (b l) c h w')
            x = f.interpolate(x, size=(skip_connection.shape[3], skip_connection.shape[4]),
                              mode='nearest')
            x = rearrange(x, '(b l) c h w -> b c l h w', b=B)
            # print(x.shape, skip_connection.shape)
            x = self.apply_skip_connection(x, skip_connection)
            x = decoder(x)
            if self.multi:
                all_pred.append(self.pred_layers[i](x))
                x = self.apply_skip_connection(x, all_pred[-1])

        if self.multi:
            return all_pred
        
        # Final Output
        final_pred = self.pred(x)

        if self.ret_last_feature:
            return final_pred, x
        
        return [final_pred]