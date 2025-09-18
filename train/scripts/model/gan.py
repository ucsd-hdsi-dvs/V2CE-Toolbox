"""
Generative adversarial network implementation
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels=64*3*2, num_classes = 10, block=ResidualBlock, layers=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=[1, 1])
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        if torch.isnan(x).any():
            torch.save(x,'x_conv1.pt')
        x = self.maxpool(x)
        if torch.isnan(x).any():
            torch.save(x,'x_maxpool.pt')
        x = self.layer0(x)
        if torch.isnan(x).any():
            torch.save(x,'x_layer0.pt')
        x = self.layer1(x)
        if torch.isnan(x).any():
            torch.save(x,'x_layer1.pt')
        x = self.layer2(x)
        out=x
        if torch.isnan(x).any():
            torch.save(x,'x_layer2.pt')
        x = self.layer3(x)
        if torch.isnan(x).any():
            torch.save(x,'x_layer3.pt')
            torch.save(out,'x_layer2.pt')
        x = self.avgpool(x)
        if torch.isnan(x).any():
            torch.save(x,'x_avgpool.pt')
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if torch.isnan(x).any():
            torch.save(x,'x_fc.pt')

        return x

class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 2
        sequence = [nn.Conv3d(
            input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv3d(ndf * nf_mult_prev,
                                  ndf * nf_mult,
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw,
                                  bias=use_bias)]
            sequence += [ nn.LeakyReLU(0.2, True) ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv3d(ndf * nf_mult_prev,
                               ndf * nf_mult,
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=use_bias)]

        sequence += [ nn.LeakyReLU(0.2, True) ]
        sequence += [nn.Conv3d(ndf * nf_mult,
                               1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(
            input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev,
                                  ndf * nf_mult,
                                  kernel_size=kw,
                                  stride=2,
                                  padding=padw,
                                  bias=use_bias)]
            sequence += [ nn.LeakyReLU(0.2, True) ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev,
                               ndf * nf_mult,
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=use_bias)]

        sequence += [ nn.LeakyReLU(0.2, True) ]
        sequence += [nn.Conv2d(ndf * nf_mult,
                               1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Patch_Discriminator(nn.Module):
    def __init__(self, in_channels=20, image_channel=1,
                 ndf=64, n_layers=3, norm_layer=None, use_3d_conv=False):
        super(Patch_Discriminator, self).__init__()
        if use_3d_conv:
            # in_channels = 2
            self.disc = NLayerDiscriminator3D(in_channels,
                                    ndf=ndf,
                                    n_layers=n_layers,
                                    norm_layer=norm_layer)
        else:
            self.disc = NLayerDiscriminator(in_channels,
                                        ndf=ndf,
                                        n_layers=n_layers,
                                        norm_layer=norm_layer)

    def forward(self, x):
        return self.disc(x)

__all__ = ['GANLoss']

class Discriminator(nn.Module):
    def __init__(self, in_channels=20) -> None:
        super(Discriminator, self).__init__()
        self.resnet = ResNet(in_channels=in_channels, num_classes=2)

    def forward(self, x):
        out = self.resnet(x)

        return out

class GANLoss(nn.Module):
    def __init__(self, 
                use_3d_conv,
                in_channels,
                gan_k, 
                use_patch_gan=False
                ) -> None:
        super(GANLoss, self).__init__()
        self.gan_k = gan_k
        self.use_patch_gan = use_patch_gan
        self.use_3d_conv = use_3d_conv
        
        if use_patch_gan:
            self.discriminator=Patch_Discriminator(in_channels=in_channels, use_3d_conv=use_3d_conv)
        else:
            self.discriminator = Discriminator(in_channels=in_channels)
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            betas=(0, 0.9),
            eps=1e-8,
            lr=1e-5,
            weight_decay=1e-5
        )

        # Use None as placeholder
        self.lr_scheduler = None # get_lr_scheduler('StepLR', **lr_scheduler_args, self.d_optimizer)

    def forward(self, fake, real):
        fake = fake.float()
        real = real.float()

        if self.use_3d_conv:
            fake = rearrange(fake, 'b (p c) h w -> b p c h w', p=2)
            real = rearrange(real, 'b (p c) h w -> b p c h w', p=2)

        # Detect whether real has a gradient or not, print it
        fake_detach = fake.detach()
        self.loss = 0
        for _ in range(self.gan_k):
            self.d_optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            
            label_fake = torch.zeros_like(d_fake)
            label_real = torch.ones_like(d_real)
            # Maximize d_fake and d_real distance
            loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                    + F.binary_cross_entropy_with_logits(d_real, label_real)
            
            self.loss += loss_d.item()
            if loss_d.requires_grad is True:
                loss_d.backward()
                self.d_optimizer.step()

        # Normalized descriminator loss, for log only
        self.loss /= self.gan_k

        # Predict fake probability, minimize <generator_pred, real_label> distance
        # Here, the discriminator generates gradient for generator, and gradient flows backward
        # through <fake>, which comes from generator.
        d_fake_prob = self.discriminator(fake)
        loss_g = F.binary_cross_entropy_with_logits(d_fake_prob, label_real)
        logger.debug(f"GAN Loss: {loss_g}")
        return loss_g
    
if __name__ == '__main__':
    test_fake = torch.randn([16, 64, 260, 346])
    test_real = torch.rand_like(test_fake)

    # Discriminator test
    test_discriminator = Discriminator(in_channels=64)
    out = test_discriminator(test_fake)
    print('final shape: ', out.shape)

    # GAN test
    test_gan = GANLoss(gan_k=3)
    loss_g = test_gan(test_fake, test_real)
    print('generator loss: ', loss_g.item())
    