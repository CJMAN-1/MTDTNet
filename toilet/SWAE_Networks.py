import torch
import torch.nn as nn
import torch.nn.functional as F
from SWAE.upfirdn2d import *
from SWAE.fused_act import *
from collections import OrderedDict

# SWAE
class SWAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        blur_kernel = [1, 2, 1]  # if self.opt.use_antialias else [1]

        self.Content = nn.Sequential(
            ResBlock(64, 128, blur_kernel, reflection_pad=True),
            ResBlock(128, 256, blur_kernel, reflection_pad=True, downsample=False)
        )

        self.Style = nn.Sequential(
            ResBlock(64, 128, blur_kernel, reflection_pad=True),  # 128 x 256
            ResBlock(128, 256, blur_kernel, reflection_pad=True), # 64 x 128
            ResBlock(256, 512, blur_kernel, reflection_pad=True), # 32 x 64
            ConvLayer(512, 512, kernel_size=3, blur_kernel=[1], downsample=True, pad=0), 
            ConvLayer(512, 1024, kernel_size=3, blur_kernel=[1], downsample=True, pad=0), 
        )

        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(1024, 1024)
            )
        )

    def forward(self, x, extract_features=False):
        c = self.Content(x)
        s = self.Style(x)
        s = s.mean(dim=(2, 3))
        s = self.ToGlobalCode(s)
        # c = normalize(c)  # B x 256 x 128 x 256
        # s = normalize(s)  # B x 1024

        return c, s

class SWAEDecoder(nn.Module):
    """ The Generator (decoder) architecture described in Figure 18 of
        Swapping Autoencoder (https://arxiv.org/abs/2007.00653).
        
        At high level, the architecture consists of regular and 
        upsampling residual blocks to transform the structure code into an RGB
        image. The global code is applied at each layer as modulation.
        
        Here's more detailed architecture:
        
        1. SpatialCodeModulation: First of all, modulate the structure code 
        with the global code.
        2. HeadResnetBlock: resnets at the resolution of the structure code,
        which also incorporates modulation from the global code.
        3. UpsamplingResnetBlock: resnets that upsamples by factor of 2 until
        the resolution of the output RGB image, along with the global code
        modulation.
        4. ToRGB: Final layer that transforms the output into 3 channels (RGB).
        
        Each components of the layers borrow heavily from StyleGAN2 code,
        implemented by Seonghyeon Kim.
        https://github.com/rosinality/stylegan2-pytorch
    """
    def __init__(self):
        super().__init__()
        self.netG_scale_capacity = 1.0
        self.netG_num_base_resnet_layers = 2
        self.netG_resnet_ch = 256
        self.netE_num_downsampling_sp = 4
        self.num_classes = 0
        self.global_code_ch = 2048
        self.spatial_code_ch = 8

        # num_upsamplings = self.netE_num_downsampling_sp
        blur_kernel = [1, 3, 3, 1]# if opt.use_antialias else [1]

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(1024, 256))

        
        self.g1 = ResolutionPreservingResnetBlock(256, 256, 1024)
        self.g2 = ResolutionPreservingResnetBlock(256, 128, 1024)
        self.g3 = UpsamplingResnetBlock(128, 64, 1024, blur_kernel, False)
    

    def forward(self, c, s):
        # s: B x 1024, c: B x 256 x 128 x 256
        x = self.SpatialCodeModulation(c, s)
        x = self.g1(x, s)
        x = self.g2(x, s)
        x = self.g3(x, s)
        return x

## SWAE copy
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        pad=None,
        reflection_pad=False,
    ):
        layers = []

        if downsample:
            factor = 2
            if pad is None:
                pad = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (pad + 1) // 2
            pad1 = pad // 2

            layers.append(("Blur", Blur(blur_kernel, pad=(pad0, pad1), reflection_pad=reflection_pad)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2 if pad is None else pad
            if reflection_pad:
                layers.append(("RefPad", nn.ReflectionPad2d(self.padding)))
                self.padding = 0


        layers.append(("Conv",
                       EqualConv2d(
                           in_channel,
                           out_channel,
                           kernel_size,
                           padding=self.padding,
                           stride=stride,
                           bias=bias and not activate,
                       ))
        )

        if activate:
            if bias:
                layers.append(("Act", FusedLeakyReLU(out_channel)))

            else:
                layers.append(("Act", ScaledLeakyReLU(0.2)))

        super().__init__(OrderedDict(layers))

    def forward(self, x):
        out = super().forward(x)
        return out

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], reflection_pad=False, pad=None, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, reflection_pad=reflection_pad, pad=pad)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel, reflection_pad=reflection_pad, pad=pad)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, blur_kernel=blur_kernel, activate=False, bias=False
        )

    def forward(self, input):
        #print("before first resnet layeer, ", input.shape)
        out = self.conv1(input)
        #print("after first resnet layer, ", out.shape)
        out = self.conv2(out)
        #print("after second resnet layer, ", out.shape)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)

class ResolutionPreservingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim):
        super().__init__()
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)

class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        use_noise=True,
        lr_mul=1.0,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.use_noise = use_noise
        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        if self.use_noise:
            out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
        self.fixed_noise = None
        self.image_size = None

    def forward(self, image, noise=None):
        if self.image_size is None:
            self.image_size = image.shape

        if noise is None and self.fixed_noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        elif self.fixed_noise is not None:
            noise = self.fixed_noise
            # to avoid error when generating thumbnails in demo
            if image.size(2) != noise.size(2) or image.size(3) != noise.size(3):
                noise = F.interpolate(noise, image.shape[2:], mode="nearest")
        else:
            pass  # use the passed noise

        return image + self.weight * noise

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.new_demodulation = True

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if style.dim() > 2:
            style = F.interpolate(style, size=(input.size(2), input.size(3)), mode='bilinear', align_corners=False)
            style = self.modulation(style).unsqueeze(1)
            if self.demodulate:
                style = style * torch.rsqrt(style.pow(2).mean([2], keepdim=True) + 1e-8)
            input = input * style
            weight = self.scale * self.weight
            weight = weight.repeat(batch, 1, 1, 1, 1)
        else:
            style = style.view(batch, style.size(1))
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            if self.new_demodulation:
                style = style[:, 0, :, :, :]
                if self.demodulate:
                    style = style * torch.rsqrt(style.pow(2).mean([1], keepdim=True) + 1e-8)
                input = input * style
                weight = self.scale * self.weight
                weight = weight.repeat(batch, 1, 1, 1, 1)
            else:
                weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, reflection_pad=False):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.reflection = reflection_pad
        if self.reflection:
            self.reflection_pad = nn.ReflectionPad2d((pad[0], pad[1], pad[0], pad[1]))
            self.pad = (0, 0)

    def forward(self, input):
        if self.reflection:
            input = self.reflection_pad(input)
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.dim() == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale)
            else:
                out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale,
                               bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
