import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

pip=StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

class LowRankConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, rank_ratio=4):
        super().__init__()
        mid_channels = int(out_channels / rank_ratio)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

class HybridStableDiffusion(nn.Module):
    def __init__(self, model, rank_ratio=4):
        super(HybridStableDiffusion, self).__init__()
        self.custom_model=nn.Sequential()
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d) and np.prod(module.weight.data.size())<=1280*1280*3*3:
                self.custom_model.add_module(name, LowRankConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding
                                                        , module.bias, rank_ratio=rank_ratio))
            else:
                self.custom_model.add_module(name, module)

    def forward(self, x):
        out = self.custom_model(x)
        return out


def FullRankStableDiffusion():
    return pip

def LowrankStableDiffusion(rank_ratio=4):
    return HybridStableDiffusion(pip, rank_ratio=rank_ratio)

# def StableDiffusionBenchmark(rank_ratio=None, num_classes=10):
#     return BenchmarkStableDiffusion(LowRankConv2d, rank_ratio=rank_ratio)


# def test():
#     net = BaselineStableDiffusion()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())


if __name__ == "__main__":
    model = LowrankStableDiffusion()