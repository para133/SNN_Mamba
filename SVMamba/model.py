import copy
from functools import partial

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.init import trunc_normal_
from torch.nn import functional as F

from SVMamba.SS2D import SS2D
from SVMamba.utils import LayerNorm, PatchMerge, Permute, Linear
from SVMamba.SpikMamba import SpikMambaBlock
from SVMamba.utils import ConvBlock
class SVSSM(nn.Module):
    def __init__(
        self, 
        T = 4,
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
    ):
        super().__init__()
        self.T = T
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            # 输入 dims 作为初始维度数，后续每层翻倍（一个层若干）
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        # 生成一个从 0 到 drop_path_rate 的等间隔序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 对每个 patch 使用卷积进行嵌入
        self.patch_embed = PatchEmbed(in_chans, dims[0], patch_size, patch_norm)

        self.layers = nn.ModuleList()
        # 层与层之间下采样， 最后一层采用恒等映射
        for i_layer in range(self.num_layers):
            downsample = DownSampleBlock(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm=True,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(SVLayer(
                T=self.T,
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=downsample,
            ))
            
            self.classifier = Classifier(
                num_features=self.num_features,
                num_classes=self.num_classes,
            )
            
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)  
        x = self.classifier(x.mean(0))
        return x

class SVLayer(nn.Module):
    def __init__(self,
        T=4,
        dim=96, 
        drop_path=[0.1, 0.1], 
        downsample=nn.Identity(),
    ):
        super().__init__()
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        self.blocks = nn.ModuleList()
        for d in range(depth):
            self.blocks.append(SpikMambaBlock(
                dim=dim,
                T=T,
            ))
        self.blocks.append(downsample)
        
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x
        
class Classifier(nn.Module):
    def __init__(self, num_features, num_classes=100):
        super().__init__() 
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Linear(num_features, num_classes)
        
    def forward(self, x: torch.Tensor):
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_norm=True):
        super().__init__()
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        
        self.conv1 = ConvBlock(T=4,
            in_channels=in_chans,
            out_channels=embed_dim // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=patch_norm,
        )
        
        self.conv2 = ConvBlock(T=4,
            in_channels=embed_dim // 2,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=patch_norm,
        )
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        
class DownSampleBlock(nn.Module):
    def __init__(self, dim=96, out_dim=192, norm=True):
        super().__init__()
        self.conv = ConvBlock(T=4,
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=norm,
        )
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x
