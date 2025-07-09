import copy
from functools import partial

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.init import trunc_normal_
from torch.nn import functional as F

from SVMamba.SS2D import SS2D
from SVMamba.utils import LayerNorm, PatchMerge, Permute, Linear
from SVMamba.VSSBlock import VSSBlock
from SVMamba.utils import ConvLif
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
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        # =========================
        posembed=False,
        imgsize=224,
        _SS2D=SS2D,
        # =========================
        **kwargs,
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

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None) # silu
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None) # gelu

        # 对每个 patch 使用卷积进行嵌入
        self.patch_embed = PatchEmbed(in_chans, dims[0]//self.T, patch_size, patch_norm)

        self.layers = nn.ModuleList()
        # 层与层之间下采样， 最后一层采用恒等映射
        for i_layer in range(self.num_layers):
            downsample = DownSampleBlock(
                self.dims[i_layer] // self.T, 
                self.dims[i_layer + 1] // self.T, 
                norm=True,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(VSSLayer(
                T=self.T,
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))
                    
        self.classifier = nn.Sequential(OrderedDict(
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))
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
        T, B, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # (B, T, C, H, W) 
        x = x.view(B, T*C,  H, W)  
        x = self.classifier(x)
        return x

class VSSLayer(nn.Module):
    def __init__(self,
        T=4,
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        # ===========================
        **kwargs,
    ):
        super().__init__()
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        self.blocks = nn.ModuleList()
        for d in range(depth):
            self.blocks.append(VSSBlock(
                T=T,
                hidden_dim=dim // 4, 
                drop_path=drop_path[d],
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        self.blocks.append(downsample)
        
    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        return x
        
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_norm=True):
        super().__init__()
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        
        self.convlif1 = ConvLif(T=4,
            in_channels=in_chans,
            out_channels=embed_dim // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=patch_norm,
        )
        
        self.convlif2 = ConvLif(T=4,
            in_channels=embed_dim // 2,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=patch_norm,
        )
        
    def forward(self, x: torch.Tensor):
        x = self.convlif1(x)
        x = self.convlif2(x)
        return x
        
class DownSampleBlock(nn.Module):
    def __init__(self, dim=96, out_dim=192, norm=True):
        super().__init__()
        self.convlif = ConvLif(T=4,
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=norm,
        )
        
    def forward(self, x: torch.Tensor):
        x = self.convlif(x)
        return x

if __name__ == "__main__":
    # Example usage of VSSM
    model = SVSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=100, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln2d", 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    print(model)  # Print the model architecture