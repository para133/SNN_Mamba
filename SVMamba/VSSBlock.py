import torch
import torch.nn as nn
from typing import Any
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint

from SVMamba.utils import LayerNorm, Mlp
from SVMamba.SS2D import SS2D

class VSSBlock(nn.Module):
    def __init__(
        self,
        T=4,
        hidden_dim: int = 0,
        drop_path: float = 0,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        # self.post_norm = post_norm
        self.T = T
        
        if self.ssm_branch:
            # self.norm1 = LayerNorm(hidden_dim, channel_first=channel_first)
            self.op = SS2D(
                T=T,
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        # self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            # self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(T=T, in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channel_first=channel_first)

    def forward(self, x: torch.Tensor):
        x = x.flatten(0,1)
        if self.ssm_branch:
            x = x + self.op(x)
        if self.mlp_branch:
            x = x + self.mlp(x)
        return x.view(self.T, -1, x.shape[1], x.shape[2], x.shape[3]) 
