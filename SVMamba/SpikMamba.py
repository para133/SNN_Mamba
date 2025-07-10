import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from SVMamba.utils import LayerNorm
from SVMamba.SpikSS2D import SS2D

class SpikeLinearAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.in_proj = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.snn_in = MultiStepLIFNode(tau=2.0)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

        self.q_snn = MultiStepLIFNode(tau=2.0)
        self.k_snn = MultiStepLIFNode(tau=2.0)

        self.out_proj = nn.Linear(dim, dim)
        self.out_snn = MultiStepLIFNode(tau=2.0)

    def forward(self, x):
        T,B,C,H,W = x.shape

        x_flat = x.contiguous().view(T*B, C, H*W)  # [T*B,C,HW]
        x_flat = x_flat.permute(0, 2, 1)  # [T*B,HW,C]
        x_proj = self.in_proj(x_flat)  # [T*B,HW,C]
        x_proj = x_proj.permute(0, 2, 1) .view(T*B, C, H, W)  # [T*B,C,H,W]

        x_conv = F.silu(self.conv(x_proj)).view(T, B, C, H, W)  # [T,B,C,H,W]
        x_conv = self.snn_in(x_conv)

        x_conv_flat = x_conv.view(T, B, C, H*W).permute(0,1,3,2).flatten(0, 1)   # [T*B,HW,C]

        Q = self.q_proj(x_conv_flat)
        K = self.k_proj(x_conv_flat)
        V = x_conv_flat  # [T*B, HW, C]

        Q = self.q_snn(Q)
        K = self.k_snn(K)
        
        Q = Q.view(T*B, H*W, self.heads, self.head_dim).permute(0,2,1,3)  # [T*B, h, H*W, d]
        K = K.view(T*B, H*W, self.heads, self.head_dim).permute(0,2,1,3) 
        V = V.view(T*B, H*W, self.heads, self.head_dim).permute(0,2,1,3) 
        
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)  # [T*B, h, seq_len, d]

        out = out.permute(0,1,3,2).contiguous().view(T, B, C, H, W) 

        return out * x  
    
class SpikeMamba(nn.Module):
    def __init__(self, dim, T=4):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.snn_in = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.ssm = SS2D(
            T=T,
            d_model=dim, 
            d_state=1, 
            ssm_ratio=1.0,
            dt_rank='auto',
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=False,
            dropout=0.0,
            initialize='v0',
            forward_type='v05_noz',
            channel_first=True,
        )
        self.snn2 = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T,B,C,H,W = x.shape
        x_flat = x.contiguous().view(T*B, C, H*W)  # [T*B,C,HW]
        x_flat = x_flat.permute(0, 2, 1)  # [T*B,HW,C]
        x_proj = self.in_proj(x_flat).permute(0, 2, 1).view(T*B, C, H, W)  # [T*B,C,H, W]
        x_snn = self.snn_in(x_proj) 
        x_conv = self.conv(x_snn)  # [T*B,C,H,W]        

        x_conv = self.act(x_conv)

        y = self.ssm(x_conv)  # (T*B,C,H,W)
        y = y.view(T, B, C, H, W) # (T,B,C,H,W)
        y = self.snn2(y)

        return y * x 


class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim * expansion, dim, kernel_size=1)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        y = self.net(x.flatten(0, 1)) 
        y = y.view(T, B, C, H, W)  
        return y

class SpikMambaBlock(nn.Module):
    def __init__(self, dim, T):
        super().__init__()
        self.norm1 = LayerNorm(dim, channel_first=True)
        self.spikmamba = SpikeMamba(dim, T=T)
        self.norm2 = LayerNorm(dim, channel_first=True)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = x.flatten(0, 1)  
        x = self.norm1(x)  
        x = x.view(T, B, C, H, W) 
        x = self.spikmamba(x) + x
        x = x.flatten(0, 1)  
        x = self.norm2(x)  
        x = x.view(T, B, C, H, W) 
        x = self.ffn(x) + x
        return x
    
if __name__ == "__main__":
    T, B, C, H, W = 4, 2, 96, 56, 56
    x = torch.randn(T, B, C, H, W)  # Random input tensor
    block = SpikeMamba(dim=96)
    output = block(x)
    print(output.shape)  # Should be [T, B, C, H, W