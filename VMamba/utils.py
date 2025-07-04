import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x

class PatchMerge(nn.Module):
    def __init__(self, channel_first=True, in_channel_first=False, out_channel_first=False,):
        nn.Module.__init__(self)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first
        # print(f"WARNING: output [(0, 0), (1, 0), (0, 1), (1, 1)] for (H, W).")

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if not self.in_channel_first:
            B, H, W, C = x.shape
        
        if (W % 2 != 0) or (H % 2 != 0):
            PH, PW = H - H % 2, W - W % 2
            pad_shape = (PW // 2, PW - PW // 2, PH // 2, PH - PH // 2)
            pad_shape = (*pad_shape, 0, 0, 0, 0) if self.in_channel_first else (0, 0, *pad_shape, 0, 0)
            x = nn.functional.pad(x, pad_shape)
        
        xs = [
            x[..., 0::2, 0::2], x[..., 1::2, 0::2], 
            x[..., 0::2, 1::2], x[..., 1::2, 1::2],
        ] if self.in_channel_first else [
            x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :], 
            x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :],
        ]

        xs = torch.cat(xs, (1 if self.out_channel_first else -1))
        
        return xs

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Linear(nn.Linear):
    def __init__(self, *args, channel_first=False, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups
    
    def forward(self, x: torch.Tensor):
        # 继承自 nn.linear 所以会更新权重 
        if self.channel_first:
            # B, C, H, W = x.shape
            if len(x.shape) == 4:
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif len(x.shape) == 3:
                return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)
        else:
            return F.linear(x, self.weight, self.bias)
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError
