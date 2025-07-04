import torch
import torch.nn as nn
from functools import partial
from VMamba.model import VSSM
from QKFormer.model import QKFormer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = VSSM(
    #     depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
    #     patch_size=4, in_chans=3, num_classes=100, 
    #     ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
    #     ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
    #     ssm_init="v0", forward_type="v05_noz", 
    #     mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
    #     patch_norm=True, norm_layer="ln2d", 
    #     downsample_version="v3", patchembed_version="v2", 
    #     use_checkpoint=False, posembed=False, imgsize=224, 
    # ).to(device)
    model = QKFormer(T = 4,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=10, sr_ratios=1,
    ).to(device)    
    test_tensor = torch.randn(2, 3, 224, 224).to(device)  # Example input tensor
    output = model(test_tensor)
    print(output.shape)  # Should print the shape of the output tensor