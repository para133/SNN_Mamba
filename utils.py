import os
import torch
import torch.nn as nn
from functools import partial
from torchvision import transforms
from thop import profile
from torchinfo import summary
from VMamba.model import VSSM
from QKFormer.model import QKFormer

def get_dataset(dataset_name, root_path, img_size=224):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), interpolation=3),  # bicubic
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(int(img_size / 0.875), interpolation=3), 
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        class_num = 10
        train_set = CIFAR10(root=root_path, train=True, transform=transform_train)
        test_set = CIFAR10(root=root_path, train=False, transform=transform_test)
        return train_set, test_set, class_num
    
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        class_num = 100
        train_set = CIFAR100(root=root_path, train=True, transform=transform_train)
        test_set = CIFAR100(root=root_path, train=False, transform=transform_test)
        return train_set, test_set, class_num
    
    elif dataset_name == "mini-imagenet":
        from torchvision.datasets import ImageFolder
        class_num = 100
        train_set = ImageFolder(root=os.path.join(root_path, 'train'), transform=transform_train)
        test_set = ImageFolder(root=os.path.join(root_path, 'val'), transform=transform_test)
        return train_set, test_set, class_num
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")
    
def get_model(model_name, cfg, class_num=100):
    if model_name == "VMamba":
        return VSSM(depths=cfg['depths'], dims=cfg['dims'], drop_path_rate=cfg['drop_path_rate'], 
            patch_size=cfg['patch_size'], in_chans=cfg['in_chans'], num_classes=class_num, 
            ssm_d_state=cfg['ssm_d_state'], ssm_ratio=cfg['ssm_ratio'], ssm_dt_rank=cfg['ssm_dt_rank'], ssm_act_layer=cfg['ssm_act_layer'],
            ssm_conv=cfg['ssm_conv'], ssm_conv_bias=cfg['ssm_conv_bias'], ssm_drop_rate=cfg['ssm_drop_rate'], 
            ssm_init=cfg['ssm_init'], forward_type=cfg['forward_type'], 
            mlp_ratio=cfg['mlp_ratio'], mlp_act_layer=cfg['mlp_act_layer'], mlp_drop_rate=cfg['mlp_drop_rate'], gmlp=cfg['gmlp'],
            patch_norm=cfg['patch_norm'], norm_layer=cfg['norm_layer'], 
            downsample_version=cfg['downsample_version'], patchembed_version=cfg['patchembed_version'], 
            use_checkpoint=cfg['use_checkpoint'], posembed=cfg['posembed'], imgsize=cfg['imgsize'], 
        )
    elif model_name == "QKFormer":
        return QKFormer(T = cfg['T'],
            img_size_h=cfg['img_size_h'], img_size_w=cfg['img_size_w'],
            patch_size=cfg['patch_size'], embed_dims=cfg['embed_dims'], num_heads=cfg['num_heads'], mlp_ratios=cfg['mlp_ratios'],
            in_channels=cfg['in_channels'], num_classes=class_num, qkv_bias=cfg['qkv_bias'],
            norm_layer=partial(nn.LayerNorm, eps=cfg['norm_layer_eps']), depths=cfg['depths'], sr_ratios=cfg['sr_ratios'],
        )
    else:
        raise ValueError(f"Model {model_name} is not recognized.")
    
def load_config(cfg_path):
    """
    Load model configuration from a YAML file.
    """
    import yaml
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_params_flops(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,))
    return params, flops

def get_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

class EmaUpdater:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
        
def adjust_learning_rate(optimizer, base_lr, epoch, warmup_epochs=20):
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr

    

if __name__ == "__main__":
    py_folder = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, class_num = get_dataset("mini-imagenet", os.path.join(py_folder, "data", "Mini-ImageNet-Dataset"), img_size=32)
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}, Number of classes: {class_num}")
    model = get_model("QKFormer", 100).to(device) 
    test_tensor = torch.randn(2, 3, 224, 224).to(device)  # Example input tensor
    output = model(test_tensor)
    print(output.shape)  
    params = get_params(model)
    print(f"Model parameters: {params / 1e6:.2f}M")
    summary(model, input_size=(2, 3, 224, 224), device=device.type)