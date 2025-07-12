import os
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from spikingjelly.clock_driven import functional

from utils import get_model, load_config

py_folder = os.path.dirname(os.path.abspath(__file__))

def get_test_dataset(dataset_name, root_path, img_size=224):
    transform = transforms.Compose([
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
        val_set = CIFAR10(root=root_path, train=False, transform=transform)
        return val_set, class_num
    
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        class_num = 100
        val_set = CIFAR100(root=root_path, train=False, transform=transform)
        return val_set, class_num
    
    elif dataset_name == "mini-imagenet":
        from torchvision.datasets import ImageFolder
        class_num = 100
        val_set = ImageFolder(root=os.path.join(root_path, 'test'), transform=transform)
        return val_set, class_num
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized.")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Model Test')
    parser.add_argument('--config', type=str, default='configs/QKFormer_miniImageNet.yaml', help='Path to config file')
    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    
    cfg = load_config(args.config)
    
    val_set, class_num = get_test_dataset(cfg['exp']['dataset'], cfg['exp']['root_path'], cfg['exp']['img_size'])
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
    
    model = get_model(cfg['model']['name'], cfg=cfg['model'], class_num=class_num).to(device)
    assert cfg['exp']['checkpoint'] is not None, \
        "Checkpoint path must be specified in the experiment config."
    checkpoint_path = cfg['exp']['checkpoint']
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
    model.eval()  
    
    total = 0
    correct = 0
    # 在测试集上评估模型
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if cfg['exp']['model'] == 'QKFormer' or cfg['exp']['model'] == 'SVMamba':
                functional.reset_net(model)
    test_acc = 100. * correct / total
    print(f'Validation Accuracy: {test_acc:.2f}%')