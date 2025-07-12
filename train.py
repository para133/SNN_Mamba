import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from spikingjelly.clock_driven import functional
from tqdm import tqdm
import argparse

from utils import load_config, get_dataset, get_model, EmaUpdater

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
py_folder = os.path.dirname(os.path.abspath(__file__))  

def train(model, ema_model, train_loader, test_loader, criterion, optimizer, scheduler, epoch, device, log, cfg):
    log_txt = os.path.join(py_folder, 'logs', f'{log}.txt')
    best_acc = 0.
    for epoch_idx in range(1, epoch + 1):
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            # 20轮 warm up
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if cfg['exp']['use_clip_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ema_model.update()  # 更新 EMA 模型
            
            if cfg['exp']['model'] == 'QKFormer' or cfg['exp']['model'] == 'SVMamba':
                functional.reset_net(model)  # 重置模型状态，恢复神经元

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            avg_loss = total_loss / total
            acc = 100. * correct / total
            tqdm_object = tqdm._instances  # 获取当前所有tqdm实例
            if tqdm_object:
                list(tqdm_object)[-1].set_postfix(loss=avg_loss, acc=acc)
                
        scheduler.step()  
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100. * correct / total

        print(f'==> Epoch [{epoch_idx}/{epoch}] Done. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # 保存日志
        with open(log_txt, 'a') as f:
            f.write(f'Epoch [{epoch_idx}/{epoch}], '
                    f'Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    
        if epoch_idx % 20 == 1:
            model.eval()
            ema_model.apply_shadow()  # 应用 EMA 模型
            
            correct = 0
            total = 0
            # 在测试集上评估模型
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    if cfg['exp']['model'] == 'QKFormer' or cfg['exp']['model'] == 'SVMamba':
                        functional.reset_net(model)
                        
            test_acc = 100. * correct / total
            print(f'Test Accuracy after epoch {epoch_idx}: {test_acc:.2f}%')
            with open(log_txt, 'a') as f:
                f.write(f'Test Accuracy after epoch {epoch_idx}: {test_acc:.2f}%\n')
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), os.path.join(py_folder, 'checkpoints', f'{log}_best.pth'))
                print(f'Saved best model with accuracy: {test_acc:.2f}%')
                with open(log_txt, 'a') as f:
                    f.write(f'Saved best model with accuracy: {test_acc:.2f}%\n')
            # 保存当前模型
            ema_model.restore()  # 恢复 EMA 模型状态  
            torch.save(model.state_dict(), os.path.join(py_folder, 'checkpoints', f'{log}_last.pth'))
   
def parse_args():
    parser = argparse.ArgumentParser(description='Model Train')
    parser.add_argument('--config', type=str, default='configs/SVMamba_miniImageNet.yaml', help='Path to config file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config) 
    assert cfg['model']['name'] == cfg['exp']['model'], \
        f"Model name in mdoel config ({cfg['model']['name']}) does not match experiment config ({cfg['exp']['model']})"
        
    seed = cfg['exp']['seed']  # 随机种子
    g = torch.Generator()
    g.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set, test_set, class_num = get_dataset(
        dataset_name=cfg['exp']['dataset'], root_path=cfg['exp']['root_path'], img_size=cfg['exp']['img_size']
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg['exp']['batch_size'], shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg['exp']['batch_size'], shuffle=False, num_workers=4
    )
    
    warmup_epochs = cfg['exp']['warmup_epochs']
    # 加载模型
    model = get_model(cfg['model']['name'], cfg=cfg['model'], class_num=class_num).to(device)
    if cfg['exp']['pretrained']:
        assert cfg['exp']['checkpoint'] is not None, \
            "Pretrained model path must be specified in the experiment config."
        checkpoint_path = cfg['exp']['checkpoint']
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained model from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            warmup_epochs = 0  # 如果加载了预训练模型，则不需要 warmup
        else:
            raise FileNotFoundError(f"Pretrained model not found at {checkpoint_path}")
        
    # EMA 更新器
    if cfg['exp']['use_ema']:
        ema_model = EmaUpdater(model, decay=0.99) 
    else:
        ema_model = EmaUpdater(model, decay=0)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['exp']['label_smoothing']).to(device)
    
    main_epochs = cfg['exp']['epoch'] - warmup_epochs
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=cfg['exp']['lr'], weight_decay=cfg['exp']['weight_decay'], betas=(0.9, 0.999)
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg['exp']['warmup_epochs']),
            CosineAnnealingLR(optimizer, T_max=main_epochs-5, eta_min=1e-6)
        ],
        milestones=[cfg['exp']['warmup_epochs']]
    )
    
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log = f"{timestamp}_{cfg['exp']['task_name']}_{cfg['exp']['model']}_{cfg['exp']['dataset']}"
    with open(os.path.join(py_folder, 'logs', log + ".txt"), 'w') as f:
        f.write(f"Training {cfg['exp']['model']} on {cfg['exp']['dataset']} with {cfg['exp']['epoch']} epochs.\n")
        f.write(f"Batch size: {cfg['exp']['batch_size']}, Learning rate: {cfg['exp']['lr']}, "
                f"Weight decay: {cfg['exp']['weight_decay']}, Label smoothing: {cfg['exp']['label_smoothing']}\n")
        f.write(f"Using EMA: {cfg['exp']['use_ema']}, Seed: {cfg['exp']['seed']}\n")
        
    print(f"Training {cfg['exp']['model']} on {cfg['exp']['dataset']} with {cfg['exp']['epoch']} epochs.")
    print('start training...')
    train( 
        model = model, ema_model=ema_model, 
        train_loader=train_loader, test_loader=test_loader, 
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
        epoch=cfg['exp']['epoch'], device=device, 
        log=log, cfg=cfg
    )