import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import argparse
from utils import get_model, load_config

py_folder = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassfierModel:
    def __init__(self, cfg):
        self.model = get_model(cfg['model']['name'], cfg['model'], cfg['model']['num_classes'])
        self.model.to(device)
        assert cfg['exp']['checkpoint'] is not None, \
            "Checkpoint path must be specified in the experiment config."
        checkpoint = cfg['exp']['checkpoint']
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint file {checkpoint} does not exist.")
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded model from {checkpoint}")
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize(int(cfg['exp']['img_size'] / 0.875), interpolation=3), 
            transforms.CenterCrop(cfg['exp']['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def predict(self, img_folder, index2class_dict):
        classes = {}
        imgs = sorted(os.listdir(img_folder))
        imgs = [os.path.join(img_folder, img) for img in imgs if img.endswith(('jpg', 'jpeg', 'png'))]
        for img_path in tqdm(imgs):
            img = self.transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
            img = img.to(device)
            with torch.no_grad():
                output = self.model(img)
                output = output.softmax(dim=1)
                output = output.cpu()
                _, predicted = output.max(1)
            img_path = os.path.basename(img_path)
            classes[img_path] = [index2class_dict[str(predicted.item())][1], output[0][predicted.item()].item()]
        return classes

def parse_args():
    parser = argparse.ArgumentParser(description='Classifier Model Prediction')
    parser.add_argument('--config', type=str, default='configs/VMamba_miniImageNet.yaml', help='Path to config file')
    parser.add_argument('--class_index_json', type=str, help='Path to class index JSON file')
    parser.add_argument('--img_folder', type=str, help='Folder containing images for prediction')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.class_index_json = os.path.join(py_folder, 'data', 'Mini-ImageNet-Dataset', 'MiniImageNet_class_index.json')
    args.img_folder = os.path.join(py_folder, 'test_images')
    cfg = load_config(args.config)
    index2class_dict = json.load(open(args.class_index_json, 'r'))
    classfier_model = ClassfierModel(cfg)
    predictions = classfier_model.predict(args.img_folder, index2class_dict)
    print(f'Predicted classes: {predictions}')