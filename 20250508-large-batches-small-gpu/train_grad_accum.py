"""
Training with gradient accumulation.
"""
from omegaconf import DictConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import yaml


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='utf-8') as file:
        cfg = DictConfig(yaml.safe_load(file))

    dataset = CIFAR10(
        cfg.data_root,
        transform=Compose([
            ToTensor(),
            Resize((224, 224)),
        ])
    )
    loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size)
    model = resnet50(num_classes=10).to(cfg.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    criteria = nn.CrossEntropyLoss()
    progbar = tqdm(loader)
    batch_counter = 0
    for imgs, labels in progbar:
        logits = model(imgs.to(cfg.device))
        loss = criteria(logits, labels.to(cfg.device))
        loss.backward()
        batch_counter += 1
        if batch_counter == cfg.batches_to_accumulate:
            optimizer.step()
            optimizer.zero_grad()
            batch_counter = 0
        progbar.set_postfix({'loss': f'{loss:.02f}'})
