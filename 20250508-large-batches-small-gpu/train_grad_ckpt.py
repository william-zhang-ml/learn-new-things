"""
Training with gradient checkpoints.
"""
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import yaml


class CheckpointedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model = resnet50(num_classes=10)
        self._tappoints = list(self._model.children())

        # insert a squeeze between GAP and FC layer
        self._tappoints.append(self._tappoints[-1])
        self._tappoints[-2] = lambda tens: tens.squeeze()

        # because checkpointing discards activation cloning
        # inplace operations are not allowed
        for layer in self._model.modules():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

    def forward(self, inp: torch.Tensor, segments: int = 6) -> torch.Tensor:
        """Forward pass with gradient checkpoint.

        Args:
            inp (torch.Tensor): input tensor BCHW
            segments (int): number of model segments to checkpoint

        Returns:
            torch.Tensor: wrapped model outputs
        """
        return checkpoint_sequential(self._tappoints, segments, inp)


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
    model = CheckpointedModel().to(cfg.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    criteria = nn.CrossEntropyLoss()
    progbar = tqdm(loader)
    for imgs, labels in progbar:
        imgs.requires_grad_(True)  # must do this to gen gradients
        logits = model(imgs.to(cfg.device))
        loss = criteria(logits, labels.to(cfg.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progbar.set_postfix({'loss': f'{loss:.02f}'})
