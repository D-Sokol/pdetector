import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset


class PDDataset(Dataset):
    grid_size = 15
    _default_targets = torch.zeros(1+4, grid_size, grid_size)

    def __init__(self, folders, targets: dict=None, device='cpu'):
        super().__init__()
        self.names = []
        for folder in folders:
            for name in os.listdir(folder):
                self.names.append(os.path.join(folder, name))
        self.targets = targets
        self.device = device

    def __getitem__(self, ix):
        name = self.names[ix]
        data = plt.imread(name)
        assert data.shape == (256, 256, 3) and 0.9 <= data.max() <= 1.0
        data = torch.as_tensor(data.transpose(2, 0, 1), dtype=torch.float32).to(self.device)
        if self.targets is not None:
            y = self.targets.get(Path(name).stem, self._default_targets)
            y = torch.as_tensor(y, dtype=torch.float32).to(self.device)
            return data, y
        else:
            return data

    def __len__(self):
        return len(self.names)

