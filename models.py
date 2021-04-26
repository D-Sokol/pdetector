import torch
import torch.nn as nn
import torch.nn.functional as F

from DDRNet.segmentation.DDRNet_23_slim import DualResNet, BasicBlock


class Net(DualResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=False)
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, (3,3), padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (3,3), padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, (3,3), padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout(.2),
            nn.LeakyReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, (3,3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (3,3), padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (3,3), padding=0),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (3,3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 64, (1,1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 5, (1,1))
        )

    def _forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)
  
        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x = self.layer4(self.relu(x))
        x_ = self.layer4_(self.relu(x_))

        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(x)),
                        size=[height_output, width_output],
                        mode='bilinear')
        return x_

    def forward(self, x):
        x = torch.cat((self._forward(x), self.body(x)), dim=1)
        return self.head(x)

    def partial_parameters(self, from_base=True):
        if from_base:
            for layer in (self.conv1, self.layer1, self.layer2, self.layer3, self.layer3_,
                          self.down3, self.layer4, self.layer4_, self.compression3,
                          self.compression4, self.relu):
                yield from layer.parameters()
        else:
            yield from self.body.parameters()
            yield from self.head.parameters()

    def load_pretrained(self, path='extra/DDRNet23s.pth'):
        state_dict = torch.load(path)
        state_dict = self._fix_snapshot(state_dict)
        self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _fix_snapshot(state_dict):
        prefix = 'model.'
        return {key[len(prefix):]: weight
                for key, weight in state_dict.items()
                if key.startswith(prefix)}


