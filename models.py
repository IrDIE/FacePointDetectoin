from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from loguru import logger
import lightning as pl

class ONet(nn.Module):
    def __init__(self, in_ch = 3, img_size = 48):
        super(ONet, self).__init__()
        self.in_ch = in_ch
        self.got_fc_dim = False
        self.setup_model(img_size=img_size)


    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    def setup_model(self, img_size):
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=1),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        if not self.got_fc_dim :
            with torch.no_grad():
                self.fc_size = self.conv(torch.ones(size=(2, self.in_ch, img_size, img_size))).shape[-1]
                self.got_fc_dim = True
        self.model = nn.Sequential(
            self.conv

        )
        self.fc = nn.Sequential(

            nn.Linear(self.fc_size, 256),
            nn.ReLU(),
            nn.Linear(256, 136)

        )
        # logger.info(f'created model :\n{self.model}')

class ONetLightning(pl.LightningModule):
    def __init__(self, lr, in_ch = 3, img_size = 48):
        super(ONetLightning, self).__init__()
        self.in_ch = in_ch
        self.got_fc_dim = False
        self.setup_model(img_size=img_size)
        self.loss_fn = nn.MSELoss()
        self.lr= lr

    def forward(self, x):
        return self.model(x)
    def setup_model(self, img_size):
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=1),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        if not self.got_fc_dim :
            with torch.no_grad():
                self.fc_size = self.conv(torch.ones(size=(1, self.in_ch, img_size, img_size))).flatten()
                logger.info(f'self.fc_size={self.fc_size}')
                self.got_fc_dim = True
        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(self.fc_size, 256),
            nn.ReLU(),
            nn.Linear(256, 136)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        return loss
    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        return [opt], [sch]


def main():

    net = ONet()


if __name__=="__main__":
    main()