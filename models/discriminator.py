import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def disc_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2)
            )
        
        # Takes Input + Target concatenated (6 channels)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            disc_block(64, 128, 2),
            disc_block(128, 256, 1),
            nn.Conv2d(256, 1, 4, 1, 1, padding_mode="reflect")
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))