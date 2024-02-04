import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, latent_dim=64, act_fn=nn.ReLU()):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # (480, 360)
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, 2 * out_channels, 3, padding=1, stride=2),  # (240, 180)
            act_fn,
            nn.Conv2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(2 * out_channels, 4 * out_channels, 3, padding=1, stride=2),  # (120, 90)
            act_fn,
            nn.Conv2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4 * out_channels * 120 * 90, latent_dim),
            act_fn
        )

    def forward(self, x):
        x = x.view(-1, 1, 480, 360)
        output = self.net(x)
        return output