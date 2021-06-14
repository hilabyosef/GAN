import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size, channels, hidden_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input.view(-1, self.latent_size, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, channels, hidden_size):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input).view(-1)

