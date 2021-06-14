import torch
import torchvision

import model
import matplotlib.pyplot as plt

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reproduce_hw3():
    pass


def main():
    latent_size = 100
    channels = 3
    hidden_layers = 64
    generator = model.Generator(latent_size, channels, hidden_layers)
    generator.load_state_dict(torch.load('generator_h5.pkl', map_location=torch.device('cpu')))
    generator.eval()
    latent = torch.randn(1, generator.latent_size)
    continuous_latent = torch.randn(1, 50)
    z1 = torch.zeros(1, 50)
    z2 = torch.empty(1, dtype=torch.long)
    z2 = z2.random_(50)
    discrete_latent = z1.scatter_(1, z2.unsqueeze(1), 1.)
    noise = torch.cat((continuous_latent, discrete_latent), 1)
    generated = generator(noise)
    grid = torchvision.utils.make_grid(generated, padding=2, normalize=True)
    plt.imshow(grid.detach().numpy().transpose(1, 2, 0))
    plt.show()
    # plt.imshow(denorm(generated[-1]).transpose(1, 2, 0))
    # plt.show()


if __name__ == '__main__':
    main()
