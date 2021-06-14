import os
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.utils import save_image
import model
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(filename_suffix="includes_h5")


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def preprocessing_image(input_path, folder_name):
    if not os.path.exists(folder_name):
        with h5py.File(folder_name, "w") as df:
            images = os.listdir(input_path)
            for i, filename in enumerate(images):
                image_path = os.path.join(input_path, filename)
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert("RGB")
                        img = np.array(transform(img))
                        df.create_dataset(image_path, data=img, compression="gzip", compression_opts=4)


class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.data = []
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            for image_name in list(f['/datashare/img_align_celeba']):
                self.data.append(image_name)

    def __getitem__(self, x: int):
        image_id = self.data[x]
        image = torch.tensor(
            h5py.File(self.h5_path, "r")[os.path.join('/datashare/img_align_celeba/', image_id)])
        return image

    def __len__(self):
        return len(self.data)


def train(generator, discriminator, train_loader, latent_size, discrete_size, num_epochs, lr, **kwargs):
    generator.cuda()
    discriminator.cuda()
    generator.train()
    discriminator.train()
    g_optimizer = torch.optim.Adam(generator.parameters(), **kwargs)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), **kwargs)
    criterion = torch.nn.BCELoss()
    g_train_loss, d_train_loss = [], []
    example_c = torch.randn(64, 50).cuda()
    z_1 = torch.zeros(64, 50).cuda()
    z_2 = torch.empty(64, dtype=torch.long).cuda()
    z_2 = z_2.random_(50).cuda()
    example_d = z_1.scatter_(1, z_2.unsqueeze(1), 1.).cuda()
    example = torch.cat((example_c, example_d), 1).cuda()
    for epoch in range(num_epochs):
        g_train_loss_curr = 0
        d_train_loss_curr = 0
        for i, image in enumerate(train_loader):
            if i % 100 == 0:
                print(f"{i} / {len(train_loader)}")
            batch_size = image.size(0)
            image = image.cuda()
            real_label = torch.ones(batch_size).cuda()
            fake_label = torch.zeros(batch_size).cuda()

            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(image), real_label)
            real_loss.backward()
            continuous_latent = torch.randn(batch_size, latent_size-discrete_size).cuda()
            z1 = torch.zeros(batch_size, discrete_size).cuda()
            z2 = torch.empty(batch_size, dtype=torch.long).cuda()
            z2 = z2.random_(discrete_size).cuda()
            discrete_latent = z1.scatter_(1, z2.unsqueeze(1), 1.).cuda()
            noise = torch.cat((continuous_latent, discrete_latent), 1).cuda()
            generated = generator(noise)
            fake_loss = criterion(discriminator(generated.detach()), fake_label)
            fake_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            gen_loss = criterion(discriminator(generated), real_label)
            gen_loss.backward()
            g_optimizer.step()

            g_train_loss_curr += gen_loss.item()
            d_train_loss_curr += real_loss.item() + fake_loss.item()

        g_loss = g_train_loss_curr / train_loader.batch_size
        d_loss = d_train_loss_curr / train_loader.batch_size
        g_train_loss.append(g_loss)
        d_train_loss.append(d_loss)
        torch.save(generator.state_dict(), 'generator.pkl')
        torch.save(discriminator.state_dict(), 'discriminator.pkl')
        if epoch % 1 == 0:
            with torch.no_grad():
                generated = generator(example).detach()
            save_image(denorm(generated.data), f'images_h5/fake_images-example-{epoch+1}.png')

        writer.add_scalars('loss/epoch', {
            'generator loss': g_loss,
            'discriminator loss': d_loss
        },  epoch + 1)

        #save_image(denorm(generated.data), f'images_h5/fake_images-denorm-{epoch}.png')


        print(f'Epoch {epoch}/{num_epochs} with loss '
              f'Generator({g_train_loss[-1]:.3f}) Discriminator({d_train_loss[-1]:.3f})')


def reproduce_hw3():
    latent_size = 100
    discrete_size = 50
    channels = 3
    batch_size = 128
    hidden_layers = 64
    num_epochs = 50
    lr = 0.0002

    input_path = '/datashare/img_align_celeba'
    output_path = 'h5py_images1'

    if not os.path.isfile(output_path):
        preprocessing_image(input_path, output_path)
    train_dataset = HDF5Dataset(output_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    generator = model.Generator(latent_size, channels, hidden_layers)
    discriminator = model.Discriminator(channels, hidden_layers)
    train(generator, discriminator, train_loader, latent_size, discrete_size, num_epochs, lr, betas=(0.5, 0.999))


if __name__ == '__main__':
    reproduce_hw3()

