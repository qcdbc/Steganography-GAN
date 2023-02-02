import torch
import torch.nn as nn
import numpy as np
import os
import sys
import torchvision.transforms as transforms
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import save_image

from model import Generator, Discriminator, Steganalyser

os.makedirs("../../generated image", exist_ok=True)

path = "../../DataSet/Celeba/"

# define hyper-parameters
EPOCH = 6
BATCH_SIZE = 64
LR = 1e-4
SAMPLE_INTERVAL = 100
LATENT_DIM = 100
N_CRITIC = 10
alpha = 0.9
# coefficient of gradient penalty
lambda_gp = 10
# Adam optimizer parameters
B1 = 0.5
B2 = 0.9

# image params
img_channels = 3
img_size = 64
img_shape = (img_channels, img_size, img_size)

# using gpu to accelerate training
cuda = True if torch.cuda.is_available() else False

celeb_set = datasets.ImageFolder(path, transform=transforms.Compose([transforms.Resize(64),
                                                                    transforms.ToTensor(),
                                                                    transforms.CenterCrop(64),
                                                                     transforms.Normalize([.5], [.5])]))
dataloader = DataLoader(celeb_set, batch_size=64, shuffle=True, drop_last=True)

print("use cuda: ", cuda)

# init nn
generator = Generator(z_dim=LATENT_DIM, img_shape=img_shape)
discriminator = Discriminator(image_shape=img_shape)
steganalyser = Steganalyser(image_shape=img_shape)

# Optimizer
optim_g = Adam(generator.parameters(), lr=LR, betas=(B1, B2))
optim_d = Adam(discriminator.parameters(), lr=LR, betas=(B1, B2))
optim_s = Adam(steganalyser.parameters(), lr=LR, betas=(B1, B2))

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    steganalyser = steganalyser.cuda()


# gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):  # samples: batch_size * channel * size * size
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

g_losses, d_losses = [], []
image_saved = 0
for epoch in range(EPOCH):
    for iters, (inputs, _) in enumerate(dataloader):
        real_imgs = Variable(inputs.type(Tensor))
        # ---------------------------------------
        # Train discriminator
        # ---------------------------------------
        optim_d.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (inputs.shape[0], LATENT_DIM, 1, 1))))
        fake_imgs = generator(z).detach()

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        loss_d.backward()
        optim_d.step()

        if iters % N_CRITIC == 0:
            # ---------------
            # train generator
            # ---------------
            optim_g.zero_grad()

            generate_img = generator(z)
            loss_g = -torch.mean(discriminator(generate_img))

            loss_g.backward()
            optim_g.step()

            print("EPOCH: [%d/%d], Batches_down: [%d/%d], d_loss = %.4f, g_loss = %.4f" % \
                  (epoch, EPOCH, iters, len(dataloader), loss_d, loss_g))

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        if iters % SAMPLE_INTERVAL == 0:
            save_image(fake_imgs.data[0], "../../generated image/{}.png".format(image_saved),)
            image_saved += 1

print("%d generate image saved." % image_saved)

x = np.array([i for i in range(len(g_losses))])
plt.figure()
plt.plot(x, g_losses)
plt.plot(x, d_losses, color='red')
plt.show()
