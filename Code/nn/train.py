import torch
import torch.nn as nn
import numpy as np
import os
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import save_image

from model import Generator, Discriminator, Steganalyser

os.makedirs("../../generated image", exist_ok=True)

path = "../../DataSet/mnist"

# define hyper-parameters
EPOCH = 8
BATCH_SIZE = 64
LR = 5e-6
SAMPLE_INTERVAL = 400
LATENT_DIM = 100
CLIP_VALUE = 0.01
N_CRITIC = 10
alpha = 0.9
# Adam optimizer parameters
B1 = 0.9
B2 = 0.999

# image params
img_channels = 1
img_size = 28
img_shape = (img_channels, img_size, img_size)

# using gpu to accelerate training
cuda = True if torch.cuda.is_available() else False

# create dataset & dataloader
# celeb_set = datasets.ImageFolder(path, transform=torchvision.transforms.ToTensor())
# data_loader = DataLoader(celeb_set, batch_size=64, shuffle=True, drop_last=True)
dataloader = DataLoader(
    datasets.MNIST(path,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5], [.5])]),
                   download=True,
                   ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

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

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(EPOCH):
    batches_done = 0
    for iters, (inputs, _) in enumerate(dataloader):
        real_imgs = Variable(inputs.type(Tensor))

        # ---------------------------------------
        # Train discriminator
        # ---------------------------------------

        optim_d.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (inputs.shape[0], LATENT_DIM))))
        fake_imgs = generator(z)

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)

        loss_d = alpha * (-torch.mean(real_validity) + torch.mean(fake_validity))
        loss_d.backward()
        optim_d.step()

        # weight clipping
        for param in discriminator.parameters():
            param.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        if iters % N_CRITIC == 0:
            # ---------------
            # train generator
            # ---------------
            optim_g.zero_grad()

            loss_g = -fake_validity
            loss_g.backward()
            optim_g.step()

            print("EPOCH: [%d/%d], Batches_down: [%d/%d], d_loss = %.4f, g_loss = %.4f" % \
                  (epoch, EPOCH, iters, len(dataloader), loss_d, loss_g))

        if iters % SAMPLE_INTERVAL == 0:
            save_image(fake_imgs.data[:25], "../../images/{}.png".format(batches_done / 400), nrow=5,
                       normalize=True)
            print("%d generate image saved." % (batches_done / 400))
        batches_done += BATCH_SIZE
