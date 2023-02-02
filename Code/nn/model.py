import torch
import torch.nn as nn
import torch.autograd


class Steganalyser(nn.Module):
    def __init__(self, image_shape=(3, 64, 64), df_dim=64):
        super(Steganalyser, self).__init__()
        self.in_channel = image_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(3, df_dim, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim, df_dim * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df_dim * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid(),
        )

    def forward(self, inputs):
        """
        :param inputs: batch * c_channel * image_size * image_size --> 64 x 3 x 64 x 64
        :return: batch * 1
        """
        out = self.conv(inputs)
        return out.view(-1, 1)


class Discriminator(nn.Module):
    def __init__(self, image_shape=(3, 64, 64), df_dim=64):
        super(Discriminator, self).__init__()
        self.in_channel = image_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(3, df_dim, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim, df_dim * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df_dim * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid(),
        )

    def forward(self, inputs):
        """
        :param inputs: batch * c_channel * image_size * image_size --> 64 x 3 x 64 x 64
        :return: batch * 1
        """
        out = self.conv(inputs)
        return out.view(-1, 1)


class Generator(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), z_dim=100, gf_dim=64):
        super(Generator, self).__init__()
        self.c_chan = img_shape[0]
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gf_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gf_dim * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(gf_dim * 2, gf_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(gf_dim, self.c_chan, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        """
        :param inputs: batch * z_dim --> 64 x 100
        :return: batch * c_channel * img_size * image_size --> 64 x 3 x 64 x 64
        """
        out = self.conv(inputs)
        return out


# test
# if __name__ == '__main__':
#     noise = torch.randn(64, 100)
#     g = Generator()
#     out = g(noise)
#     print(out.shape)
