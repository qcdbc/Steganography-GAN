import torch
import torch.nn as nn
import torch.autograd


class Discriminator(nn.Module):
    def __init__(self, image_shape=(3, 64, 64), df_dim=64):
        super(Discriminator, self).__init__()
        self.in_channel = image_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, df_dim, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim),
            nn.LeakyReLU(),
            nn.Conv2d(df_dim, df_dim * 2, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(),
            nn.Conv2d(df_dim * 2, df_dim * 4, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(),
            nn.Conv2d(df_dim * 4, df_dim * 8, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(),
        )
        self.fcn = nn.Sequential(
            nn.Linear(df_dim * 8, 1),
            )
        
    def forward(self, img):
        # input shape: batch_size * length * width * channel -->  64 x 64 x 64 x 3
        batch_size = img.shape[0]
        out = self.conv(img).view(batch_size, -1)  # batch_size * (df_dim * 8)
        out = self.fcn(out)
        print(out.shape)
        
        return out


class Steganalyser(nn.Module):
    def __init__(self, image_shape=(3, 64, 64), df_dim=64):
        super(Steganalyser, self).__init__()
        self.in_channel = image_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, df_dim, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim),
            nn.LeakyReLU(),
            nn.Conv2d(df_dim, df_dim * 2, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim * 2),
            nn.LeakyReLU(),
            nn.Conv2d(df_dim * 2, df_dim * 4, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim * 4),
            nn.LeakyReLU(),
            nn.Conv2d(df_dim * 4, df_dim * 8, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(df_dim * 8),
            nn.LeakyReLU(),
        )
        self.fcn = nn.Sequential(
            nn.Linear(df_dim * 8, 1),
        )

    def forward(self, img):
        # input shape: batch_size * length * width * channel -->  64 x 64 x 64 x 3
        batch_size = img.shape[0]
        out = self.conv(img).view(batch_size, -1)  # batch_size * (df_dim * 8)
        out = self.fcn(out)
        print(out.shape)

        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100, gf_dim=64, img_shape=(3, 64, 64)):
        super(Generator, self).__init__()
        self.c_dim = img_shape[0]
        self.gf_dim = gf_dim

        self.linear = nn.Linear(z_dim, gf_dim * 8 * 4 * 4)

        self.trans_conv = nn.Sequential(
            nn.BatchNorm2d(self.gf_dim * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(gf_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(gf_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim * 2, gf_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(gf_dim, self.c_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, noise):
        img = self.linear(noise).view(-1, self.gf_dim * 8, 4, 4)
        out = self.trans_conv(img)

        return out


# test
# if __name__ == '__main__':
#     noise = torch.randn(64, 100)
#     g = Generator()
#     out = g(noise)
#     print(out.shape)
