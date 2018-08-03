import torch as T
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.avgpool = nn.AvgPool2d(4,1,0)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.x_to_mu = nn.Linear(512,opt.n_z)
        self.x_to_logvar = nn.Linear(512, opt.n_z)

    def reparameterize(self, x):
        mu = self.x_to_mu(x)
        logvar = self.x_to_logvar(x)
        z = T.randn(mu.size())
        z = get_cuda(z)
        z = mu + z * T.exp(0.5 * logvar)
        kld = (-0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1))
        return z, kld

    def forward(self, x):
        x = self.resnet(x).squeeze()
        z, kld = self.reparameterize(x)
        return z, kld


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(opt.n_z, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_gen = self.convs(z)
        return x_gen


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):

        f_d = self.convs(x)
        x = self.last_conv(f_d)
        f_d = F.avg_pool2d(f_d, 4, 1, 0)
        return x.squeeze(), f_d.squeeze()





