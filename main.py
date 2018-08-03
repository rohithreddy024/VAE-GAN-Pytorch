import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import torch as T
import torch.nn.functional as F
from model import *
import random
from helper_functions import *
from dataset import get_data_loader
import torchvision.utils as utils
import argparse


save_path = "data/saved_models/saved_model.tar"

if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=301)
parser.add_argument('--lr_e', type=float, default=0.0002)
parser.add_argument('--lr_g', type=float, default=0.0002)
parser.add_argument('--lr_d', type=float, default=0.0002)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--n_samples", type=int, default=36)
parser.add_argument('--n_z', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--w_kld', type=float, default=1)
parser.add_argument('--w_loss_g', type=float, default=0.01)
parser.add_argument('--w_loss_gd', type=float, default=1)

def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False

parser.add_argument('--resume_training', type=str2bool, default=False)
parser.add_argument('--to_train', type=str2bool, default=True)

opt = parser.parse_args()
print(opt)

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
T.manual_seed(manual_seed)
if T.cuda.is_available():
    T.cuda.manual_seed_all(manual_seed)

train_loader = get_data_loader(opt)

E = get_cuda(Encoder(opt))
G = get_cuda(Generator(opt)).apply(weights_init)
D = get_cuda(Discriminator()).apply(weights_init)

device_ids = range(T.cuda.device_count())
E = nn.DataParallel(E, device_ids)
G = nn.DataParallel(G, device_ids)
D = nn.DataParallel(D, device_ids)

E_trainer = T.optim.Adam(E.parameters(), lr=opt.lr_e)
G_trainer = T.optim.Adam(G.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
D_trainer = T.optim.Adam(D.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))


def train_batch(x_r):
    batch_size = x_r.size(0)
    y_real = get_cuda(T.ones(batch_size))
    y_fake = get_cuda(T.zeros(batch_size))

    #Extract latent_z corresponding to real images
    z, kld = E(x_r)
    kld = kld.mean()
    #Extract fake images corresponding to real images
    x_f = G(z)

    #Extract latent_z corresponding to noise
    z_p = T.randn(batch_size, opt.n_z)
    z_p = get_cuda(z_p)
    #Extract fake images corresponding to noise
    x_p = G(z_p)

    #Compute D(x) for real and fake images along with their features
    ld_r, fd_r = D(x_r)
    ld_f, fd_f = D(x_f)
    ld_p, fd_p = D(x_p)

    #------------D training------------------
    loss_D = F.binary_cross_entropy(ld_r, y_real) + 0.5*(F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
    D_trainer.zero_grad()
    loss_D.backward(retain_graph = True)
    D_trainer.step()

    #------------E & G training--------------

    #loss corresponding to -log(D(G(z_p)))
    loss_GD = F.binary_cross_entropy(ld_p, y_real)
    #pixel wise matching loss and discriminator's feature matching loss
    loss_G = 0.5 * (0.01*(x_f - x_r).pow(2).sum() + (fd_f - fd_r.detach()).pow(2).sum()) / batch_size

    E_trainer.zero_grad()
    G_trainer.zero_grad()
    (opt.w_kld*kld+opt.w_loss_g*loss_G+opt.w_loss_gd*loss_GD).backward()
    E_trainer.step()
    G_trainer.step()


    return loss_D.item(), loss_G.item(), loss_GD.item(), kld.item()

def load_model_from_checkpoint():
    global E, G, D, E_trainer, G_trainer, D_trainer
    checkpoint = T.load(save_path)
    E.load_state_dict(checkpoint['E_model'])
    G.load_state_dict(checkpoint['G_model'])
    D.load_state_dict(checkpoint['D_model'])
    E_trainer.load_state_dict(checkpoint['E_trainer'])
    G_trainer.load_state_dict(checkpoint['G_trainer'])
    D_trainer.load_state_dict(checkpoint['D_trainer'])
    return checkpoint['epoch']

def training():
    start_epoch = 0
    if opt.resume_training:
        start_epoch = load_model_from_checkpoint()

    for epoch in range(start_epoch, opt.epochs):
        E.train()
        G.train()
        D.train()

        T_loss_D = []
        T_loss_G = []
        T_loss_GD = []
        T_loss_kld = []

        for x, _ in train_loader:
            x = get_cuda(x)
            loss_D, loss_G, loss_GD, loss_kld = train_batch(x)
            T_loss_D.append(loss_D)
            T_loss_G.append(loss_G)
            T_loss_GD.append(loss_GD)
            T_loss_kld.append(loss_kld)


        T_loss_D = np.mean(T_loss_D)
        T_loss_G = np.mean(T_loss_G)
        T_loss_GD = np.mean(T_loss_GD)
        T_loss_kld = np.mean(T_loss_kld)

        print("epoch:", epoch, "loss_D:", "%.4f"%T_loss_D, "loss_G:", "%.4f"%T_loss_G, "loss_GD:", "%.4f"%T_loss_GD, "loss_kld:", "%.4f"%T_loss_kld)

        generate_samples("data/results/%d.jpg" % epoch)
        T.save({
            'epoch': epoch + 1,
            "E_model": E.state_dict(),
            "G_model": G.state_dict(),
            "D_model": D.state_dict(),
            'E_trainer': E_trainer.state_dict(),
            'G_trainer': G_trainer.state_dict(),
            'D_trainer': D_trainer.state_dict()
        }, save_path)


def generate_samples(img_name):
    z_p = T.randn(opt.n_samples, opt.n_z)
    z_p = get_cuda(z_p)
    E.eval()
    G.eval()
    D.eval()
    with T.autograd.no_grad():
        x_p = G(z_p)
    utils.save_image(x_p.cpu(), img_name, normalize=True, nrow=6)



if __name__ == "__main__":
    if opt.to_train:
        training()
    else:
        checkpoint = T.load(save_path)
        G.load_state_dict(checkpoint['G_model'])
        generate_samples("data/testing_img.jpg")