# VAE-GAN-Pytorch
Generation of 128x128 bird images using VAE-GAN with additional feature matching loss.

## Model Description
Resnet18 based Encoder. Generator and Discriminator architectures are similar to that of DCGAN. Discriminator and Generator are trained with Heuristic non saturating loss. Encoder is trained with KL-Divergence loss to ensure latent 'z' generated is close to standard normal distribution. In addition, the combination of Encoder and Generator is trained with reconstruction loss and Discriminator's feature matching loss. 

## Prerequisites
* Python 2.7
* Pytorch 0.4.0
* Torchvision 0.2.1

## Data
Download CUB-2011 dataset from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and copy the ```images``` folder to ```data``` folder in the repository

## Running on GPUs
Enabled running on multiple GPUs. Edit cuda device numbers in ```main.py```

## Training
To train the model from the beginning, run:
```
python main.py
```
To resume training from a saved model, run:
```
python main.py --resume_training=True
```
Samples generated from noise at each epoch can be viewed at ```data/results``` folder

## Testing
To generate images from saved model, run:
```
python main.py --to_train=False
```
### Sample when weight of feature matching loss = 0.01 at epoch=300:
<img src="https://github.com/rohithreddy024/VAE-GAN-Pytorch/blob/master/imgs/sample_img.jpg" width="480">
