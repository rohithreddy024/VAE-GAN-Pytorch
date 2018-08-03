import torch as T
import numpy as np

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

# def get_onehot_labels(labels, n_c):
#     batch_size = len(labels)
#     class_onehot = np.zeros((batch_size, n_c))
#     class_onehot[np.arange(batch_size), labels] = 1
#     class_onehot = T.from_numpy(class_onehot)
#     return class_onehot

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)