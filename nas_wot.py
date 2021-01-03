import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

from weight_init import weight_init

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


def get_batch_jacobian(network, images, labels):
    network.zero_grad()

    images.requires_grad_(True)

    y = network(images)

    y.backward(torch.ones_like(y))
    jacob = images.grad.detach()

    return jacob, images.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


def score(network, train_loader, num_run):
    scores = []
    for _ in range(num_run):
        network.apply(weight_init)

        batch = next(iter(train_loader))
        images, labels = batch
        
        jacobs, labels = get_batch_jacobian(network, images, labels)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

        s = eval_score(jacobs, labels)
        scores.append(s)
    return sum(scores)/len(scores)