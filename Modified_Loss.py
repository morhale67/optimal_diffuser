import numpy as np
import torch
from torch import nn
from Lasso import sparse_encode


def loss_function(diffuser, sim_object, n_masks, img_dim, device):
    criterion = nn.MSELoss()
    diffuser_batch = diffuser.reshape(-1, n_masks, img_dim)
    sim_object = sim_object.transpose(1, 2)
    sim_bucket = torch.matmul(diffuser_batch, sim_object)
    sim_bucket = torch.transpose(sim_bucket, 1, 2)

    reconstruct_imgs_batch = breg_rec(diffuser_batch.reshape(n_masks, img_dim), sim_bucket)
    sim_object = torch.squeeze(sim_object)
    loss = criterion(reconstruct_imgs_batch, sim_object)
    return loss, reconstruct_imgs_batch


def breg_rec(diffuser, bucket_batch):
    n_masks, img_dim = diffuser.shape[0], diffuser.shape[1]  # diffuser in shape (n_masks, img_dim)
    batch_size = len(bucket_batch)
    recs_container = torch.zeros(batch_size, img_dim)
    for rec_ind, sample in enumerate(bucket_batch):
        maxiter, niter_inner, alpha = 1, 1, 1
        rec = sparse_encode(bucket_batch[rec_ind].reshape(1, n_masks), diffuser, maxiter=maxiter, niter_inner=niter_inner, alpha=alpha,
                            algorithm='split-bregman')
        # plot_rec_image(rec, maxiter, niter_inner, alpha)
        recs_container = recs_container.clone()
        recs_container[rec_ind] = rec

    return recs_container
