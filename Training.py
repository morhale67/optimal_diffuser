import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import wandb
import math
from Model import breg_rec
from LogFunctions import print_and_log_message, print_training_messages
from Testing import test_net
from main_training import update_numerical_outputs
from testers import check_diff
from testers import compare_buckets
from Model import Gen_no_batch
import Model
from testers import check_diff_ac
import matplotlib.pyplot as plt
from OutputHandler import save_orig_img, save_outputs, calc_cumu_ssim_batch, save_randomize_outputs, calc_cumu_psnr_batch


def get_sb_params_batch(prob_vector1, prob_vector2, prob_vector3, prob_vector4, i_batch):
    _, maxiter = torch.max(prob_vector1.data, 1)
    _, niter_inner = torch.max(prob_vector2.data, 1)
    maxiter += 1
    niter_inner += 1
    _, alpha_class = torch.max(prob_vector3.data, 1)
    _, tv_pow = torch.max(prob_vector4.data, 1)
    alpha = (alpha_class+1) * 0.2
    total_variation_rate = 0.1**(tv_pow+1)

    sb_params_batch = {'maxiter': maxiter[i_batch],
                       'niter_inner': niter_inner[i_batch],
                       'alpha': alpha[i_batch],
                       'total_variation_rate': total_variation_rate[i_batch]}
    return sb_params_batch


def train_epoch(epoch, network, loader, optimizer, batch_size, img_dim, n_masks, device, log_path, folder_path,
                wb_flag, save_img=False):
    network.train()
    cumu_loss, cumu_psnr, cumu_ssim = 0, 0, 0
    n_batchs = len(loader.batch_sampler)
    n_samples = n_batchs * batch_size
    pic_width = int(math.sqrt(img_dim))

    for batch_index, sim_bucket_tensor in enumerate(loader):
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        input_sim_object = sim_object.view(-1, 1, pic_width, pic_width).to(device)
        diffuser, prob_vector1, prob_vector2, prob_vector3, prob_vector4 = network(input_sim_object)

        loss, reconstruct_imgs_batch = loss_function(diffuser, prob_vector1, prob_vector2, prob_vector3, prob_vector4,
                                                     sim_object, batch_size, n_masks, img_dim, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

        optimizer.step()
        cumu_loss += loss.item()
        torch.cuda.empty_cache()

        batch_psnr = calc_cumu_psnr_batch(reconstruct_imgs_batch, sim_object, pic_width)
        batch_ssim = calc_cumu_ssim_batch(reconstruct_imgs_batch, sim_object, pic_width)
        cumu_psnr += batch_psnr
        cumu_ssim += batch_ssim
        print_and_log_message(f"Epoch number {epoch}, batch number {batch_index}/{n_batchs}:"
                              f"       batch loss {loss.item()}", log_path)
        if wb_flag:
            wandb.log({"Loss Batch": loss.item(), "PSNR Batch": batch_psnr/batch_size,
                       "SSIM Batch": batch_ssim/batch_size})
        if save_img:
            save_randomize_outputs(epoch, batch_index, reconstruct_imgs_batch, sim_object, int(math.sqrt(img_dim)),
                                   folder_path, 'train_images', wb_flag)

    train_loss, train_psnr, train_ssim = cumu_loss / n_samples, cumu_psnr / n_samples, cumu_ssim / n_samples
    if wb_flag:
        wandb.log({"Epoch": epoch, 'Train Loss': train_loss, 'Train PSNR': train_psnr, 'Train SSIM': train_ssim})
    #        try:
    #           num_images = reconstruct_imgs_batch.shape[0]  # most of the time = batch_size
    #          pic_width = int(math.sqrt(img_dim))
    #         image_reconstructions = [wandb.Image(i.reshape(pic_width, pic_width)) for i in reconstruct_imgs_batch]
    #        sim_object_images = [wandb.Image(i.reshape(pic_width, pic_width)) for i in sim_object]
    #       wandb.log({'sim_diffuser': [wandb.Image(i) for i in diffuser]})
    #      wandb.log({'train image reconstructions': image_reconstructions})
    #     wandb.log({'train original images': sim_object_images})

    return train_loss, train_psnr, train_ssim


def loss_function(diffuser, prob_vector1, prob_vector2, prob_vector3, prob_vector4, sim_object,
                  batch_size, n_masks, img_dim, device):
    criterion = nn.MSELoss()
    diffuser = diffuser.reshape(batch_size, n_masks, img_dim)
    diffuser = diffuser[0]
    sim_object = sim_object.transpose(1, 2)
    sim_bucket = torch.matmul(diffuser, sim_object)
    sim_bucket = torch.transpose(sim_bucket, 1, 2)

    sb_params_batch = get_sb_params_batch(prob_vector1, prob_vector2, prob_vector3, prob_vector4, 0)
    reconstruct_imgs_batch = breg_rec(diffuser, sim_bucket, batch_size, sb_params_batch).to(device)
    sim_object = torch.squeeze(sim_object)
    loss = criterion(reconstruct_imgs_batch, sim_object)
    return loss, reconstruct_imgs_batch


def hard_samples_extractor(trained_network, train_loader, cur_avg_loss, batch_size, n_masks, img_dim, device):
    pic_width = int(np.sqrt(img_dim))
    hard_examples = []
    with torch.no_grad():
        for sim_object, label in train_loader:
            sim_object = sim_object.view(-1, 1, img_dim).to(device)
            input_sim_object = sim_object.view(-1, 1, pic_width, pic_width).to(device)
            diffuser, prob_vector1, prob_vector2, prob_vector3, prob_vector4 = trained_network(input_sim_object)
            loss = loss_function(diffuser, prob_vector1, prob_vector2, prob_vector3, prob_vector4, sim_object,
                                 batch_size, n_masks, img_dim, device)
            if loss.item() > cur_avg_loss:
                hard_examples.append((sim_object, label))

    hard_loader = DataLoader(hard_examples, batch_size=train_loader.batch_size, shuffle=True)
    return hard_loader


def train_hard_samples(network, hard_loader, test_loader, lr, params, optimizer, device, log_path, folder_path, wb_flag,
                       numerical_outputs, num_extra_epochs=10):
    """Train the model on the hard examples"""
    for epoch in range(num_extra_epochs):
        start_epoch = time.time()
        train_loss_epoch, train_psnr_epoch, train_ssim_epoch = train_epoch(epoch, network, hard_loader, optimizer,
                                                                           params['batch_size'], params['img_dim'],
                                                                           params['n_masks'], device, log_path,
                                                                           folder_path,
                                                                           wb_flag, save_img=True)
        print_training_messages(epoch, train_loss_epoch, lr, start_epoch, log_path)
        test_loss_epoch, test_psnr_epoch, test_ssim_epoch = test_net(epoch, network, test_loader, device, log_path,
                                                                     folder_path, params['batch_size'],
                                                                     params['img_dim'], params['n_masks'], wb_flag,
                                                                     save_img=True)
        numerical_outputs = update_numerical_outputs(numerical_outputs, train_loss_epoch, test_loss_epoch,
                                                     train_psnr_epoch,
                                                     test_psnr_epoch, train_ssim_epoch, test_ssim_epoch)

    return numerical_outputs
