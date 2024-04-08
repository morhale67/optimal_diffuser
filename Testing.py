import torch
from torch import nn
from Model import breg_rec
import wandb
from LogFunctions import print_and_log_message
import math
from OutputHandler import save_orig_img, save_outputs, calc_cumu_ssim_batch, save_randomize_outputs, calc_cumu_psnr_batch
import numpy as np
from Training import loss_function


def test_net(epoch, network, loader, device, log_path, folder_path, batch_size, img_dim, n_masks,
             wb_flag, save_img=False):
    network.eval()
    network.to(device)
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
        cumu_loss += loss.item()
        torch.cuda.empty_cache()  # Before starting a new forward/backward pass
        batch_psnr = calc_cumu_psnr_batch(reconstruct_imgs_batch, sim_object, pic_width)
        batch_ssim = calc_cumu_ssim_batch(reconstruct_imgs_batch, sim_object, pic_width)
        cumu_psnr += batch_psnr
        cumu_ssim += batch_ssim
        print_and_log_message(f"Epoch number {epoch}, batch number {batch_index}/{n_batchs}:"
                              f"       batch loss {loss.item()}", log_path)
        if wb_flag:
            wandb.log({"test_loss_batch": loss.item(), "test_psnr_batch": batch_psnr / batch_size,
                       "test_ssim_batch": batch_ssim / batch_size, "test_batch_index": batch_index})
        if save_img:
            save_randomize_outputs(epoch, batch_index, reconstruct_imgs_batch, sim_object, int(math.sqrt(img_dim)),
                                   folder_path, 'test_images', wb_flag)


    test_loss, test_psnr, test_ssim = cumu_loss / n_samples, cumu_psnr / n_samples, cumu_ssim / n_samples

    if wb_flag:
        wandb.log({"epoch": epoch, 'test_loss': test_loss})

    return test_loss, test_psnr, test_ssim
