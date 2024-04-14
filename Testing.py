import torch
from torch import nn
import wandb
from LogFunctions import print_and_log_message
import math
from OutputHandler import save_orig_img, save_outputs, calc_cumu_ssim_batch, save_randomize_outputs, \
    calc_cumu_psnr_batch
import numpy as np
from Modified_Loss import loss_function, breg_rec


def get_sb_params_batch(indices):
    sb_params_dict = {'maxiter': np.arange(1, 11),
                       'niter_inner': np.arange(1, 11),
                       'alpha': np.arange(0.1, 3, 0.2),
                       'total_variation_rate': 0.1 * (np.power(0.1, np.arange(10)))
                       }
    sb_params_batch = {key: sb_params_dict[key][index] for key, index in zip(sb_params_dict.keys(), indices)}

    return sb_params_batch


def get_avg_output_batch(diffuser, prob_vectors):
    diffuser_batch = torch.mean(diffuser, dim=0)
    sb_p_indices = []
    for prob_vector in prob_vectors:
        _, indice = torch.max(torch.mean(prob_vector, dim=0), 0)
        sb_p_indices.append(indice.item())
    return diffuser_batch, sb_p_indices


def modify_output(diffuser, prob_vectors):
    diffuser_batch, sb_p_indices = get_avg_output_batch(diffuser, prob_vectors)
    sb_params_batch = get_sb_params_batch(sb_p_indices)
    return diffuser_batch, sb_params_batch


def creat_diffuser_for_batch(trained_network, train_loader, img_dim, device):
    pic_width = int(math.sqrt(img_dim))
    for batch_index, sim_bucket_tensor in enumerate(train_loader):
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        input_sim_object = sim_object.view(-1, 1, pic_width, pic_width).to(device)
        diffuser, prob_vectors = trained_network(input_sim_object)
        diffuser_batch, sb_params_batch = modify_output(diffuser, prob_vectors)
        break
    return diffuser_batch, sb_params_batch


def test_diffuser(epoch, diffuser, sb_params, test_loader, device, log_path, folder_path, img_dim, n_masks, wb_flag,
                  save_img):
    cumu_loss, cumu_psnr, cumu_ssim = 0, 0, 0
    n_batchs = len(test_loader.batch_sampler)
    batch_size = test_loader.batch_size
    n_samples = n_batchs * batch_size
    pic_width = int(math.sqrt(img_dim))

    for batch_index, sim_bucket_tensor in enumerate(test_loader):
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        loss, reconstruct_imgs_batch = loss_function(diffuser, sb_params, sim_object, n_masks, img_dim, device)
        cumu_loss += loss.item()
        torch.cuda.empty_cache()
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


def test_net(epoch, trained_network, train_loader, test_loader, device, log_path, folder_path, img_dim, n_masks, wb_flag,
             save_img):
    diffuser, sb_params = creat_diffuser_for_batch(trained_network, train_loader, img_dim, device)
    test_loss, test_psnr, test_ssim = test_diffuser(epoch, diffuser, sb_params, test_loader, device, log_path,
                                                    folder_path, img_dim, n_masks, wb_flag, save_img)
    return test_loss, test_psnr, test_ssim
