import torch
from torch import nn
from Model import breg_rec
import wandb
from LogFunctions import print_and_log_message
import math
from OutputHandler import save_orig_img, save_outputs, calc_cumu_ssim_batch, save_randomize_outputs, calc_cumu_psnr_batch
import numpy as np


def test_net(epoch, model, loader, device, log_path, folder_path, batch_size, z_dim, img_dim, cr, epochs,
             TV_beta, save_img=False):
    model.eval()
    model.to(device)
    cumu_loss, cumu_psnr, cumu_ssim = 0, 0, 0
    n_batchs = len(loader.batch_sampler)
    n_samples = n_batchs * batch_size
    pic_width = int(math.sqrt(img_dim))

    for batch_index, sim_bucket_tensor in enumerate(loader):
        sim_object, _ = sim_bucket_tensor
        sim_object.to(device)
        noise = torch.randn(batch_size, z_dim, requires_grad=True).to(device)
        sim_diffuser = model(noise)

        sim_diffuser_reshaped = sim_diffuser.reshape(batch_size, math.floor(img_dim / cr), img_dim)

        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(sim_diffuser_reshaped, sim_object)
        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        reconstruct_imgs_batch = breg_rec(sim_diffuser_reshaped, sim_bucket, batch_size, beta=TV_beta)

        reconstruct_imgs_batch = reconstruct_imgs_batch.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(reconstruct_imgs_batch, sim_object)
        cumu_loss += loss.item()
        cumu_psnr += calc_cumu_psnr_batch(reconstruct_imgs_batch, sim_object, pic_width)
        cumu_ssim += calc_cumu_ssim_batch(reconstruct_imgs_batch, sim_object, pic_width)
        if save_img:
            save_randomize_outputs(epoch, batch_index, reconstruct_imgs_batch, sim_object, pic_width, folder_path,
                                   'test_images')

    test_loss, test_psnr, test_ssim = cumu_loss / n_samples, cumu_psnr / n_samples, cumu_ssim / n_samples

   # try:
   #     pic_width = int(math.sqrt(img_dim))
   #     image_reconstructions = [wandb.Image(i.reshape(pic_width, pic_width)) for i in reconstruct_imgs_batch]
   #     sim_object_images = [wandb.Image(i.reshape(pic_width, pic_width)) for i in sim_object]
   #     wandb.log({'sim_diffuser': [wandb.Image(i) for i in sim_diffuser_reshaped]})
   #     wandb.log({'image reconstructions': image_reconstructions})
   #     wandb.log({'test original images': sim_object_images})
   #     wandb.log({'Test_loss': test_loss})
   #
   #     print(f"epoch [{epoch} / {epochs}] \ "
   #           f"genValLoss: {test_loss:.4f}")
   # except:
   #     print_and_log_message('Test Loss: {:.6f}\n'.format(test_loss), log_path)

    return test_loss, test_psnr, test_ssim
