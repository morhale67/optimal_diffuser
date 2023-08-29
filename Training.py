import torch
from torch import nn
import wandb
import math
import torchvision.datasets as dset
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
from Model import Gen
from Model import breg_rec
from LogFunctions import print_and_log_message
from OutputHandler import save_outputs
from DataFunctions import get_data
from testers import check_diff
from testers import compare_buckets
from Model import Gen_no_batch
import Model


def build_dataset(batch_size, num_workers, pic_width, data_root, Medical=False):
    if Medical:
        train_loader, test_loader = get_data(batch_size=batch_size, pic_width=pic_width, num_workers=num_workers,
                                             data_root=data_root)
    else:
        transform = tr.Compose([
            tr.ToTensor(),
            tr.Resize((pic_width, pic_width))
        ])

        train_dataset = dset.MNIST(data_root, train=True, download=True, transform=transform)

        indices = torch.arange(20000)
        mnist_20k = data.Subset(train_dataset, indices)

        train_dataset_size = int(len(mnist_20k) * 0.8)
        test_dataset_size = int(len(mnist_20k) * 0.2)
        # val_dataset_size = int(len(mnist_20k) - train_dataset_size)
        train_set, test_set = data.random_split(mnist_20k, [train_dataset_size, test_dataset_size])

        train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader


def build_network(z_dim, img_dim, n_masks, device, ac_stride=5):
    network = Model.Gen_big_diff(z_dim, img_dim, n_masks, ac_stride)
    # # Use DataParallel to wrap your model for multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     network = nn.DataParallel(network)
    torch.cuda.empty_cache()  # Before starting a new forward/backward pass
    return network.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer


def make_masks_from_big_diff(diffuser, ac_stride):
    """ diffuser shape: [pic_width, diff_width]"""
    batch_size, pic_width, diff_width = diffuser.shape
    img_dim = pic_width**2

    mask_indices = torch.arange(0, diff_width-pic_width, ac_stride, dtype=torch.long).view(-1, 1)
    diffuser_expanded = diffuser[:, :, mask_indices + torch.arange(pic_width)]
    masks = diffuser_expanded.view(batch_size, -1, img_dim)
    return masks


def train_epoch(epoch, network, loader, optimizer, batch_size, z_dim, img_dim, n_masks, device, log_path, folder_path,
                ac_stride=5, save_img=False, big_diffuser=True):
    cumu_loss = 0
    network.train()

    for batch_index, sim_bucket_tensor in enumerate(loader):
        # with torch.autograd.set_detect_anomaly(True):
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        sim_object = sim_object.to(device)
        if big_diffuser:
            pic_width = int(math.sqrt(img_dim))
            noise = torch.randn(int(batch_size), int(z_dim), requires_grad=True).to(device)
            diffuser = network(noise)
            diffuser = diffuser.reshape(batch_size, pic_width, -1)
            diffuser = make_masks_from_big_diff(diffuser, ac_stride)
        else:
            noise = torch.randn(int(batch_size), int(z_dim), requires_grad=True).to(device)
            diffuser = network(noise)
            diffuser = diffuser.reshape(batch_size, n_masks, img_dim)
            # diffuser = diffuser.unsqueeze(0).expand(batch_size, -1, -1)  # new tensor that contains repeated copies of the original tensor's data


        check_diff(diffuser, sim_object)

        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(diffuser, sim_object)
        sim_bucket = torch.transpose(sim_bucket, 1, 2)

        # buckets_same = compare_buckets(sim_bucket[0], diffuser[0], sim_object[0])

        reconstruct_imgs_batch = breg_rec(diffuser, sim_bucket, batch_size).to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(reconstruct_imgs_batch, sim_object)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumu_loss += loss.item()
        torch.cuda.empty_cache()  # Before starting a new forward/backward pass
        try:
            wandb.log({"batch loss": loss.item()})
            print(f"batch loss {batch_index}/{len(loader.batch_sampler)}: {loss.item()}, epoch {epoch}")
        except:
            print_and_log_message(f"batch loss {batch_index}/{len(loader.batch_sampler)}: {loss.item()}", log_path)

    train_loss = cumu_loss / len(loader)
    try:
        wandb.log({'train_loss': train_loss})
    except Exception as e:
        pass

    if save_img and epoch % 5 == 0:
        try:
            num_images = reconstruct_imgs_batch.shape[0]  # most of the time = batch_size
            pic_width = int(math.sqrt(img_dim))
            image_reconstructions = [wandb.Image(i.reshape(pic_width, pic_width)) for i in reconstruct_imgs_batch]
            sim_object_images = [wandb.Image(i.reshape(pic_width, pic_width)) for i in sim_object]

            wandb.log({'sim_diffuser': [wandb.Image(i) for i in diffuser]})
            wandb.log({'train image reconstructions': image_reconstructions})
            wandb.log({'train original images': sim_object_images})
        except:
            save_outputs(reconstruct_imgs_batch, sim_object, int(math.sqrt(img_dim)), folder_path,
                         'train_images')

    return train_loss

