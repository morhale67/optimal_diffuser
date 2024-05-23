import math
import wandb


def get_run_parameters():
    p = {'AWS_machine': True,
         'data_name': 'stl-10',
         'model_name': 'ResConv',
         'TV_beta': 0.5,
         'lr_vec': [0.1, 0.01, 0.001],
         'epochs_vec': [5, 30, 5],
         'learn_vec_lr': False,
         'pic_width': 96,
         'n_samples': 2000,
         'cr': 1,
         'batch_size': 10,
         'lr': 0.001,
         'epochs': 100,
         'optimizer': 'adam',
         'weight_decay': 5e-7,
         'num_workers': 4,
         'z_dim': 100}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
    if p['learn_vec_lr']:
        p['epochs'] = sum(p['epochs_vec'])
        p['cum_epochs'] = [sum(p['epochs_vec'][:i + 1]) for i in range(len(p['epochs_vec']))]
    return p


def load_config_parameters(p):
    wandb.init()
    config = wandb.config
    p['batch_size'] = config.batch_size
    p['pic_width'] = config.pic_width
    p['z_dim'] = config.z_dim
    p['weight_decay'] = config.weight_decay
    p['TV_beta'] = config.TV_beta
    p['cr'] = config.cr

    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
    if p['learn_vec_lr']:
        p['epochs'] = sum(p['epochs_vec'])
        p['cum_epochs'] = [sum(p['epochs_vec'][:i + 1]) for i in range(len(p['epochs_vec']))]
    return p

