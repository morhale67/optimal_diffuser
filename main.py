import math
import wandb
from main_training import train
from Params import get_run_parameters
from LogFunctions import print_and_log_message
from LogFunctions import print_run_info_to_log
from OutputHandler import make_folder
from Params import load_config_parameters
import traceback
from io import StringIO


def main(wb_flag=False):
    p = get_run_parameters()
    if wb_flag:
        wandb.login(key='8aec627a04644fcae0f7f72d71bb7c0baa593ac6')
        wandb.init()
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="Optimal Diffuser")
        wandb.agent(sweep_id, function=main, count=4)
        # p = load_config_parameters(p)
    else:
        # spec_multiple_runs(p)
        # search_parameters(p)
        run_model(p, False)


def search_parameters(p):
    for lr in [0.001, 0.0001, 0.00001]:
        p['lr'] = lr
        for bs in [8, 10, 15, 20, 25]:
            p['batch_size'] = bs
            for weight_decay in [1e-7, 5e-7, 10e-7]:
                p['weight_decay'] = weight_decay
                for cr in [1, 2, 5, 10]:
                    p['cr'] = cr
                    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
                    run_model(p, False)
    print('finished successfully')


def get_runs():
    """ cr, lr, weight_decay"""
    runs = [[1, 10**(-4), 10**(-6)],
            [1, 10**(-4), 10**(-5)],
            [1, 10**(-4), 10**(-4)],
            [1, 10**(-5), 10**(-6)],
            [1, 10**(-6), 10**(-6)],
            [2, 10**(-4), 10**(-6)],
            [2, 10**(-4), 10**(-5)],
            [2, 10**(-4), 10**(-4)],
            [5, 10 ** (-4), 10 ** (-6)],
            [5, 10 ** (-4), 10 ** (-5)],
            [5, 10 ** (-4), 10 ** (-4)],
            [10, 10**(-3), 10**(-7)],
            [10, 10**(-3), 10**(-6)],
            [10, 10**(-3), 10**(-5)]]
    return runs


def spec_multiple_runs(p):
    runs = get_runs()
    n_runs = len(runs)
    for i, run in enumerate(runs):
        print(f'Run {i}/{n_runs}')
        p['cr'] = run[0]
        p['lr'] = run[1]
        p['weight_decay'] = run[2]
        p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
        run_model(p, False)



def run_model(p, wb_flag):
    try:
        folder_path = make_folder(p)
        log_path = print_run_info_to_log(p, folder_path)
        print_and_log_message(f'learning rate: {p["lr"]}', log_path)
        train(p, log_path, folder_path, wb_flag)
    except Exception as e:
        trace_output = StringIO()
        traceback.print_exc(file=trace_output)
        error_message1 = f"Error occurred for this parameters. Exception: {str(e)}"
        error_message2 = trace_output.getvalue()
        print_and_log_message(error_message1, log_path)
        print_and_log_message(error_message2, log_path)
        print_and_log_message(folder_path, log_path)


if __name__ == '__main__':
    main()


# sweep_configuration = {
#     "method": "bayes",
#     "metric": {
#         "goal": "minimize",
#         "name": "Wall Time"
#     },
#     "parameters": {
#         "batch_size": {
#             "values": [2, 4]
#         },
#         "pic_width": {
#             "values": [28, 32]
#         },
#         "z_dim": {
#             "values": [32, 128, 256]
#         },
#         "weight_decay": {
#             "values": [1e-7, 5e-7, 10e-7]
#         },
#         "TV_beta": {
#             "values": [0.1, 0.5, 1.0, 10, 100]
#         },
#         "cr": {
#             "values": [2, 5, 10]
#         }
#     }
# }

