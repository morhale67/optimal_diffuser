import math
import wandb
from main_training import train
from Params import get_run_parameters
from LogFunctions import print_and_log_message
from LogFunctions import print_run_info_to_log
from OutputHandler import make_folder
from torch.utils.tensorboard import SummaryWriter
import traceback
from io import StringIO


def main():
    p = get_run_parameters()
    search_parameters(p)
    # spec_multiple_runs(p)
    # writer_cr = SummaryWriter(f"TB/cr_{p['cr']}", mode='a')
    # run_model(p, writer_cr)


def search_parameters(p):
    for lr in [0.01, 0.001, 0.0001, 0.00001]:
        p['lr'] = lr
        for bs in [8, 10, 15, 20, 25]:
            p['batch_size'] = bs
            for weight_decay in [1e-5, 1e-4, 1e-3]:
                p['weight_decay'] = weight_decay
                for cr in [1, 2, 5, 10]:
                    writer_cr = SummaryWriter(f"TB/cr_{p['cr']}")
                    p['cr'] = cr
                    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
                    run_model(p, writer_cr)
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
        writer_cr = SummaryWriter(f"TB/cr_{p['cr']}")
        p['lr'] = run[1]
        p['weight_decay'] = run[2]
        p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
        run_model(p, writer_cr)


def run_model(p, writer_cr):
    try:
        folder_path = make_folder(p)
        logs, run_name = print_run_info_to_log(p, folder_path)
        print_and_log_message(f'learning rate: {p["lr"]}', logs[0])
        writer_run = SummaryWriter(f"TB/cr_{p['cr']}/{run_name}")
        writers = [writer_cr, writer_run]
        train(p, logs, folder_path, writers)
    except Exception as e:
        trace_output = StringIO()
        traceback.print_exc(file=trace_output)
        error_message1 = f"Error occurred for this parameters. Exception: {str(e)}"
        error_message2 = trace_output.getvalue()
        print_and_log_message(error_message1, logs[0])
        print_and_log_message(error_message2, logs[0])
        print_and_log_message(folder_path, logs[0])



if __name__ == '__main__':
    main()

