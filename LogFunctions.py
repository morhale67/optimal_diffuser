import logging
import os
import time
from datetime import datetime


def generate_log_tensorborad(model_name, cr_name, run_name):
    logs_base_dir = os.path.join('logs', 'models')
    log_tb = os.path.join(logs_base_dir, model_name, f'cr_{cr_name}', run_name)
    log_cr = os.path.join(logs_base_dir, model_name, cr_name)
    return log_tb, log_cr


def print_run_info_to_log(p, run_number, folder_path='Logs/'):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    log_name = f"Log_{dt_string}.log"
    print(f'Name of log file: {log_name}')
    log_path = folder_path + '/' + log_name

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.info(f"Number of patterns: {p['m_patterns']} which is {p['prc_patterns']}% of {p['pic_width']}^2")
    # logger.info(f'Number of gray levels in output image: {n_gray_levels}')
    # logger.info(f"Initial lr: {p['initial_lr']}, Division factor: {p['div_factor_lr']}, {p['num_dif_lr']} divisions with {p['n_epochs']} epochs for each")
    # # logger.info(f"lr_vector {lr_vector}, epochs_vector{epochs_vector}")
    # print_and_log_message(f"Number of samples in train is {p['num_train_samples']}", log_path)
    # print_and_log_message(f"Number of samples in test is {p['num_test_samples']}", log_path)

    logger.info(f"This is a summery of the run:")
    logger.info(f"Batch size for this run: {p['batch_size']}")
    logger.info(f"Size of original image: {p['pic_width']} X {p['pic_width']}")
    logger.info(f'number of masks: {p["n_masks"]}')
    logger.info(f'Compression ratio: {p["cr"]}')
    if p['learn_vec_lr']:
        logger.info(f'number of epochs per lr: {p["epochs_vec"]}')
        logger.info(f'diff learning rate in the run: {p["lr_vec"]}')
    else:
        logger.info(f'epochs : {p["epochs"]}')
        logger.info(f'one learning rate: {p["lr"]}')

    logger.info(f'optimizer: {p["optimizer"]}')
    logger.info(f'weight_decay: {p["weight_decay"]}')

    # logger.info(f'number of fully connected layers in model: {p["n_fc"]}')
    logger.info('***************************************************************************\n\n')
    run_name = f"run_number_{run_number}_bs_{p['batch_size']}wd_{p['weight_decay']}_lr_{p['lr']}"
    log_tb, log_cr = generate_log_tensorborad(p['model_name'], str(p['cr']), run_name)
    logs = [log_path, log_tb, log_cr]
    return logs, run_name


def print_and_log_message(message, log_path):
    logging.basicConfig(filename=log_path, format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    if isinstance(message, str):
        logger.warning(message)
        print(message)
    else:
        logger.exception(message)


def print_training_messages(epoch, train_loss, lr, start, log_path):
    end = time.time()
    print_and_log_message(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f}', log_path)
    print_and_log_message(f"Time for epoch {epoch + 1} : {round(end - start)} sec", log_path)
    # print_and_log_message(f'Average PSNR for epoch {epoch + 1} on training set is {avg_psnr:.6f}', log_path)
    print_and_log_message(f'lr for epoch {epoch + 1} is {lr:.5f}', log_path)

