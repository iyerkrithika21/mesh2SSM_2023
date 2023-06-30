import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from chamfer_distance import ChamferDistance



def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.model_type)
    
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.model_type, 'epochs')
    make_dir(logger_path)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.model_type, 'logger.log')
    log_fd = open(logger_file, 'a')

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return epochs_dir, log_fd, train_writer, val_writer



CD = ChamferDistance()


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.
    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2,_,_ = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0