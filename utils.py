import os
import sys
import shutil
import json
from datetime import datetime

import numpy as np
import torch
from prettytable import PrettyTable

from tensorboardX import SummaryWriter
from subprocess import check_call

import ast

########################
# param loading helper functions
#########################

class param2args(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def update_args(args, update_dict):
    for key in update_dict:
        setattr(args, key, update_dict[key])
    return args

def parse_params(param_filename):
    file = open(param_filename, "r")
    contents = file.read()
    params = ast.literal_eval(contents)
    file.close()
    args = param2args(params)
    return args 

def parse_params_wremove(param_filename, removelist): # remove parameters in removelist
    file = open(param_filename, "r")
    contents = file.read()
    itemlist = contents.split('\n')
    removelist = set(removelist)
    itemlist = [s for s in itemlist if not any(w in s for w in removelist)]
    contents = '\n'.join(itemlist)
#     if contents[-1] != '}' and contents[-1] != '\n':
    if contents[-1] != '}':
        if contents[-1] == '\n':
            if contents[-2] == '}':
                contents = contents[:-1]
            else:
                contents = contents+'}'
    params = ast.literal_eval(contents)
    file.close()
    args = param2args(params)
    return args 

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

        
########################
# exp logging helper functions
#########################

def set_writer(log_path, comment='', restore=False):
    """ setup a tensorboardx summarywriter """
    current_time = datetime.now().strftime('%b%d_%H%M')
    if restore:
        log_path = os.path.dirname(log_path)
    else:
        if comment:
            log_path = os.path.join(log_path, current_time + '_' + comment)
        else:
            log_path = os.path.join(log_path, current_time)
    writer = SummaryWriter(log_dir=log_path)
    return writer, log_path


def save_checkpoint(state, checkpoint, is_best=None, quiet=False):
    """ saves model and training params at checkpoint + 'last.pt'; if is_best also saves checkpoint + 'best.pt'

    args
        state -- dict; with keys model_state_dict, optimizer_state_dict, epoch, scheduler_state_dict, etc
        is_best -- bool; true if best model seen so far
        checkpoint -- str; folder where params are to be saved
    """

    filepath = os.path.join(checkpoint, 'state_checkpoint.pt')
    if not os.path.exists(checkpoint):
        if not quiet:
            print('Checkpoint directory does not exist Making directory {}'.format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)

#    if is_best:
#        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_state_checkpoint.pt'))

    if not quiet:
        print('Checkpoint saved.')


def load_checkpoint(checkpoint, models, optimizers=None, scheduler=None, best_metric=None, map_location='cpu'):
    """ loads model state_dict from filepath; if optimizer and lr_scheduler provided also loads them

    args
        checkpoint -- string of filename
        model -- torch nn.Module model
        optimizer -- torch.optim instance to resume from checkpoint
        lr_scheduler -- torch.optim.lr_scheduler instance to resume from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise('File does not exist {}'.format(checkpoint))

    checkpoint = torch.load(checkpoint, map_location=map_location)
    models = [m.load_state_dict(checkpoint['model_state_dicts'][i]) for i, m in enumerate(models)]

    if optimizers:
        try:
            optimizers = [o.load_state_dict(checkpoint['optimizer_state_dicts'][i]) for i, o in enumerate(optimizers)]
        except KeyError:
            print('No optimizer state dict in checkpoint file')

    if best_metric:
        try:
            best_metric = checkpoint['best_val_acc']
        except KeyError:
            print('No best validation accuracy recorded in checkpoint file.')

    if scheduler:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except KeyError:
            print('No lr scheduler state dict in checkpoint file')

    return checkpoint['epoch']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    
def get_bestmodel_paths(path_results):
    path_bestfile_list = []
    path_expname_list = []
    for dir_name in os.listdir(path_results):
        path_filedir = '/'.join([path_results, dir_name])
        path_expname_list.append(dir_name[11:])
        epoch = 0
        for file_name in os.listdir(path_filedir):
            if file_name.startswith("best_model"):
                if int(file_name.split('_')[2][5:]) > epoch:
                    epoch = int(file_name.split('_')[2][5:])
                    bestfile_name = file_name
        path_bestfile = '/'.join([path_filedir, bestfile_name])
        path_bestfile_list.append(path_bestfile)
    return path_bestfile_list, path_expname_list