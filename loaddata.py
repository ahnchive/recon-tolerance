# --------------------
# Data
# --------------------
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

import random
import numpy as np


from PIL import Image, ImageFilter

from torchvision import datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
    def __init__(self, ann_file, transform=None, target_transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.target_transform = target_transform
        self.init()
        
    
    def init(self):
        
        self.im_names = []
        self.targets = []
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(int(data[1]))

    def __getitem__(self, index):
        im_name = self.im_names[index]
        recon_name = self.im_names[index].replace("imagenet-16", "imagenet-16-recon") ###############
        target = self.targets[index]

        img = Image.open(im_name).convert('RGB') 
        recon = Image.open(recon_name).convert('RGB') 
        
        if img is None:
            print(im_name)
        if recon is None:
            print(recon_name)
            
        if self.transform is not None:
            img, recon = self.transform(img, recon)
        if self.target_transform is not None:
            target =self.target_transform(target)

        return img, recon, target

    def __len__(self):
        return len(self.im_names)

    
def train_loader(imgpath_list, batch_size, n_worker, n_class, parallel=False):

#     augmentation = [
#         T.RandomResizedCrop(224, scale=(0.2, 1.)),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#     ]

#     train_trans = T.Compose(augmentation)

    def train_trans(image, recon):
        # Resize
        resize = T.Resize(size=(256, 256))
        image = resize(image)
        recon = resize(recon)

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        recon = TF.crop(recon, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            recon = TF.hflip(recon)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            recon = TF.vflip(recon)

        # Transform to tensor
        image = TF.to_tensor(image)
        recon = TF.to_tensor(recon)
        
        return image, recon
        
    train_dataset = ImageDataset(imgpath_list, 
                                 transform=train_trans, 
                                 target_transform = T.Lambda(lambda y: torch.zeros(n_class, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))   
    
    if parallel:
        rank = None
        num_replicas =None
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset,
                            rank= rank,
                            num_replicas= num_replicas,
                            shuffle=True)         
    else:  
        train_sampler = None    

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=(train_sampler is None),
                    batch_size= batch_size,
                    num_workers=n_worker,
                    pin_memory=True,
                    sampler=train_sampler,
                    drop_last=(train_sampler is None))

    return train_loader

def val_loader(imgpath_list, batch_size, n_worker, n_class):

    
    def val_trans(image, recon):
        # Resize
        resize = T.Resize(size=(256, 256))
        image = resize(image)
        recon = resize(recon)

        # Center crop
        crop = T.CenterCrop(224)
        image = crop(image)
        recon = crop(recon)

        # Transform to tensor
        image = TF.to_tensor(image)
        recon = TF.to_tensor(recon)
        
        return image, recon
    
#     val_trans = T.Compose([
#                     T.Resize(256),                   
#                     T.CenterCrop(224),
#                     T.ToTensor()
#                   ])

    val_dataset = ImageDataset(imgpath_list, 
                               transform=val_trans, 
                               target_transform = T.Lambda(lambda y: torch.zeros(n_class, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))      

    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=batch_size,
                    num_workers=n_worker,
                    pin_memory=True)

    return val_loader 
        

def fetch_dataloader(task, batch_size, train=True):
    """
    load dataset depending on the task
    currently implemented tasks:
        -mnist
        -cifar10

    args
        -args
        -batch size
        -train: if True, load train dataset, else test dataset
    """
    ################
    # load taskwise dataset
    ################
    n_worker = 1
    if task == 'imagenet-16':
        if train:
            train_dataloader =  train_loader('./datasource/imagenet-16-train_list.txt', batch_size, n_worker, n_class=16, parallel=False)
            val_dataloader = val_loader('./datasource/imagenet-16-val_list.txt', batch_size, n_worker, n_class=16)
            return train_dataloader, val_dataloader
        else:
            return val_loader('./datasource/imagenet-16-val_list.txt', batch_size, n_worker, n_class=16)
            
        
    
 
        
