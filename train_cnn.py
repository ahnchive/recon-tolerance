from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import random
import numpy as np

import sys
import pprint
from loaddata import *
from utils import *
from ourmodel import ResNet50Encoder

class ConvNet(nn.Module):
    def __init__(self, encoder, outputs=16):
        super().__init__()
        if encoder == 'resnet50':
            self.enc = ResNet50Encoder()
            self.enc_feature_shape =  (2048, 7,7) #(128*16,7,7)#2048x1x1  
        else:
            raise NotImplementedError

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()   
        self.fc = nn.Linear(self.enc_feature_shape[0], outputs)

    def forward(self, x):
        x = self.enc(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
#         input = self.fc1(input)
#         input = self.dropout2(input)
#         input = self.fc2(input)
        x = F.log_softmax(x, dim=1)
        return x
    


def train(args, model, device, train_loader, optimizer, epoch, writer):
    print(f'Epoch {epoch}:')
         
    model.train()
    train_loss = 0
    
    if args.class_weight is not None:
        weight = torch.FloatTensor(args.class_weight).to(device)
        
    for batch_idx, (data, recon, target) in enumerate(train_loader):
        if len(target.size())>1:
            target = torch.argmax(target, dim=1) # change from one hot to integer index

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        

        loss = F.nll_loss(output, target, weight=weight)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    writer.add_scalar('Train/Loss', train_loss, epoch)
        
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


def test(args, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    
    if args.class_weight is not None:
        weight = torch.FloatTensor(args.class_weight).to(device)
        
    with torch.no_grad():
        for data, recon, target in test_loader:
            if len(target.size())>1:
                target = torch.argmax(target, dim=1) # change from one hot to integer index

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum', weight=weight).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    writer.add_scalar('Val/Loss', test_loss, epoch)
    writer.add_scalar('Val/Loss', test_acc, epoch)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Resnet Training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1 every 30 epochs)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    parser.add_argument('--cuda', '-c', help="cuda index", type= int, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--expname', type=str, default='test')
    parser.add_argument('--print', action='store_true', help="if true, just print model info, false continue training", default=False)
    
    parser.add_argument('--class_weight_path', help='class weights for weighted loss; json file format', type=str, default=None)

    args = parser.parse_args()
    
    args.output_dir = './results/imagenet-16/'
    args.restore_file = None
    
    
    if args.class_weight_path:
        import json
        with open(args.class_weight_path, 'r') as handle:
            args.class_weight = json.load(handle) # rotocol=pickle.HIGHEST_PROTOCOL
#             args.class_weight = torch.FloatTensor(args.class_weight).to(args.device)
    else:
        args.class_weight = None
    
    COMMENT = args.expname
    
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)
# seed for reproducibility
    def seed_torch(seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    seed_torch(args.seed)
    
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() and args.cuda is not None else 'cpu')
#     model = Net().to(device)
    model = ConvNet(encoder='resnet50', outputs=16).to(device)
    # load dataloader
    train_loader, test_loader = fetch_dataloader(args.task, args.batch_size, train=True)

#     train_kwargs = {'batch_size': args.batch_size}
#     test_kwargs = {'batch_size': args.test_batch_size}
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 1,
#                        'pin_memory': True,
#                        'shuffle': True}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)

#     transform=transforms.Compose([
#         transforms.ToTensor(),
# #         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     dataset1 = datasets.MNIST('../data', train=True, download=True,
#                        transform=transform)
#     dataset2 = datasets.MNIST('../data', train=False,
#                        transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    
    # set writer for tensorboard
    writer, current_log_path = set_writer(log_path = args.output_dir if args.restore_file is None else args.restore_file,
                        comment = COMMENT, 
                        restore = args.restore_file is not None) 
    args.log_dir = current_log_path
    
    # save used param info to writer and logging directory for later retrieval
    writer.add_text('Params', pprint.pformat(args.__dict__))
    with open(os.path.join(args.log_dir, 'params.txt'), 'w') as f:
        pprint.pprint(args.__dict__, f, sort_dicts=False)
        
    # set optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)    
    scheduler = StepLR(optimizer, step_size=30, gamma=args.gamma)
    
    if args.print:
        print('\n==> print model architecture')
        print(model)

        print('\n==> print model params')
        count_parameters(model)
    else:
        best_acc=0.
        path_best=None
        epoch_no_improve=0
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, writer)
            current_acc= test(args, model, device, test_loader, epoch, writer)
            scheduler.step()

            if (epoch%10==0) & (args.save_model):
                path_save = args.log_dir +f'/archive_epoch{epoch}_{current_acc:.4f}.pt'
                torch.save(model.state_dict(), path_save)

            if round(current_acc,4) > round(best_acc,4):
                best_acc= current_acc
                epoch_no_improve=0
                if path_best:
                    os.remove(path_best)
                path_best =args.log_dir +f'/best_epoch{epoch}_{current_acc:.4f}.pt'
                torch.save(model.state_dict(), path_best)
            else:
                epoch_no_improve+=1
            
            if epoch_no_improve >= 20:
                path_save = args.log_dir +f'/earlystop_epoch{epoch}_{current_acc:.4f}.pt'
                torch.save(model.state_dict(), path_save)
                status = f'===== EXPERIMENT EARLY STOPPED (no progress on val_acc for last 20 epochs) ===='
                writer.add_text('Status', status, epoch)
                print(status)
                break
                
            if epoch == args.epochs:
                torch.save(model.state_dict(), args.log_dir +f'/last_{epoch:d}_acc{val_acc:.4f}.pt')
                status = f'===== EXPERIMENT RAN TO THE END EPOCH ===='
                writer.add_text('Status', status, epoch)
                print(status)
                
        writer.close()
                
if __name__ == '__main__':
    main()