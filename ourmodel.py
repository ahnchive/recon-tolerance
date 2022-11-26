# --------------------
# RRcapsnet
# --------------------
# code Refs:
# modified from https://github.com/XifengGuo/CapsNet-Pytorch

import os
import sys
import time
from tqdm import tqdm
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils import *
from loaddata import *


# --------------------
# functions & modules
# --------------------

def scale_image(x):
    '''channel-wise linear contrast stretching 
       same as minmax scaling among nonzero values
    '''
#         eps = torch.finfo(x.dtype).eps
    eps = 1e-6
    n_batch = x.size(0)
    n_channel = x.size(1)
    x_flatten = x.view(n_batch, n_channel, -1) 

    # if it's too small value replace with non-zero value
    x_nozero = x_flatten.clone()
    x_nozero[x_flatten<eps] = 1.0

    # get min and max pixel intensity 
    cmin = x_nozero.min(2, keepdim=True)[0] 
    cmax = x_flatten.max(2, keepdim=True)[0]
    x_flatten_new = (x_flatten-cmin+eps)/(cmax-cmin+eps) # epsilon to avoid nan, when all values are the same
    x_flatten_new = torch.clip(x_flatten_new,0,1)
    x_new = x_flatten_new.view(x.size())
#                     x = F.instance_norm(x)
    return x_new
      
def scale_coef(x, dim=1, vrange=(0.0, 1.0), num_bgcapsule_to_exclude=0):
    """
    [batch, n_objcaps, n_pricaps] or [batch, n_objcaps]
    normalize over specified dimension, output keeps original shape
    """
    
    if num_bgcapsule_to_exclude>0:
        # x narrow to include only objects
        x = x.narrow(dim=1,start=0, length=x.size(dim=1)-num_bgcapsule_to_exclude)

    # softmax
#     out = F.softmax(x, dim=dim)

    # minmax scale
    devmin = x - torch.min(x, dim=dim, keepdim=True)[0] 
    maxmin = torch.max(x, dim=dim, keepdim=True)[0] - torch.min(x, dim=dim, keepdim=True)[0]
    maxmin = torch.clip(maxmin, min=1e-6) # to avoid devision by zero
    out =  (vrange[1]- vrange[0])*devmin/maxmin +vrange[0]  

    # get max value index
#     onehot = torch.zeros_like(x)
#     _, idx = torch.max(x, dim=1)
#     if len(x.shape)==3:
#         mask = torch.arange(onehot.size(1)).reshape(1, -1, 1).to(x.device) == idx.unsqueeze(1)
#     elif len(x.shape)==2:
#         mask = torch.arange(onehot.size(1)).reshape(1, -1).to(x.device) == idx.unsqueeze(1)        
#     onehot[mask] =1
#     out = onehot


    if num_bgcapsule_to_exclude>0:
        # attach value for bgcapsule as zero
        shape = list(x.size())
        shape[1] =1 # change obj dimension value to 1
        bgx =  vrange[0]*torch.ones(shape, device=x.device)
        out = torch.cat([out, bgx], dim=1)
    
    out = torch.clip(out,min=0.5)
    
    return out
        
def compute_rscore(x, recon, method='error'):
    """
    compute reconstruction score (higher value, less recon error)
    output: rscore [batch,]
    """
    batch_size = x.shape[0]

    # recon = torch.clip(recon,0,1) #83125->8315
#     recon = scale_input(recon) #8369  

    if method == 'error':
        rerror = F.mse_loss(x, recon, reduction='none').view(batch_size,-1).sum(dim=1)
        rscore = torch.neg(rerror)
        
    elif method == 'multiply':
        rscore = (x*(recon>0.1)).view(batch_size,-1).sum(dim=1)
        rscore = (x*recon).view(batch_size,-1).sum(dim=1)
                
    elif method == 'explain-away':
        diff = recon - x
        diff[diff < 0] = 0 # ignore negative values, focus on the parts not explained yet 
        rerror = diff.view(batch_size,-1).sum(dim=1)/recon.view(batch_size,-1).sum(dim=1) # calculate percentage of not explained yet
        rscore = torch.neg(rerror)
        
    elif method == 'auroc':
        import sklearn.metrics as metrics
        rerror = []
        for x_i, recon_i in zip(x, recon):
            auc = metrics.roc_auc_score(x_i.to(device).view(-1).cpu().numpy(), recon_i.view(-1).cpu().numpy())
            rerror.append(auc)
        rerror = torch.from_numpy(np.array(rerror).astype('float32')).to(device)
        rscore = torch.neg(rerror)
    else:
        raise NotImplementedError
        
    return rscore

# get rscore over all possible objects; todo: subset of object hypothesis?
def get_every_obj_rscore(x_input, objcaps, Decoder, save_recon=False):
    '''
    x_input, and recon should be in the same shape
    '''
    n_obj = objcaps.shape[1] #objectcaps [n_batch, n_objects(class+bkg), dim_objectcaps]
    device = objcaps.device
    
    # get every object recon using onehot
    onehots = F.one_hot(torch.arange(0, n_obj)).to(device)
    
    obj_recon = []
    obj_rscore = []
    for n in range(n_obj): 
        y_hot = onehots[n].repeat(len(objcaps), 1)
        objcaps_one_capsule_hot = (objcaps * y_hot[:, :, None])
        recon, _ = Decoder(objcaps_one_capsule_hot)
        
        # save recon
        if save_recon:
            obj_recon.append(recon)
            
        # compute reconstruction score (higher value, less recon error)
        rscore = compute_rscore(x_input.to(device), recon)
        obj_rscore.append(rscore)

    # stack rscore
    obj_rscore = torch.stack(obj_rscore, dim=1) #torch.Size([batch, out_num_caps])
    
    # return outputs
    if save_recon:
        obj_recon = torch.stack(obj_recon, dim=1)
        return obj_rscore, obj_recon

    else:
        return obj_rscore
    
def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    
    return scale * inputs


class nnCapsulate(nn.Module):
    '''
    Note that input.size(0) is usually the batch size.
    Given any input with input.size(0) # of batches,
    features will be capsulated [batch, num_caps, dim_caps] + squash
    '''
    def __init__(self, num_caps, dim_caps):
        super().__init__()
        self.num_caps = num_caps
        self.dim_caps = dim_caps
    
    def forward(self, input):
        
        input_shape = input.size()
        batch_size = input_shape[0]
        
        if len(input_shape) == 4: ## BxCxHxW shape
            out = input.permute(0, 2, 3, 1).contiguous().view(batch_size, self.num_caps, self.dim_caps)       
        else:
            out = input.contiguous().view(batch_size, self.num_caps, self.dim_caps)     
        
        return squash(out) 

class nnReshape(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.H, self.W = height, width 

    def forward(self, x):
        return x.view(x.size(0), -1, self.H, self.W)
    
# class nnSquash(nn.Module):
#     '''
#     pytorch module version of squash
#     '''
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, input):
#         return squash(input) 


class AttentionWindow(nn.Module):
    """ if use_reconstruction_mask = True,
            - generate attention window using reconstruction mask, return x_mask, and masked x_input
        else:
            - attention window to entire x (no masking)
    """
    def __init__(self, use_reconstruction_mask, time_steps, mask_threshold):
        super().__init__()
        self.use_reconstruction_mask = use_reconstruction_mask
        if self.use_reconstruction_mask: 
            self.mask_type = 'bool'
            self.mask_threshold = mask_threshold
            self.mask_apply_method ='match'
            if time_steps>1: # top-down recon mask is only functional after 1step
                print(f'...use recon mask for attention: True')
                print(f'...with mask type {self.mask_type}, threshold {self.mask_threshold}, apply_method {self.mask_apply_method}')
                

    def forward(self, x, x_recon, timestep):
        
        if not self.use_reconstruction_mask or timestep==1:
            x_mask = torch.ones(x.shape, device = x.device)
            x_input = x
        else:
            ##########################
            # generate recon mask
            #########################
            if self.mask_type == 'raw':
                x_mask = x_recon
            elif self.mask_type == 'weight':
                x_mask = scale_image(x_recon)
            elif self.mask_type == 'bool':
                x_mask = x_recon
    #             blurrer = T.GaussianBlur(kernel_size=5, sigma=2)
    #             x_mask = blurrer(x_mask)
#                 x_mask_flatten = x_mask.view(x_recon.shape[0],-1)
#                 qtiles = torch.quantile(x_mask_flatten, 0.5, dim=1, keepdim=True)
#                 x_mask_flatten = x_mask_flatten>=qtiles
#                 x_mask = x_mask_flatten.view(x_recon.shape)

                x_mask = (x_mask>self.mask_threshold).float()
            else:
                raise NotImplementedError("given mask type is not implemented") 
 
            ##########################
            # apply mask to input
            #########################      
            if self.mask_apply_method == 'match':
                x_input = x*x_mask
                x_input = scale_image(x_input)
            elif self.mask_apply_method == 'mismatch':
                x_input = x*(1-x_mask)
                x_input = scale_image(x_input)
            elif self.mask_apply_method == 'subtract':
                x_input = x - x_mask
                x_input = scale_image(x_input)
            elif self.mask_apply_method == 'add':
                x_input = x + x_mask
                x_input = scale_image(x_input)
            else:
                raise NotImplementedError("given mask apply method is not implemented")                
            
        return x_mask, x_input



class Encoder(nn.Module):
    """ Encoding process is two steps:
        1) convolution: feature extraction (either using pretrained modle or train-from-scratch)
        2) encoder: transform/convert extracted features to capsulizable size (either recurrent or just one-step linear way)
        
        Input:
        - x = image # x.size = (x.shape[0], -1, self.read_size, self.read_size) # should on the device
        Outputs:
        - primary caps  [batch, num_caps, dim_caps] 
    """
    def __init__(self, encoder_type, projection_type, img_channels, dim_caps):
        super().__init__()
        self.encoder_type= encoder_type # use pretrained feature extractor
        self.projection_type = projection_type
        self.dim_caps = dim_caps
        
        # define encoder type
        if self.encoder_type == 'two-conv-layer':                        
            self.enc = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, stride=1, padding=0)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,  stride=1, padding=0)),
                ('relu2', nn.ReLU()),
#                 ('bn1', nn.BatchNorm2d(num_features=64)),
                ('pool', nn.MaxPool2d(kernel_size=2, padding=0)),
                ('encoder-dropout', nn.Dropout(0.25)),
            ]))
            self.enc_feature_shape =  (8*8, 12, 12) #(4*8, 11, 11)

        elif self.encoder_type == 'resnet':
            self.enc = ResNet50Encoder()
            self.enc_feature_shape =  (2048, 7,7) #(128*16,7,7)#2048x1x1  
                    
        else:
            raise NotImplementedError("the specified encoder not implemented yet")        
                         
                
                
        # define projection type
        if self.projection_type == None:
            enc_feature_size = self.enc_feature_shape[0]*self.enc_feature_shape[1]*self.enc_feature_shape[2]
            prj_feature_size = enc_feature_size
            self.projection_layer = nn.Identity()

        elif self.projection_type =='conv1x1':
            prj_width = 512
            prj_feature_size = prj_width*self.enc_feature_shape[1]*self.enc_feature_shape[2]
            self.projection_layer =  nn.Conv2d(self.enc_feature_shape[0], prj_width, kernel_size=1, stride=1, bias=False)
            
#         elif self.projection_type =='linear':
#             enc_feature_size = self.enc_feature_shape[0]*self.enc_feature_shape[1]*self.enc_feature_shape[2]
#             prj_feature_size = 64*self.enc_feature_shape[1]*self.enc_feature_shape[2] #3136
#             self.projection_layer =nn.Sequential(nn.Flatten(), 
#                                                  nn.Linear(enc_feature_size, prj_feature_size)) # TODO:activation function


#         elif self.projection_type =='rnn':
#             enc_feature_size = self.enc_feature_shape[0]*self.enc_feature_shape[1]*self.enc_feature_shape[2]
#             prj_feature_size = 360 # something small enough 
            
#             class EncoderRNNProject(nn.Module):
#                 def __init__(self, enc_feature_size, prj_feature_size):
#                     super().__init__()
#                     self.flatten = nn.Flatten()
#                     self.linear =  nn.Linear(enc_feature_size, prj_feature_size)
#                     self.rnn =  nn.GRUCell(prj_feature_size, prj_feature_size)

#                 def forward(self, x, hidden):
#                     x = self.flatten(x)
#                     x = self.linear(x)
#                     x = self.rnn(x, hidden)
#                     return x
    
#             self.projection_layer = EncoderRNNProject(enc_feature_size, prj_feature_size)


                
        else:
            raise NotImplementedError("the specified enc projection not implemented yet")  

            
        # encapsulation layer; compute final n_caps 
        if prj_feature_size % self.dim_caps == 0:
            self.num_caps = int(prj_feature_size/self.dim_caps)
        else:
            raise ValueError("feature size must be devided to dim_caps")
        self.capsulate = nnCapsulate(dim_caps=self.dim_caps, num_caps=self.num_caps)
        print(f"...resulting primary caps #: {self.num_caps}, dim: {self.dim_caps}")


    def forward(self, x, hidden_state=None):
        
        batch_size = x.size(0)
        features = self.enc(x)
        if self.projection_type =='rnn':
            hidden_state = self.projection_layer(features, hidden_state)
            pricaps = self.capsulate(hidden_state)
        else:
            features = self.projection_layer(features)
            pricaps = self.capsulate(features)
        
        
        return pricaps, hidden_state

class Decoder(nn.Module):
    """ Decoding process is two steps:
        1) decoder: transform/convert capsule features to original feature dimension (either recurrent or one-step linear)
        2) deconvolution: image reconstruction 
    
        Input:
        - object capsules # size  
        
        Output:
        - x_recon: image reconstruction
        - if decoder is recurrent, hidden_state and cell_state (for lstm)
    """
    def __init__(self, decoder_type, projection_type, outcaps_size, output_shape, enc_feature_shape):
        super().__init__()
        self.decoder_type = decoder_type
        self.projection_type = projection_type
        self.outcaps_size = outcaps_size # num_objcapsule * dim_objcapsule                 
        self.nfeature, self.fH, self.fW = enc_feature_shape # original enc features shape (before projection and encapsulation)
        self.C, self.H, self.W = output_shape # target output shape (=input shape)

        
        # define projection type 
        if self.projection_type == None:
            self.dec_feature_size = self.outcaps_size
            self.projection_layer = nn.Identity()

        elif self.projection_type == 'conv1x1':
            self.dec_feature_size = self.nfeature*self.fH*self.fW             
            self.projection_layer =  nn.Sequential(nnReshape(self.fH, self.fW),
                                                   nn.Conv2d(int(self.outcaps_size/(self.fH*self.fW)), 512, kernel_size=1, stride=1, bias=False),
                                                   nn.Conv2d(512, int(self.nfeature), kernel_size=1, stride=1, bias=False))

#         elif self.projection_type == 'linear':
#             self.dec_feature_size = self.nfeature*self.fH*self.fW             
#             self.projection_layer =nn.Sequential(nn.Flatten(), 
#                                                   nn.Linear(self.outcaps_size, self.dec_feature_size)) # TODO:activation function
#         elif self.projection_type =='rnn':
#             self.dec_feature_size = self.nfeature*self.fH*self.fW 
#             self.rnn_dec = nn.GRUCell(self.outcaps_size, self.outcaps_size)
#             self.linear_dec = nn.Linear(self.outcaps_size, self.dec_feature_size)
            
#             class DecoderRNNProject(nn.Module):
#                 def __init__(self, outcaps_size, dec_feature_size):
#                     super().__init__()
#                     self.flatten = nn.Flatten()
#                     self.rnn =  nn.GRUCell(outcaps_size, outcaps_size)
#                     self.linear =  nn.Linear(outcaps_size, dec_feature_size)

#                 def forward(self, x, hidden):
#                     x = self.flatten(x)
#                     hidden = self.rnn(x, hidden)
#                     x = self.linear(hidden)
#                     return x, hidden
    
#             self.projection_layer = DecoderRNNProject(self.outcaps_size, self.dec_feature_size)
        else:
            raise NotImplementedError("the specified dec projection not implemented yet")  
        
        # define decoder type
        if self.decoder_type == 'fcn':
            self.dec = nn.Sequential(OrderedDict([
                ('flatten', nn.Flatten()),
                ('delinear1', nn.Linear(self.dec_feature_size, 512)),
                ('derelu1', nn.ReLU()),
                ('decoder-dropout1', nn.Dropout(0.5)),
                ('delinear2', nn.Linear(512, 1024)),
                ('derelu2', nn.ReLU()),
                ('decoder-dropout2', nn.Dropout(0.5)),
                ('delinear3', nn.Linear(1024, self.C*self.H*self.W)),
#                 ('derelu3', nn.ReLU())
                ('sigmoid', nn.Sigmoid())
            ]))            
            
        elif self.decoder_type == 'resnet':
            self.dec = ResNet50Decoder(enc_feature_shape)
        else:
            raise NotImplementedError("the specified dec projection not implemented yet")              
    
    def forward(self, outcaps, hidden_state=None):     
        batch_size = outcaps.size(0)
        outcaps = outcaps.contiguous().view(batch_size, -1) # flatten all object capsules into a single vector

        # projection
        if self.projection_type == 'rnn':
            features, hidden_state = self.projection_layer(outcaps, hidden_state)
        else:
            features = self.projection_layer(outcaps) 
        
        # decode
        x_recon = self.dec(features)
        
        return x_recon.view(batch_size, self.C, self.H, self.W), hidden_state      




class ResNet50Encoder(nn.Module):
    '''
    Resnet50-like encoder for imagenet-16 task
    Code adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    '''

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

        configs = [3,4,6,3]

        self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=configs[0], downsample_method="pool")
        self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=configs[1], downsample_method="conv")
        self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=configs[2], downsample_method="conv")
        self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3], downsample_method="conv")


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class ResNet50Decoder(nn.Module):

    def __init__(self, enc_feature_shape):
        super().__init__()

        self.nfeature, self.fH, self.fW = enc_feature_shape


        configs = [3,4,6,3]

        self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0])
        self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=configs[1])
        self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=configs[2])
        self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = x.view(x.size(0),self.nfeature, self.fH, self.fW)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv"):
        super().__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

    
class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super().__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x
    
    
class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super().__init__()
    
        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x
    
    

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super().__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x
    
    
class PDReconCapsuleRouting(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings, num_bgcapsule):
        super().__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.1 * torch.randn(self.out_num_caps, self.in_num_caps, self.out_dim_caps, self.in_dim_caps))
        self.num_bgcapsule = num_bgcapsule
        
    def forward(self, incaps, x_input, Decoder):
        device = incaps.device
        batch_size = incaps.shape[0]
        
        # initialize coefficients
        self.b = torch.zeros(batch_size, self.out_num_caps, self.in_num_caps, device=device) # part-whole matching
        self.c = torch.ones(batch_size, self.out_num_caps, self.in_num_caps, device=device) # normalized b
        self.r = torch.ones(batch_size, self.out_num_caps, self.in_num_caps, device=device) # reconstruction scores 
        self.rc = torch.ones(batch_size, self.out_num_caps, self.in_num_caps, device=device) # reconstruction scores
        
        # incaps.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1, in_num_caps, in_dim_caps,  1]
        # weight.size   =[out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => outcaps_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        outcaps_hat = torch.squeeze(torch.matmul(self.weight, incaps[:, None, :, :, None]), dim=-1)
        
        # In forward pass, `outcaps_hat_detached` = `outcaps_hat`;
        # In backward, no gradient can flow from `outcaps_hat_detached` back to `outcaps_hat`.
        outcaps_hat_detached = outcaps_hat.detach()

        # keep all the coupling coefficients for all routing steps; todo: remove lists for visualization
        coups = []; betas=[]; rscores = []; recon_coups = []
        outcapslen = []; outcapslen_before = [] # before and after recon routing applied
        
        for i in range(self.routings):
            # record coups and betas
            coups.append(self.c.detach())
            betas.append(self.b.detach())
            rscores.append(self.r.detach())
            recon_coups.append(self.rc.detach())
            
            # Use `outcaps_hat_detached` to update `b`. No gradients flow on this path.       
            if i < self.routings-1: 
                outcaps = squash(torch.sum(self.rc[:, :, :, None] * outcaps_hat_detached, dim=-2, keepdim=True))
                outcapslen.append(outcaps.norm(dim=-1).squeeze(-1).detach())
                
                # for visualization
                outcaps_before = squash(torch.sum(self.c[:, :, :, None] * outcaps_hat_detached, dim=-2, keepdim=True))
                outcapslen_before.append(outcaps_before.norm(dim=-1).squeeze(-1).detach())
                # update beta and coupling coefficients
                # outcaps.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # outcaps_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => resulting b.size          =[batch, out_num_caps, in_num_caps]
                self.b = torch.sum(outcaps * outcaps_hat_detached, dim=-1) ## todo; whether accumulate?     
#                 self.b = torch.div(self.b, outcaps_before.norm(dim=-1))
#                 self.b = F.cosine_similarity(outcaps, outcaps_hat_detached, dim=-1)

                # use a MAX-MIN normalization to separate the coefficients (original was softmax)
                self.c = scale_coef(self.b, dim=1, num_bgcapsule_to_exclude=self.num_bgcapsule)

                # get recon score from current outcaps
                # objcaps.size =[batch, out_num_caps, out_dim_caps]
                objcaps = torch.squeeze(outcaps, dim=-2) 
                obj_rscore = get_every_obj_rscore(x_input, objcaps, Decoder, save_recon=False) #[batch, out_num_caps]
                obj_rscore = scale_coef(obj_rscore, dim=1, num_bgcapsule_to_exclude=self.num_bgcapsule)
                                
                # modulate coefficients based on recon error
                self.r = obj_rscore[:,:,None].repeat(1,1,self.in_num_caps) # tile to c.size, from [batch, out_num_caps] to [batch, out_num_caps,  in_num_caps]
                self.rc = self.c*self.r


            # At last iteration, use `outcaps_hat` to compute `outcaps` in order to backpropagate gradient
            elif i == self.routings-1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # outcaps_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outcaps.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outcaps = squash(torch.sum(self.rc[:, :, :, None] * outcaps_hat, dim=-2, keepdim=True))
                outcapslen.append(outcaps.norm(dim=-1).squeeze(-1).detach())
                
                # for visualization
                outcaps_before = squash(torch.sum(self.c[:, :, :, None] * outcaps_hat, dim=-2, keepdim=True))
                outcapslen_before.append(outcaps_before.norm(dim=-1).squeeze(-1).detach())
                
                # for debugging ###########################################################
                # get rscore over all possible object hypothesis; 
#                 objcaps = torch.squeeze(outcaps, dim=-2)
#                 obj_rscore = get_every_obj_rscore(x_input, objcaps, Decoder)
#                 obj_rscore = scale_coef(obj_rscore, dim=1, method='minmax', clip_under=self.min_rscore)

#                 # modulate coefficients based on recon error
#                 self.r = obj_rscore[:,:,None].repeat(1,1,self.in_num_caps) 
#                 rscores.append(self.r.detach())
                
                break

            

        outputs = {}
        outputs['coups'] = coups
        outputs['betas'] = betas
        
        outputs['rscores'] = rscores
        outputs['recon_coups'] = recon_coups
        outputs['outcaps_len'] = outcapslen
        outputs['outcaps_len_before'] =outcapslen_before
        
        outcaps_squeeze = torch.squeeze(outcaps, dim=-2)
        
        return outcaps_squeeze, outputs



#########################
# Model build and Training
#########################

class RRCapsNet(nn.Module):
    """
    todo: describe this model
    This model 
        1) encodes an image through recurrent read operations that forms a spatial attention on objects
        2) effectively binds features and classify objects through capsule representation and its dynamic routing
        3) decodes/reconstructs an image from class capsules through recurrent write operations 
    """
        
    def __init__(self, args):
        super().__init__()
        
        print(f'\n=========== model instantiated like below: =============')
        # dataset info
        print(f'TASK: {args.task} (# targets: {args.num_targets}, # classes: {args.num_classes}, # background: {args.backg_objcaps})')
        self.task = args.task
        self.C, self.H, self.W = args.image_dims
        self.image_dims = args.image_dims
        self.num_targets = args.num_targets # the number of objects in the image
        self.num_classes = args.num_classes # number of categories 
        
        # time steps
        print(f'TIMESTEPS #: {args.time_steps}')
        self.time_steps = args.time_steps # the number of complete cycle of encoder-decoder 
        
#         print(f'use RNN?: {args.use_rnn}')
#         self.use_rnn= args.use_rnn # whether use rnn for decoder and encoder
#         if self.time_steps < 2:
#             self.use_rnn = False
#         elif self.time_steps >= 2:
#             self.use_rnn = True

        
        # encoder (given image --> pricaps)
        print(f'ENCODER: {args.encoder_type} w/ {args.enc_projection_type} projection')
        self.encoder = Encoder(encoder_type= args.encoder_type,
                               projection_type = args.enc_projection_type,
                               img_channels= self.C,
                               dim_caps = args.dim_pricaps,
                              )
        

        # capsule network (pricaps --> objcaps)
        print(f'ROUTINGS # {args.routings}')
        self.dim_pricaps = args.dim_pricaps
        self.num_pricaps = self.encoder.num_caps
        self.num_objcaps = args.num_classes + args.backg_objcaps
        self.dim_objcaps = args.dim_objcaps
        
        self.capsule_routing = PDReconCapsuleRouting(in_num_caps= self.num_pricaps, 
                                            in_dim_caps= self.dim_pricaps,
                                            out_num_caps= self.num_objcaps,
                                            out_dim_caps=self.dim_objcaps, 
                                            routings= args.routings,
                                            num_bgcapsule = args.backg_objcaps)



        print(f'Object #: {self.num_objcaps}, BG Capsule #: {args.backg_objcaps}')
        # decoder (objcaps--> reconstruction)
        self.use_decoder = args.use_decoder # whether use decoder for reconstruction 
        if self.use_decoder:
            print(f'DECODER: {args.decoder_type}, w/ {args.dec_projection_type} projection')
        self.use_reconstruction_mask = args.use_reconstruction_mask 
        self.clip_recon = args.clip_recon
        if self.use_decoder: 
            self.decoder = Decoder(decoder_type= args.decoder_type,  
                                   projection_type= args.dec_projection_type,
                                   outcaps_size= self.num_objcaps*args.dim_objcaps,
                                   output_shape= (self.C, self.H, self.W), 
                                   enc_feature_shape = self.encoder.enc_feature_shape
                                  ) 
    
            self.reconstruct_only_one_capsule = args.reconstruct_only_one_capsule # or reconstruct based on all object capsules
            print(f'...recon only one object capsule: {self.reconstruct_only_one_capsule}')
        
        else:
            # can't use recon mask when no decoder
            self.use_reconstruction_mask = False
        
        # attention window
        self.input_window = AttentionWindow(self.use_reconstruction_mask, self.time_steps, args.mask_threshold)
        
        print(f'========================================================\n')
      
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        device = x.device

        # initialization
        x_recon_for_mask = torch.ones(x.shape, device = device)
        obj_rscore = torch.zeros(batch_size, self.num_objcaps, device=device)  #torch.Size([batch, out_num_caps]), # only need for TD routing
        objcaps =torch.zeros(batch_size, self.num_objcaps, self.dim_objcaps, device=device) #todo 0.1 *torch.randn        
        # for hidden units for recurrent encoder,decoder 
        h_enc = torch.zeros(batch_size, self.dim_pricaps*self.num_pricaps, device = device)
        h_dec = torch.zeros(batch_size, self.dim_pricaps*self.num_pricaps, device = device)

        # log output; accumulate step-wise capsule length and reconstruction, # todo; change to list and concat
        objcaps_len_step = torch.zeros(batch_size, self.time_steps, self.num_objcaps, device = device)
        x_recon_step = torch.zeros(batch_size, self.time_steps, self.C, self.H, self.W, device = device)
       
        
        # run model forward
        for t in range(1, self.time_steps+1):
            
            #############################
            # get initial reconstruction mask, only when test phase
            #############################
#             if t==1:
#                 obj_rscore, obj_recon = get_every_obj_rscore(x, objcaps.detach(), self.decoder, scale=False, save_recon=True)  #[n_batch, n_objects]
# #                 obj_prob =  objcaps_len #[n_batch, n_objects]
# #                 obj_prob =  F.softmax(objcaps_len, dim=1) #[n_batch, n_objects]
#                 obj_prob =  F.softmax(obj_rscore, dim=1) #[n_batch, n_objects]
# #                 obj_prob = scale_coef(objcaps_len, dim=1)
    
#                 topk = 1
#                 topkvalue = obj_prob.topk(topk, dim=1, sorted=True)[0][:,topk-1]
#                 bool_index = obj_prob>=topkvalue.view(-1,1)
# #                 topk_obj_prob = bool_index*obj_prob
# #                 x_recon_for_mask = (topk_obj_prob[:,:, None, None,None]*obj_recon).sum(dim=1)
#                 x_recon_for_mask = (bool_index[:,:, None, None,None]*obj_recon).sum(dim=1)

# #                 x_recon_for_mask = torch.clip(x_recon_for_mask, max=1.0)
# #                 x_recon_for_mask = scale_image(x_recon_for_mask)                
                            
            ##################
            # attending input (based on reconstruction mask from previous step, if the param set True)
            ##################
            x_mask, x_input = self.input_window(x, x_recon_for_mask, timestep=t) #todo what if not detached? 
     
            #############
            # encoding feature
            #############
            
            pricaps, h_enc = self.encoder(x_input, hidden_state=h_enc) # h_enc is used only when encoder use rnn projection
            
            
            ######################
            # capsule layer: dynamic feature routing for classification
            ######################
            # copy decoder first and make it eval mode
            decoder_copy = copy.deepcopy(self.decoder)
            for param in decoder_copy.parameters():
                param.requires_grad = False
            decoder_copy.eval()

            objcaps, _=  self.capsule_routing(pricaps, x_input.detach(), decoder_copy) # objectcaps [n_batch, n_objects(class+bkg), dim_objectcaps] 

                
            # save classification output for each step
#             objcaps_len = obj_rscore    
            objcaps_len = objcaps.norm(dim=-1)  #[n_batch, n_objects]
            objcaps_len_step[:,t-1:t,:] = torch.unsqueeze(objcaps_len, 1)

            ############
            # decoding reconstruction
            ############
            if self.use_decoder:
                
                ####################
                # whether use one most likely capsule for reconstruction
                ###################
                if self.reconstruct_only_one_capsule:

                    # get background recon
                    if self.num_objcaps > self.num_classes:
                        y_bkg = torch.zeros(objcaps_len.size(), device= objcaps_len.device)
                        y_bkg[:, self.num_classes:]= 1
                        objcaps_bkg = (objcaps * y_bkg[:, :, None])
                        x_recon_bkg, _ = self.decoder(objcaps_bkg, h_dec)
                    else:
                        x_recon_bkg = torch.zeros(x.shape, device=device)

                    # get most likely obj recon
                    objcaps_len_narrow = objcaps_len.narrow(dim=1,start=0, length=self.num_classes)
                    idx_max_obj = objcaps_len_narrow.max(dim=1)[1]
                    # idx_max_obj = obj_rscore.max(dim=1)[1]
                    y_onehot = torch.zeros(objcaps_len.size(), device= objcaps_len.device).scatter_(1, idx_max_obj.view(-1,1), 1.)
                    objcaps_onehot = (objcaps * y_onehot[:, :, None])
                    x_recon_obj, h_dec = self.decoder(objcaps_onehot, h_dec) # h_dec is only used when decoder use rnn projection

                    # get final recon
                    x_recon = x_recon_obj + x_recon_bkg
                    x_recon_for_mask = x_recon_obj.detach()    
                else: # or recon from all capsules combined
                    x_recon, h_dec = self.decoder(objcaps, h_dec) # h_dec is only used when decoder use rnn projection
                    x_recon_for_mask = x_recon.detach()

               
                if self.clip_recon:
                    x_recon = torch.clip(x_recon, min=0.0, max=1.0)
                
                # save reconstruction output
                x_recon_step[:,t-1:t,:,:,:] = torch.unsqueeze(x_recon, 1)
                
                
                
                ###########################
                # create weighted recon for mask
                 ###########################
#                 obj_rscore, obj_recon = get_every_obj_rscore(x_input, objcaps.detach(), self.decoder, save_recon=True)  #[n_batch, n_objects]
# #                 obj_prob =  objcaps_len #[n_batch, n_objects]
# #                 obj_prob =  F.softmax(objcaps_len, dim=1) #[n_batch, n_objects]
#                 obj_prob =  F.softmax(obj_rscore, dim=1) #[n_batch, n_objects]
# #                 obj_prob = scale_coef(objcaps_len, dim=1)
    
#                 topk = 10 #3
#                 topkvalue = obj_prob.topk(topk, dim=1, sorted=True)[0][:,topk-1]
#                 bool_index = obj_prob>=topkvalue.view(-1,1)
#                 topk_obj_prob = bool_index*obj_prob
                
#                 x_recon_for_mask = (topk_obj_prob[:,:, None, None,None]*obj_recon).sum(dim=1)
#                 x_recon_for_mask = (bool_index[:,:, None, None,None]*obj_recon).sum(dim=1)

#                 x_recon_for_mask = torch.clip(x_recon_for_mask, max=1.0)
#                 x_recon_for_mask = scale_image(x_recon_for_mask)
                
        return objcaps_len_step, x_recon_step


# ------------------
# Loss functions
# ------------------
def margin_loss(y_pred, y_true, weight=None):
    '''
    margin loss is used for classification loss
    '''
    # narrow down to valid objects (excluding bkg capsule output)
    num_classes = y_true.size(dim=1)
    y_pred = y_pred.narrow(dim=1,start=0, length=num_classes) 
    
    # calculate losses
    m_neg = 0.1 # margin loss allowed for negative case (for absent digits)
    lam_abs = 0.5 # down-weighting loss for absent digits (prevent the initial learning from shrinking the lengths of the class capsules    
    L_present =  y_true* torch.clamp((y_true-m_neg) - y_pred, min=0.) ** 2   
#     L_present =  torch.clamp(y_true, min=0., max=1.) * torch.clamp((y_true-m_neg) - y_pred, min=0.) ** 2 # clamped version  
    L_absent = lam_abs * torch.clamp(1 - y_true, min=0.) * torch.clamp(y_pred-m_neg, min=0.) ** 2
    
    if weight is not None:
        L_margin = weight*(L_present+L_absent).sum(dim=1)
    else:
        L_margin = (L_present+L_absent).sum(dim=1)

    return L_margin.mean()

def margin_loss_allstep(y_pred_step, y_true, weight=None):
    '''
    margin loss is used for classification loss over all steps
    '''
    time_steps = y_pred_step.size(dim=1)
    loss =0.0
    for t in range(time_steps):
        y_pred = y_pred_step[:,t]
        loss += margin_loss(y_pred, y_true, weight)
    return loss

def mse_loss(x_recon, x, clip=False, weight=None):
    '''
    mse loss is used for reconstruction loss
    '''
    if clip:  # for clipping cumulative recon canvas, not necessary for individual recon  
        x_recon = torch.clip(x_recon,0,1)
            
    if weight is not None:
        mse_loss = nn.MSELoss(reduction='none')(x_recon, x).view(x.size(0), -1).mean(dim=1) #(B,)
        mse_loss = weight*mse_loss
    else:
        mse_loss = nn.MSELoss(reduction='none')(x_recon, x).view(x.size(0), -1).mean(dim=1) #(B,)
    
    return mse_loss.mean()

def mse_loss_allstep(x_recon_step, x, time_steps,  weight=None):
    '''
    mse loss is used for reconstruction loss over all steps
    '''
    loss =0.0
    for t in range(time_steps):
        x_recon= x_recon_step[:,t]
        loss += mse_loss(x_recon, x, clip=False, weight=weight)
    return loss


def loss_fn(objcaps_len_step, y_true, x_recon_step, x, args, gtx=None, use_recon_loss=True, class_weight=None):
 
    if torch.is_tensor(gtx): # if separate grountruth x (intact version) is given
        x = gtx # replace x with gtx    
    
    if class_weight is not None:
        class_weight = torch.FloatTensor(class_weight).to(args.device)
        indices = torch.argmax(y_true, dim=1)
        weight = torch.gather(class_weight, 0, indices) #(B,)
    else:
        weight = None
        
    if args.class_loss_is_computed_on_what_step == 'every':
        class_loss = margin_loss_allstep(objcaps_len_step, y_true, weight)
    elif args.class_loss_is_computed_on_what_step == 'sum':
        y_pred = torch.sum(objcaps_len_step, dim=1)
        class_loss = margin_loss(y_pred, y_true, weight)
    elif args.class_loss_is_computed_on_what_step == 'last':
        y_pred = objcaps_len_step[:,-1]
        class_loss = margin_loss(y_pred, y_true, weight)

    if use_recon_loss:
        if args.recon_loss_is_computed_on_what_step == 'every':
            recon_loss = mse_loss_allstep(x_recon_step, x, args.time_steps, weight)
        if args.recon_loss_is_computed_on_what_step == 'sum':
            x_recon = torch.sum(x_recon_step, dim=1)
            recon_loss = mse_loss(x_recon, x, clip=True, weight=weight)
        if args.recon_loss_is_computed_on_what_step == 'last':
            x_recon= x_recon_step[:,-1]
            recon_loss = mse_loss(x_recon, x, clip=False, weight=weight)
        total_loss = class_loss + args.lam_recon*recon_loss

    else:
        recon_loss = torch.Tensor([-99])
        total_loss = class_loss
        
    return total_loss, class_loss, recon_loss


# ------------------
# Accuracy
# ------------------

def topkacc(y_pred: torch.Tensor, y_true:  torch.Tensor, topk=1):
    """
    if one of the top2 predictions are accurate --> 1, none--> 0
    
    Input: 
        - y_pred should be a vector of prediction score 
        - y_true should be in multi-hot encoding format (one or zero; can't deal with duplicates)
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        topk_indices = y_pred.topk(topk, sorted=True)[1] 
        accs = torch.gather(y_true, dim=1, index=topk_indices).sum(dim=1)

    return accs

def exactmatch(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    if y_pred and y_true matches exactly --> 1, not --> 0
    
    Input: torch tensor 
        - both y_pred and y_true should be in the same format
        e.g., if y_true is multi-hot, then y_pred should be made in multi-hot as well
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        accs = (y_pred == y_true).all(dim=1).float()
    
    return accs

def partialmatch(y_pred: torch.Tensor, y_true:  torch.Tensor, n_targets=2):
    """
    when n_targets=2, if one of the two predictions are accurate --> 0.5, none--> 0
    
    Input: 
        - y_pred should be a vector of prediction score 
        - y_true should be in multi-hot encoding format (one or zero; can't deal with duplicates)
    Return: 
        - a vector of accuracy from each image --> [n_images,]
    """
    with torch.no_grad():
        topk_indices = y_pred.topk(n_targets, sorted=True)[1] 
        accs = torch.gather(y_true, dim=1, index=topk_indices).sum(dim=1)/n_targets

    return accs


def compute_hypothesis_based_acc(objcaps_len_step_narrow, y_hot, only_acc=True):
    def get_first_zero_index(x, axis=1):
        cond = (x == 0)
        return ((cond.cumsum(axis) == 1) & cond).max(axis, keepdim=True)[1]

    pstep = objcaps_len_step_narrow.max(dim=-1)[1]

    # check whether consecutive predictions are the same (diff=0)
    pnow = pstep[:,1:]
    pbefore = pstep[:,:-1]
    pdiff = (pnow-pbefore)
    null_column = -99*torch.ones(pdiff.size(0),1).to(pdiff.device) # add one null column at start
    pdiff = torch.cat([null_column, pdiff], dim=1)
    pdiff[:,-1]=0 # add diff= 0 to final step (to use final prediction if no criterion made)

    # get first two consecutive (diff zero) index and model predictions
    first_zero_index = get_first_zero_index(pdiff)
    final_pred= torch.gather(pstep, 1, first_zero_index).flatten()
    accs = torch.eq(final_pred.to(y_hot.device), y_hot.max(dim=1)[1]).float()
    nstep = (first_zero_index.flatten()+1) #.cpu().numpy()
    if only_acc:
        return accs
    else:
        return accs, final_pred, nstep



def compute_entropy_based_acc(objcaps_len_step_narrow, y_hot, threshold=0.6, use_cumulative = False, only_acc= True):
    from torch.distributions import Categorical

    def get_first_true_index(boolarray, axis=1, when_no_true='final_index'):
        # boolarray = Batch x Stepsize
        first_true_index = ((boolarray.cumsum(axis) == 1) & boolarray).max(axis, keepdim=True)[1] # when no true, set as 0

        if when_no_true == 'final_index': # when there is no true, use final index
            final_index = boolarray.shape[1]-1
            no_true_condition = (~boolarray).all(dim=1).reshape(-1,1)
            first_true_index = first_true_index + final_index * no_true_condition
            return  first_true_index, no_true_condition
        else:
            return first_true_index

    if use_cumulative:
        score = objcaps_len_step_narrow.cumsum(dim=1)
        pred = score.max(dim=-1)[1]
    else:
        score = objcaps_len_step_narrow # Batch x Stepsize x Category
        pred = score.max(dim=-1)[1] # Batch x Stepsize


    # compute entropy from softmax output with Temp scale
    T=0.2
    softmax = F.softmax(score/T, dim=-1) # torch.Size([1000, 4, 10])
    entropy = Categorical(probs = softmax).entropy() # torch.Size([1000, 4])

    # entropy thresholding
    stop = entropy<threshold
    boolarray = (stop == True)

    # get first index that reached threshold
    first_true_index, no_stop_condition = get_first_true_index(boolarray, axis=1, when_no_true='final_index')

    final_pred = torch.gather(pred, dim=1, index= first_true_index).flatten()
    accs = torch.eq(final_pred.to(y_hot.device), y_hot.max(dim=1)[1]).float()
    nstep = (first_true_index.flatten()+1) #.cpu().numpy()

    if only_acc:
        return accs
    else:
        return accs, final_pred, nstep, no_stop_condition, entropy
    
def acc_fn(objcaps_len_step, y_true, acc_type= 'entropy', single_step=None):
    '''
    1) entropy
    2) hypothesis
    3) topk accuracy: format should be top@k,
    '''
    num_classes = y_true.size(dim=1) #onehot vector
    objcaps_len_step_narrow = objcaps_len_step.narrow(dim=-1,start=0, length=num_classes) # in case a background cap was added    

    if acc_type == 'entropy':
        if single_step:
            y_pred = objcaps_len_step_narrow[:,-1]
            accs = topkacc(y_pred, y_true, topk=1)
        else:
            accs = compute_entropy_based_acc(objcaps_len_step_narrow, y_true,  threshold=0.6, use_cumulative = False)
    elif acc_type == 'hypothesis':
        if single_step:
            y_pred = objcaps_len_step_narrow[:,-1]
            accs = topkacc(y_pred, y_true, topk=1)
        else:
            accs = compute_hypothesis_based_acc(objcaps_len_step_narrow, y_true)
            
    elif 'top' in acc_type:
        # get final prediction
        y_pred = objcaps_len_step_narrow[:,-1]
        topk = int(acc_type.split('@')[1])
        accs = topkacc(y_pred, y_true, topk=topk)
        
    else:
        raise NotImplementedError('given acc functions are not implemented yet')
        
    return accs


# --------------------
# Train and Test
# --------------------

def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, writer, args):
    """
    for each batch:
        - forward pass  
        - compute loss
        - param update
    """    
    losses = AverageMeter('Loss', ':.4e')
    
    model.train() 
    with tqdm(total=len(train_dataloader), desc='epoch {} of {}'.format(epoch, args.n_epochs)) as pbar:
#     time.sleep(0.1)        
        
        # load batch from dataloader 
        for i, data in enumerate(train_dataloader):
            global_step = (epoch-1) * len(train_dataloader) + i + 1 #global batch number
            
            # load dataset on device
#             x = x.view(x.shape[0], -1).to(args.device)
            if len(data)==2:
                x, y = data
                x = x.to(args.device)
                y = y.to(args.device)
                gtx = None
            elif len(data)==3:
                x, gtx, y = data
                x = x.to(args.device)
                y = y.to(args.device)
                gtx = gtx.to(args.device)

            # forward pass
            objcaps_len_step, x_recon_step = model(x)

            # compute loss for this batch and append it to training loss
            loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
                                 gtx=gtx, use_recon_loss = args.use_decoder, class_weight = args.class_weight) 
            # separate class/recon
#             if epoch<50:
#                 loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
#                                      gtx=gtx, use_recon_loss = False)
#             elif epoch>=50:
#                 loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
#                                      gtx=gtx, use_recon_loss = args.use_decoder) 

            losses.update(loss.item(), x.size(0))



            # minibatch update; zero out previous gradients and backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # record grad norm and clip to prevent exploding gradients
            if args.record_gradnorm:
                grad_norm = 0
                for name, p in model.named_parameters():
                    grad_norm += p.grad.norm().item() if p.grad is not None else 0
                writer.add_scalar('grad_norm', grad_norm, global_step)                
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            # update param
            optimizer.step()
            
            # update schduler for cycle learning
            if args.clr:
                scheduler.step()

            # end of each batch, update tqdm tracking
            pbar.set_postfix(batch_loss='{:.3f}'.format(loss.item()))
            pbar.update()
    
    # update scheduler
    if not args.clr:
        scheduler.step()
    return losses.avg

def evaluate(model, x, y, args, acc_type, gtx=None):
    """
    Run model prediction on testing dataset and compute loss/acc 
    """
    
    # evaluate
    model.eval()
    
    # load testing dataset on device
#     x = x.view(x.shape[0], -1).float().to(args.device)
    x= x.to(args.device)                
    y = y.to(args.device)
    
    if torch.is_tensor(gtx):
        gtx = gtx.to(args.device)

    with torch.no_grad():
        
        # run model with testing data and get predictions
        objcaps_len_step, x_recon_step = model(x)
        
        # compute batch loss and accuracy
        loss, loss_class, loss_recon = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
                                               gtx=gtx, use_recon_loss = args.use_decoder, class_weight = args.class_weight)
        if args.time_steps==1:
            acc = acc_fn(objcaps_len_step, y, acc_type, single_step=True)
        else:
            acc = acc_fn(objcaps_len_step, y, acc_type)

    return (loss, loss_class, loss_recon), acc, objcaps_len_step, x_recon_step


def test(model, dataloader, args, acc_type):
    """
    for each batch:
        - evaluate loss & acc ('evaluate')
    log average loss & acc  
    """   
    losses = AverageMeter('Loss', ':.4e')
    losses_class = AverageMeter('Loss_class', ':.4e')
    losses_recon = AverageMeter('Loss_recon', ':.4e')
    accs_topk = AverageMeter(acc_type, ':6.2f')
    
    # load batch data
    for data in dataloader:
        if len(data)==2:
            x, y = data
            gtx = None
        elif len(data)==3:
            x, gtx, y = data
            
        # evaluate
        batch_losses, batch_acc, objcaps_len_step, x_recon_step, \
        =  evaluate(model, x, y, args, acc_type, gtx)
            
        # aggregate loss and acc
        losses.update(batch_losses[0].item(), x.size(0))
        losses_class.update(batch_losses[1].item(), x.size(0))
        losses_recon.update(batch_losses[2].item(), x.size(0))
        accs_topk.update(batch_acc.mean().item(), x.size(0))

    return losses.avg, losses_class.avg, losses_recon.avg, accs_topk.avg


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, writer, args, acc_type):
    """
    for each epoch:
        - train the model, update param, and log the training loss ('train_epoch')
        - save checkpoint
        - compute and log average val loss/acc
        - save best model
    """
    start_epoch = 1

    if args.restore_file:
        print(f'Restoring parameters from {args.restore_file}')
        start_epoch = load_checkpoint(args.restore_file, [model], [optimizer], map_location=args.device.type)
        args.n_epochs += start_epoch
        print(f'Resuming training from epoch {start_epoch}')
    
    path_best = None
    epoch_no_improve = 0

    for epoch in range(start_epoch, args.n_epochs+1):
        
        # train epoch and lod to tensorboard writer

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch, writer, args)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        
        # save checkpoint 
        if args.save_checkpoint:
            save_checkpoint({'epoch': epoch,
                             'model_state_dicts': [model.state_dict()],
                             'optimizer_state_dicts': [optimizer.state_dict()]},
                            checkpoint=args.log_dir,
                            quiet=True)

        # compute validation loss and acc
        if (epoch) % args.validate_after_howmany_epochs == 0:
            val_loss, val_loss_class, val_loss_recon, val_acc = test(model, val_dataloader, args, acc_type)
            
            # logging validation info to tensorboard writer
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Loss_Class', val_loss_class, epoch)
            writer.add_scalar('Val/Locc_Recon', val_loss_recon, epoch)
            writer.add_scalar('Val/Acc', val_acc, epoch)
                        
            if args.verbose:
                print(f"==> Epoch {epoch:d}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_loss_class={val_loss_class:.5f}, val_loss_recon={val_loss_recon:.5f}, val_acc={val_acc:.4f}")
               
            # update best validation acc and save best model to output dir
            if (round(val_acc,4) > round(args.best_val_acc,4)):
                args.best_val_acc = val_acc
                epoch_no_improve = 0
                # remove previous best
                if path_best:
                    try:
                        os.remove(path_best)
                    except:
                        print("Error while deleting file ", path_best)

                # save current best
                path_best = args.log_dir +f'/best_epoch{epoch:d}_acc{val_acc:.4f}.pt'
                torch.save(model.state_dict(), path_best)  
                print(f"the model with best val_acc ({val_acc:.4f}) was saved to disk")
            else:
                epoch_no_improve += 1
                
        # archive models        
        if (epoch%10==0):  
            torch.save(model.state_dict(), args.log_dir +f'/archive_epoch{epoch:d}_acc{val_acc:.4f}.pt')  #output_dir
            print(f"model archived at epoch ({epoch})")            


        # abort training early if acc below criterior or exploding
        if (epoch%100 == 0) and (epoch < args.n_epochs):
            if hasattr(args, 'abort_if_valacc_below'):
                if (args.best_val_acc < args.abort_if_valacc_below) or math.isnan(val_acc):
                    torch.save(model.state_dict(), args.log_dir +f'/aborted_epoch{epoch:d}_acc{val_acc:.4f}.pt')
                    status = f'===== EXPERIMENT ABORTED: val_acc is {val_acc} at epoch {epoch} (Criterion is {args.abort_if_valacc_below}) ===='
                    writer.add_text('Status', status, epoch)
                    print(status)
#                     sys.exit()
                    break
                else:
                    status = '==== EXPERIMENT CONTINUE ===='
                    writer.add_text('Status', status, epoch)
                    print(status)
                    
        if epoch_no_improve >= 20:
            torch.save(model.state_dict(), args.log_dir +f'/earlystop_{epoch:d}_acc{val_acc:.4f}.pt')
            status = f'===== EXPERIMENT EARLY STOPPED (no progress on val_acc for last 20 epochs) ===='
            writer.add_text('Status', status, epoch)
            print(status)
#             sys.exit()
            break

        if epoch == args.n_epochs:
            torch.save(model.state_dict(), args.log_dir +f'/last_{epoch:d}_acc{val_acc:.4f}.pt')
            status = f'===== EXPERIMENT RAN TO THE END EPOCH ===='
            writer.add_text('Status', status, epoch)
            print(status)
            
            
            
            


def lr_range_test(model, train_dataloader,  optimizer, scheduler, args, acc_type):
    """
    for each epoch:
    """
    all_losses, all_lrs = [], []
    for epoch in range(1, args.n_epochs+1):        
        losses, lrs = train_epoch_lr(model, train_dataloader, optimizer, scheduler, epoch, args)
        all_losses.extend(losses)
        all_lrs.extend(lrs)
    
    # save results to save
    import pandas as pd
    df = pd.DataFrame()
    df['loss'] = all_losses
    df['lr'] = all_lrs
    df.to_csv('lr_range_test.csv',index=False)
    print('test results saved to disk')
    

def train_epoch_lr(model, train_dataloader, optimizer, scheduler, epoch, args):
    """
    for each batch:
    """    
    
    model.train() 
    with tqdm(total=len(train_dataloader), desc='epoch {} of {}'.format(epoch, args.n_epochs)) as pbar:
#     time.sleep(0.1)        
        
        # load batch from dataloader 
        losses = []
        lrs = []
        for i, data in enumerate(train_dataloader):
            
            # load dataset on device
#             x = x.view(x.shape[0], -1).to(args.device)
            if len(data)==2:
                x, y = data
                x = x.to(args.device)
                y = y.to(args.device)
                gtx = None
            elif len(data)==3:
                x, gtx, y = data
                x = x.to(args.device)
                y = y.to(args.device)
                gtx = gtx.to(args.device)

            # forward pass
            objcaps_len_step, x_recon_step = model(x)

            # compute loss for this batch and append it to training loss
            loss, _, _ = loss_fn(objcaps_len_step, y, x_recon_step, x, args, 
                                 gtx=gtx, use_recon_loss = args.use_decoder) 
            
            if torch.isnan(torch.tensor([loss.item()])).any():
                break
            
            
            # minibatch update; zero out previous gradients and backward pass
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)

            # update param
            lrs.append(scheduler.get_last_lr()[0])
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()


            # end of each batch, update tqdm tracking
            pbar.set_postfix(batch_loss='{:.3f}'.format(loss.item()))
            pbar.update()
    
    return losses, lrs
                


            
            
            
            
            
            
            
            

