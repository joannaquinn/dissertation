import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

import nnets
from ndata import DATAMAP, STATIC_POINTS, grid_dataset, grid_seq_dataset
from loss_functions import *
from constants_jo import *
from parsing import CNNRunParameters
from ntrain import plot_loss, save_model_state, save_model_architecture, load_model, PLOT_DIRS, loss_masked2

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
torch.manual_seed(25)
    
if __name__ == '__main__':
    
    ### data / params ###
    params = CNNRunParameters()
    args = params.args # parse model parameters from command line
    print(args)
    
    model_name = args['model_name']
    save_model_architecture(args, model_name, args['save_suffix'])
    
    static_points = np.load(STATIC_POINTS[model_name])
    seq_len = args['seq_len']
    directories = DATAMAP[model_name]
    
    data = {}
    data['train'] = grid_seq_dataset(directories['train'], seq_len, static_points=static_points, pm_cols=(0,), normalise=args['normalise'])
    data['valid'] = grid_seq_dataset(directories['valid'], seq_len, static_points=static_points, pm_cols=(0,), normalise=False)
    if args['normalise']:
        data['valid'].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    
    for t in ('train', 'valid'):
        print(f'Size of {t} data: {len(data[t])}')
    
    example = data['train'][0]
    print(f'Example x data point has shape {example["x"].size()}')
    
    print(f'Example y data point has shape {example["y"].size()}')
    
    batch_size = args['batch_size']
    train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=False, num_workers=0)
    
    in_channels, height, width = example['x'].size(1), example['x'].size(2), example['x'].size(3)
    out_channels = example['y'].size(0)
    
    model = nnets.CRNN(height, width, in_channels, out_channels,  encoded_size=args['encoded_size'],
                       encode_channels=args['encode_channels'], encode_conv_layers=args['encode_conv_layers'], encode_kernel_sizes=args['encode_kernel_sizes'], pooling=args['pooling'], pool_sizes=args['pool_sizes'], encode_conv_activation=args['encode_conv_activation'], encode_batch_norm=args['encode_batch_norm'], encode_skip=args['encode_skip'],
                       encode_fc_layers=args['encode_fc_layers'], encode_fc_neurons=args['encode_fc_neurons'], encode_fc_activation = args['encode_fc_activation'], encode_dropout=args['encode_dropout'],
                       decode_channels=args['decode_channels'], decode_conv_layers=args['decode_conv_layers'], decode_kernel_sizes=args['decode_kernel_sizes'], upsample_scales=args['upsample_scales'], decode_conv_activation=args['decode_conv_activation'], decode_batch_norm=args['decode_batch_norm'], decode_skip=args['decode_skip'],
                       decode_fc_layers=args['decode_fc_layers'], decode_fc_neurons=args['decode_fc_neurons'], decode_fc_activation = args['decode_fc_activation'], decode_dropout=args['decode_dropout'],
                       lstm_hidden_size=args['lstm_hidden_size'], lstm_layers=args['lstm_layers'], lstm_bias=args['lstm_bias'], lstm_dropout=args['lstm_dropout'], bidirectional=args['bidirectional']).to(device)

    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    num_epochs = args['num_epochs']
    loss_fn = nn.MSELoss(reduction='mean')
    if args['optimiser'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    patience = args['patience']
    
    start_epoch = args['start_epoch']
    if start_epoch > -1: # load from previous
        model = load_model(model, model_name, start_epoch, args["save_suffix"])
    
    train_losses = []
    valid_losses = []
    for epoch in range(start_epoch+1, num_epochs):
        # training
        model.train()
        train_loss = 0.0
        for i, d in enumerate(train_loader):
            x, y = d['x'].to(device).float(), d['y'].to(device).float()
            x = x.permute(1,0,2,3,4) # time first batch second
            optimizer.zero_grad()
            out = model(x) # forward pass
            loss = loss_masked2(y, out)
            loss.backward()
            optimizer.step() # backward pass
            train_loss += loss.item() # weight loss by batch size
        train_loss = train_loss / len(data['train']) # get avg
        train_losses.append(train_loss)
        
        #validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad(): # don't track these gradients
            for i, d in enumerate(valid_loader):
                x, y = d['x'].to(device).float(), d['y'].to(device).float()
                x = x.permute(1,0,2,3,4) # time first batch second
                out = model(x) # forward pass
                loss = loss_masked2(y, out)
                valid_loss += loss.item()
            valid_loss = valid_loss / len(data['valid']) # get avg
            valid_losses.append(valid_loss)
            
        if args['print_epochs']:
            print(f'Epoch:{epoch}, train_loss:{train_loss}, valid_loss:{valid_loss}')
        if args['save']:
            save_model_state(model, model_name, epoch, args['save_suffix'])
            
        # early stopping
        if epoch-patience > 0:
            min_val_loss = min(valid_losses)
            if min(valid_losses[-patience:]) > min_val_loss:
                print(f'Early stopping: terminating at epoch {epoch}')
                break
    
    plot_filename = os.path.join(PLOT_DIRS[model_name], f'{model_name}_{args["save_suffix"]}_loss.png')
    plot_loss(train_losses, valid_losses, filename=plot_filename)
    min_val_loss = min(valid_losses)
    print(f'Min validation loss: epoch{valid_losses.index(min_val_loss)+start_epoch+1}, {min_val_loss}')