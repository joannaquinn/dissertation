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
from ndata import DATAMAP, STATIC_POINTS, grid_dataset
from loss_functions import *
from constants_jo import *
from parsing import CNNRunParameters

from model_names import ALL_DATA_DIR, INTERVALS, CELL_SIZES

PLOT_DIRS = {f'{location}_{net}_{interval}min_{cell_size}': os.path.join(ALL_DATA_DIR, location, 'plots', 'nnets') 
             for net in INTERVALS
             for location in INTERVALS[net]
             for interval in INTERVALS[net][location]
             for cell_size in CELL_SIZES[location]}

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
torch.manual_seed(25)

def loss_masked2(target,pred, loss_function=torch.nn.MSELoss()):
    '''
    mse loss function which masks out points for which we
    don't have data
    uses MSE
    inputs are size b x k x h x w
    '''
    # create mask
    idx = target.nonzero(as_tuple=True)
    mask = torch.zeros(size=target.size()).to(device).float()
    mask[idx] = 1.0
    target_masked = target*mask.to(device).float()
    pred_masked = pred*mask.to(device).float()
    loss = loss_function(target_masked, pred_masked)
    return loss

def load_model(model, model_name, epoch, suffix):
    '''
    loads the pretrained model state from an epoch
    '''
    model_dir = os.path.join(os.getcwd(), 'models', model_name)
    filename = f'{model_name}_epoch{epoch}_{suffix}.pth'
    model.load_state_dict(torch.load(os.path.join(model_dir, filename), map_location=lambda storage, loc: storage))
    return model

def plot_loss(train_losses, valid_losses, filename=None, figsize=(20,15)):
    sns.set()
    epochs = list(range(len(train_losses)))
    df = pd.DataFrame(data={'epoch':epochs, 'train':train_losses, 'valid':valid_losses})
    df = pd.melt(df, id_vars='epoch', value_vars=['train','valid'])
    fig, ax = plt.subplots()
    plot = sns.lineplot(x='epoch', y='value', hue='variable', data=df, ax=ax)
    
    if filename is not None: # save loss figure
        fig = plot.get_figure()
        fig.savefig(filename)
        
def save_model_state(model, model_name, epoch, suffix):
    model_dir = os.path.join(os.getcwd(), 'models', model_name)
    filename = f'{model_name}_epoch{epoch}_{suffix}.pth'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, filename))
    print(f'Saved model {os.path.join(model_dir, filename)}')
    
def save_model_architecture(args, model_name, suffix):
    model_dir = os.path.join(os.getcwd(), 'models', model_name)
    filename = f'{model_name}_{suffix}_args.pkl'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, filename), 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    
    ### data / params ###
    params = CNNRunParameters()
    args = params.args # parse model parameters from command line
    print(args)
    
    model_name = args['model_name']
    save_model_architecture(args, model_name, args['save_suffix'])
    
    static_points = None
    directories = DATAMAP[model_name]
    data = {}
    data['train'] = grid_dataset(directories['train'], static_points=static_points, pm_cols=(0,), normalise=args['normalise'])
    data['valid'] = grid_dataset(directories['valid'], static_points=static_points, pm_cols=(0,), normalise=False)
    if args['normalise']:
        data['valid'].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    
    for t in ('train', 'valid'):
        print(f'Size of {t} data: {len(data[t])}')
    
    example = data['train'][0]
    print(f'Example x data point has shape {example["x"].size()}')
    print(f'Example y data point has shape {example["y"].size()}')
    
    batch_size = args['batch_size']
    train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=0)
    
    in_channels, height, width = example['x'].size(0), example['x'].size(1), example['x'].size(2)
    out_channels = example['y'].size(0)
    
    model = nnets.CNNAutoEncoder(height, width, in_channels, out_channels,  encoded_size=args['encoded_size'],
                                 encode_channels=args['encode_channels'], encode_conv_layers=args['encode_conv_layers'], encode_kernel_sizes=args['encode_kernel_sizes'], pooling=args['pooling'], pool_sizes=args['pool_sizes'], encode_conv_activation=args['encode_conv_activation'], encode_batch_norm=args['encode_batch_norm'], encode_skip=args['encode_skip'],
                                 encode_fc_layers=args['encode_fc_layers'], encode_fc_neurons=args['encode_fc_neurons'], encode_fc_activation = args['encode_fc_activation'], encode_dropout=args['encode_dropout'],
                                 decode_channels=args['decode_channels'], decode_conv_layers=args['decode_conv_layers'], decode_kernel_sizes=args['decode_kernel_sizes'], upsample_scales=args['upsample_scales'], decode_conv_activation=args['decode_conv_activation'], decode_batch_norm=args['decode_batch_norm'], decode_skip=args['decode_skip'],
                                 decode_fc_layers=args['decode_fc_layers'], decode_fc_neurons=args['decode_fc_neurons'], decode_fc_activation = args['decode_fc_activation'], decode_dropout=args['decode_dropout']).to(device)

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
        model = load_model(model, model_name, start_epoch, suffix)
    
    train_losses = []
    valid_losses = []
    for epoch in range(start_epoch+1, num_epochs):
        # training
        model.train()
        train_loss = 0.0
        for i, d in enumerate(train_loader):
            x, y = d['x'].to(device).float(), d['y'].to(device).float()
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