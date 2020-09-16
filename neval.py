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
from ndata import DATAMAP, STATIC_POINTS, grid_dataset, reverse_scale_outputs
from loss_functions import *
from constants_jo import *
from ntrain import PLOT_DIRS, loss_masked2
from parsing import TestParameters

from model_names import ALL_DATA_DIR, INTERVALS, CELL_SIZES, ROOT_MODEL_DIR

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
torch.manual_seed(25)

OUTPUT_DIRS = {f'{location}_{net}_{interval}min_{cell_size}': os.path.join(ROOT_MODEL_DIR, f'{location}_{net}_{interval}min_{cell_size}', 'outputs') 
             for net in INTERVALS
             for location in INTERVALS[net]
             for interval in INTERVALS[net][location]
             for cell_size in CELL_SIZES[location]}

def load_model_architecture(model_name, suffix):
    '''
    loads the hyperparameters of the pre-trained model
    '''
    model_dir = os.path.join(os.getcwd(), 'models', model_name)
    filename = f'{model_name}_{suffix}_args.pkl'
    with open(os.path.join(model_dir, filename), 'rb') as f:
        return pickle.load(f)
 
def load_model(model, model_name, epoch, suffix):
    '''
    loads the pretrained model state from an epoch
    '''
    model_dir = os.path.join(os.getcwd(), 'models', model_name)
    filename = f'{model_name}_epoch{epoch}_{suffix}.pth'
    model.load_state_dict(torch.load(os.path.join(model_dir, filename), map_location=lambda storage, loc: storage))
    return model

def save_predicted_data(x, y, out, model_name, epoch, suffix, d, index):
    '''
    saves the input, target and predicted values for pm to output directory
    '''
    directory = OUTPUT_DIRS[model_name]
    np.save(os.path.join(directory, f'{model_name}_epoch{epoch}_{suffix}_{d}_{index}_x.npy'), x)
    np.save(os.path.join(directory, f'{model_name}_epoch{epoch}_{suffix}_{d}_{index}_y.npy'), y)
    np.save(os.path.join(directory, f'{model_name}_epoch{epoch}_{suffix}_{d}_{index}_out.npy'), out)

def plot_pm_heatmaps(target, pred, filename=None, figsize=(10,7), vmin=0, vmax=8):
    sns.set()
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize, squeeze=True)
    
    fig1=sns.heatmap(target, ax=axes[0], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    fig2=sns.heatmap(pred, ax=axes[1], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    
    plt.colorbar(fig1.get_children()[0], ax = axes, orientation = 'horizontal');
    axes[0].set_title('Target values')
    axes[1].set_title('Predicted values')
    
    if filename is not None:
        fig.savefig(filename)
    
if __name__ == '__main__':
    
    ### data / params ###
    params = TestParameters()
    eval_args = params.args # parse model parameters from command line
    print(eval_args)
    
    model_name = eval_args['model_name']
    args = load_model_architecture(model_name, eval_args['save_suffix'])
    print(args)
    
    static_points = None
    directories = DATAMAP[model_name]
    data = {}
    t = eval_args['data']
    data[t] = grid_dataset(directories[t], static_points=static_points, pm_cols=(0,), normalise=False)
    if args['normalise']:
        data['train'] = grid_dataset(directories['train'], static_points=static_points, pm_cols=(0,), normalise=True)
        data[t].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    print(f'Size of {t} data: {len(data[t])}')
    
    example = data[t][0]
    print(f'Example x data point has shape {example["x"].size()}')
    print(f'Example y data point has shape {example["y"].size()}')
    
    batch_size = 1
    loader = DataLoader(data[t], batch_size=batch_size, shuffle=False, num_workers=0)
    
    in_channels, height, width = example['x'].size(0), example['x'].size(1), example['x'].size(2)
    out_channels = example['y'].size(0)
    
    model = nnets.CNNAutoEncoder(height, width, in_channels, out_channels,  encoded_size=args['encoded_size'],
                                 encode_channels=args['encode_channels'], encode_conv_layers=args['encode_conv_layers'], encode_kernel_sizes=args['encode_kernel_sizes'], pooling=args['pooling'], pool_sizes=args['pool_sizes'], encode_conv_activation=args['encode_conv_activation'], encode_batch_norm=args['encode_batch_norm'], encode_skip=args['encode_skip'],
                                 encode_fc_layers=args['encode_fc_layers'], encode_fc_neurons=args['encode_fc_neurons'], encode_fc_activation = args['encode_fc_activation'], encode_dropout=args['encode_dropout'],
                                 decode_channels=args['decode_channels'], decode_conv_layers=args['decode_conv_layers'], decode_kernel_sizes=args['decode_kernel_sizes'], upsample_scales=args['upsample_scales'], decode_conv_activation=args['decode_conv_activation'], decode_batch_norm=args['decode_batch_norm'], decode_skip=args['decode_skip'],
                                 decode_fc_layers=args['decode_fc_layers'], decode_fc_neurons=args['decode_fc_neurons'], decode_fc_activation = args['decode_fc_activation'], decode_dropout=args['decode_dropout']).to(device)
    epoch = eval_args['epoch']
    model = load_model(model, model_name, epoch, eval_args['save_suffix'])
    model.eval()
    
    loss_fn = nn.MSELoss(reduction='mean')
    avg_loss = 0.0
    # mape_loss = 0.0
    # mae_loss = 0.0
    mape_losses = []
    mae_losses = []
    
    with torch.no_grad(): # don't track these gradients
        for i, d in enumerate(loader):
            x, y = d['x'].to(device).float(), d['y'].to(device).float()
            out = model(x) # forward pass
            loss = loss_masked2(y, out)
            print(f'Loss for {i} datapoint: {loss.item()}')
            avg_loss += loss.item()
            
            # save to file for further analysis (scale back to original values using training max and mins)
            if args['normalise']:
                x_scale, y_scale, out_scale = reverse_scale_outputs(x.cpu().numpy()[0], y.cpu().numpy()[0], out.cpu().numpy()[0], 
                                                                    data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
            else:
                x_scale, y_scale, out_scale = x.cpu().numpy()[0], y.cpu().numpy()[0], out.cpu().numpy()[0]
            save_predicted_data(x_scale, y_scale, out_scale, model_name, epoch, eval_args['save_suffix'], t, i)
            
            mape_calc = mape_masked_unbatch(y_scale, out_scale)
            mae_calc = mae_masked_unbatch(y_scale, out_scale)
            mape_losses.extend(mape_masked_unbatch(y_scale, out_scale, return_avg=False))
            mae_losses.extend(mae_masked_unbatch(y_scale, out_scale, return_avg=False))
            # mape_loss += mape_calc
            # mae_loss += mae_calc
            print(f'MAPE for {i} datapoint: {mape_calc}')
            print(f'MAE for {i} datapoint: {mae_calc}')
            
            if eval_args['plot']:
                for k in range(y.size(1)): # pm dims
                    plot_filename = os.path.join(PLOT_DIRS[model_name], f'{model_name}_epoch{epoch}_{eval_args["save_suffix"]}_{t}_data{i}_heatmap{k}.png')
                    target = y_scale[k]
                    pred = out_scale[k]
                    vmax = max(target.max(), pred.max())
                    plot_pm_heatmaps(target, pred, filename=plot_filename, vmax=vmax)
            
        avg_loss = avg_loss / len(data[t]) # get avg
        # mape_loss = mape_loss / len(data[t])
    
    all_mape = np.mean(mape_losses)
    all_mae = np.mean(mae_losses)
    print(f'{t} average total loss: {avg_loss}')
    print(f'{t} average MAPE: {all_mape}')
    print(f'{t} average MAE: {all_mae}')