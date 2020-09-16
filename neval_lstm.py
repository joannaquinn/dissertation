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
from ndata import DATAMAP, STATIC_POINTS, grid_dataset, grid_seq_dataset, reverse_scale_outputs
from loss_functions import *
from constants_jo import *
from ntrain_lstm import PLOT_DIRS
from ntrain import loss_masked2
from parsing import TestParameters
from neval import load_model_architecture, load_model, plot_pm_heatmaps, OUTPUT_DIRS

from model_names import ALL_DATA_DIR, INTERVALS, CELL_SIZES, ROOT_MODEL_DIR

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(device)
torch.manual_seed(25)

def save_predicted_data(x, y, out, model_name, epoch, suffix, d, index):
    '''
    saves the input, target and predicted values for pm to output directory
    '''
    directory = OUTPUT_DIRS[model_name]
    np.save(os.path.join(directory, f'{model_name}_epoch{epoch}_{suffix}_{d}_{index}_x.npy'), x)
    np.save(os.path.join(directory, f'{model_name}_epoch{epoch}_{suffix}_{d}_{index}_y.npy'), y)
    np.save(os.path.join(directory, f'{model_name}_epoch{epoch}_{suffix}_{d}_{index}_out.npy'), out)
    
if __name__ == '__main__':
    
    ### data / params ###
    params = TestParameters()
    eval_args = params.args # parse model parameters from command line
    print(eval_args)
    
    model_name = eval_args['model_name']
    args = load_model_architecture(model_name, eval_args['save_suffix'])
    print(args)
    
    static_points = np.load(STATIC_POINTS[model_name])
    seq_len = args['seq_len']
    directories = DATAMAP[model_name]
    data = {}
    t = eval_args['data']
    data[t] = grid_seq_dataset(directories[t], seq_len, static_points=static_points, pm_cols=(0,), normalise=False)
    if args['normalise']:
        data['train'] = grid_dataset(directories['train'], static_points=static_points, pm_cols=(0,), normalise=True)
        data[t].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    print(f'Size of {t} data: {len(data[t])}')
    
    example = data[t][0]
    print(f'Example x data point has shape {example["x"].size()}')
    print(f'Example y data point has shape {example["y"].size()}')
    
    batch_size = 1
    loader = DataLoader(data[t], batch_size=batch_size, shuffle=False, num_workers=0)
    
    in_channels, height, width = example['x'].size(1), example['x'].size(2), example['x'].size(3)
    out_channels = example['y'].size(0)
    
    model = nnets.CRNN(height, width, in_channels, out_channels,  encoded_size=args['encoded_size'],
                       encode_channels=args['encode_channels'], encode_conv_layers=args['encode_conv_layers'], encode_kernel_sizes=args['encode_kernel_sizes'], pooling=args['pooling'], pool_sizes=args['pool_sizes'], encode_conv_activation=args['encode_conv_activation'], encode_batch_norm=args['encode_batch_norm'], encode_skip=args['encode_skip'],
                       encode_fc_layers=args['encode_fc_layers'], encode_fc_neurons=args['encode_fc_neurons'], encode_fc_activation = args['encode_fc_activation'], encode_dropout=args['encode_dropout'],
                       decode_channels=args['decode_channels'], decode_conv_layers=args['decode_conv_layers'], decode_kernel_sizes=args['decode_kernel_sizes'], upsample_scales=args['upsample_scales'], decode_conv_activation=args['decode_conv_activation'], decode_batch_norm=args['decode_batch_norm'], decode_skip=args['decode_skip'],
                       decode_fc_layers=args['decode_fc_layers'], decode_fc_neurons=args['decode_fc_neurons'], decode_fc_activation = args['decode_fc_activation'], decode_dropout=args['decode_dropout'],
                       lstm_hidden_size=args['lstm_hidden_size'], lstm_layers=args['lstm_layers'], lstm_bias=args['lstm_bias'], lstm_dropout=args['lstm_dropout'], bidirectional=args['bidirectional']).to(device)
    epoch = eval_args['epoch']
    model = load_model(model, model_name, epoch, eval_args['save_suffix'])
    model.eval()
    
    loss_fn = nn.MSELoss(reduction='mean')
    avg_loss = 0.0
    mape_losses = []
    mae_losses = []
    
    with torch.no_grad(): # don't track these gradients
        for i, d in enumerate(loader):
            x, y = d['x'].to(device).float(), d['y'].to(device).float()
            x = x.permute(1,0,2,3,4) # time first batch second
            out = model(x) # forward pass
            loss = loss_masked2(out, y)
            print(f'Loss for {i} datapoint: {loss.item()}')
            avg_loss += loss.item()
            mape_calc = mape_masked(y, out)
            mae_calc = mae_masked(y, out)
            mape_losses.extend(mape_masked(y, out, return_avg=False))
            mae_losses.extend(mae_masked(y, out, return_avg=False))
            print(f'MAPE for {i} datapoint: {mape_calc}')
            
            # save to file for further analysis (scale back to original values using training max and mins)
            x_scales = []
            if args['normalise']:
                for j in range(x.size(0)): # each timestep
                    x_scale, y_scale, out_scale = reverse_scale_outputs(x[j].cpu().numpy()[0], y.cpu().numpy()[0], out.cpu().numpy()[0], 
                                                                        data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
                    x_scales.append(x_scale)
            else:
                for j in range(x.size(0)): # each timestep
                    x_scale, y_scale, out_scale = x[j].cpu().numpy()[0], y.cpu().numpy()[0], out.cpu().numpy()[0]
                    x_scales.append(x_scale)
                    
            x_scales = np.stack(x_scales)
            save_predicted_data(x_scales, y_scale, out_scale, model_name, epoch, eval_args['save_suffix'], t, i)
            
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