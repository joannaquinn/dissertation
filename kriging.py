import os
import datetime as dt
import csv
from chardet import detect
import random
import math
import numpy as np
import pandas as pd

from basic_imports import *
from misc_utils import *
from constants_jo import *
from load_files import *
from download_data import *

from loss_functions import rmse, mae, mape, mape_masked_unbatch
from ndata import *
from neval import OUTPUT_DIRS

import pykrige.kriging_tools as kt
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging

PLOT_DIRS = {f'{location}_{net}_{interval}min_{cell_size}': os.path.join(ALL_DATA_DIR, location, 'plots', 'kriging') 
             for net in INTERVALS
             for location in INTERVALS[net]
             for interval in INTERVALS[net][location]
             for cell_size in CELL_SIZES[location]}

def grid_kriging(static_grid, personal_grid, kriging_type = 'ordinary', variogram_model='linear', plot_filename = None, vmin=0, vmax=8):
    '''
    perform kriging on PM values
    static_grid, personal_grid are spatial only (dimension n x m) pm values
    kriging type can be ordinary or universal
    variogram model can be linear, gaussian, exponential, spherical
    if plot filename specified plot heatmaps
    returns 
    '''
    grid_dim = static_grid.shape
    # get position of non zero values, will be our x/y coords
    # note y coords must be reversed
    y_coords, x_coords = static_grid.nonzero()
    pm_values = static_grid[(y_coords, x_coords)]
    
    # create grid of all values
    personal_grid_mask = personal_grid.copy()
    personal_grid_mask[static_grid.nonzero()] = 0
    all_grid = static_grid + personal_grid_mask
    
    # grid objects to interpolate on
    xgrid = np.arange(grid_dim[1]).astype(np.float64)
    ygrid = np.arange(grid_dim[0]).astype(np.float64)
    
    # kriging
    if kriging_type == 'ordinary':
        krige = OrdinaryKriging(x_coords, grid_dim[0]-1-y_coords, pm_values, variogram_model=variogram_model)
    elif kriging_type == 'universal':
        krige = UniversalKriging(x_coords, grid_dim[0]-1-y_coords, pm_values, variogram_model=variogram_model, drift_terms=['regional_linear'])
    z, ss = krige.execute('grid', xgrid, ygrid)
    z_flip = np.flip(z,axis=0) # y coordinate must be flipped to compare to np array indexing
    
    # losses as dictionary
    errors = {}
    predicted = z_flip[all_grid.nonzero()]
    true = all_grid[all_grid.nonzero()]
    errors['rmse'] = rmse(predicted, true)
    errors['mae'] = mae(predicted, true)
    errors['mape'] = mape(predicted, true)
    
    if plot_filename is not None:
        plot_kriging_heatmaps(all_grid, static_grid, z_flip, plot_filename = plot_filename, vmin=vmin, vmax=vmax)
    
    return z_flip, errors

def plot_kriging_heatmaps(all_grid, static_grid, z_flip, plot_filename = None, 
                          figsize=(15,7), vmin=0, vmax=8):
    sns.set()
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=figsize, squeeze=True)
    fig1=sns.heatmap(all_grid, ax=axes[0], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    fig2=sns.heatmap(static_grid, ax=axes[1], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    fig3=sns.heatmap(z_flip, ax=axes[2], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    
    plt.colorbar(fig1.get_children()[0], ax = axes,orientation = 'horizontal');
    axes[0].set_title('All data')
    axes[1].set_title('Static points only')
    axes[2].set_title('Kriging output')
    
    if plot_filename is not None:
        fig.savefig(plot_filename)
        
def create_kriged_grid(static_temporal_grid, personal_temporal_grid, kriging_type = 'ordinary', variogram_model='linear', 
                       grid_filename = None, plot_filename = None, static_points_filename = None, vmin=0, vmax=8):
    '''
    creates kriged grids for input into machine learning models
    will be dimension T x N x M
    static_temporal_grid and personal_temporal_grid must have same dimensions
    static_points_filename will be where static points information is saved
    '''
    T = static_temporal_grid.shape[0]
    assert T == personal_temporal_grid.shape[0], 'grid must be of same temporal dimension'
    
    zs = []
    for t in range(T):
        static_grid = static_temporal_grid[t]
        personal_grid = personal_temporal_grid[t]
        personal_grid[static_grid.nonzero()] = 0
        all_grid = static_grid + personal_grid
        if np.count_nonzero(all_grid) > 2:
            z_flip, errors = grid_kriging(all_grid, all_grid, kriging_type=kriging_type, variogram_model=variogram_model, plot_filename=f'{plot_filename[:-4]}_{t}.png', vmin=vmin, vmax=vmax)
            zs.append(np.expand_dims(z_flip, 0))
        else:
            print(f'Not enough cell values at timestep {t}, will be set to previous step')
            z_flip = zs[-1]
            zs.append(z_flip)
    z = np.concatenate(zs, axis=0)
    
    if grid_filename is not None:
        np.save(grid_filename, z.filled(0))
    if static_points_filename is not None:
        np.savetxt(static_points_filename, static_grid.nonzero(),fmt='%i')
        
    return z

def create_nn_input_grid(static_temporal_grid, personal_temporal_grid, grid_filename = None, static_points_filename = None):
    T = static_temporal_grid.shape[0]
    assert T == personal_temporal_grid.shape[0], 'grid must be of same temporal dimension'
    
    static_points = static_temporal_grid[0].nonzero()
    print(f'Number of static points: {len(static_points[0])}')
    
    zs = []
    for t in range(T):
        static_grid = static_temporal_grid[t]
        personal_grid = personal_temporal_grid[t]
        personal_grid[static_grid.nonzero()] = 0
        all_grid = static_grid + personal_grid
        if np.count_nonzero(all_grid) >= len(static_points[0]):
            zs.append(all_grid)
        else:
            print(f'Not enough static cell values at timestep {t}, will be set to previous step')
            all_grid = zs[-1]
            zs.append(all_grid)
    z = np.stack(zs)
    
    if grid_filename is not None:
        np.save(grid_filename, z)
    if static_points_filename is not None:
        np.savetxt(static_points_filename, static_points,fmt='%i')
        
    return z

def reverse_scale_outputs_kriging(y, out, pm_max, pm_min):
    '''
    scale outputs of model back up to original values
    y and out are numpy array outputs of rfsi_predict
    '''
    y_scale = np.zeros(y.shape)
    out_scale = np.zeros(y.shape)
    denom = pm_max[0] - pm_min[0]
    if denom == 0:
        denom = 1
    y_scale = y*denom + pm_min[0]
    out_scale = out*denom + pm_min[0]
    return y_scale, out_scale

def krige_all(model_name):
    '''
    wrapper for performing kriging on all data in a model
    returns errors and saves grids
    '''
    # import data
    static_points = np.load(STATIC_POINTS[model_name])
    data = {}
    for t in ('train','valid','test'):
    #for t in ('test',):
        data[t] = grid_dataset(DATAMAP[model_name][t], static_points=static_points, normalise=False)
        #data[t].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    # perform kriging on all data
    errors = {'ordinary':{'linear':[], 'gaussian':[], 'exponential':[]},
              'universal':{'linear':[], 'gaussian':[], 'exponential':[]}}
    for t in data:
        for i in range(len(data[t])):
            datapoint = data[t][i]
            static_grid, personal_grid = datapoint['x'][0].detach().numpy(), datapoint['y'][0].detach().numpy() # get pm only grids
            for kriging_type in ('ordinary','universal'):
                for vm in ('linear','gaussian','exponential'):
                    # kriging
                    try:
                        out, _ = grid_kriging(static_grid, personal_grid, kriging_type=kriging_type, variogram_model=vm)
                        cont = True
                    except ValueError: # ValueError: zero-size array to reduction operation maximum which has no identity
                        print(f'Error at {i}th datapoint of {t} set; {kriging_type},{vm}')
                        cont = False
                    if cont:
                        y_scale, out_scale = personal_grid, out
                        out_scale = out_scale.filled(fill_value=0)
                        #y_scale, out_scale = reverse_scale_outputs_kriging(personal_grid, out, data['train'].get_pm_max(), data['train'].get_pm_min())
                        y_scale, out_scale = np.expand_dims(y_scale, 0), np.expand_dims(out_scale, 0) # reshape to have 3 axes
                        
                        # mape
                        nonzero_points = y_scale.nonzero()
                        for k in range(len(static_points[0])):
                            n, m = static_points[0][k], static_points[1][k]
                            y_scale[0][n,m] = static_grid[n,m]
                        mape_loss = np.mean(np.abs(y_scale[nonzero_points]-out_scale[nonzero_points])/y_scale[nonzero_points])
                        errors[kriging_type][vm].append(mape_loss)
                        
                        # save outputs for plotting later
                        output_dir = OUTPUT_DIRS[model_name]
                        np.save(os.path.join(output_dir, f'{model_name}_kriging_{kriging_type}_{vm}_{t}_{i}_y.npy'), y_scale)
                        np.save(os.path.join(output_dir, f'{model_name}_kriging_{kriging_type}_{vm}_{t}_{i}_out.npy'), out_scale)
                        print(f'MAPE for {i}th datapoint of {t} set; {kriging_type}, {vm}: {mape_loss}')
                        
                        # plot
                        plot_filename = os.path.join(PLOT_DIRS[model_name], f'{model_name}_kriging_{kriging_type}_{vm}_{t}_{i}.png')
                        vmin = 0
                        vmax = max([personal_grid.max(), static_grid.max(), out.max()])
                        plot_kriging_heatmaps(personal_grid, static_grid, out, plot_filename=plot_filename, vmin=vmin, vmax=vmax)
    return errors

def krige_all_location(location, intervals, cell_sizes):
    errors = {}
    for interval in intervals:
        errors[interval] = {}
        for cell_size in cell_sizes:
            errors[interval][cell_size] = {}
            model_name = f'{location}_cnn_{interval}min_{cell_size}'
            print(model_name)
            krige_error = krige_all(model_name)
            for kriging_type in krige_error:
                errors[interval][cell_size][kriging_type] = {}
                for vm in krige_error[kriging_type]:
                    errors[interval][cell_size][kriging_type][vm] = np.mean(krige_error[kriging_type][vm])
    return errors