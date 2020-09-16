import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from loss_functions import *
from ndata import *
from neval import OUTPUT_DIRS
import seaborn as sns
import matplotlib.pyplot as plt

from model_names import ALL_DATA_DIR, INTERVALS, CELL_SIZES, SEQ_LENS

PLOT_DIRS = {f'{location}_{net}_{interval}min_{cell_size}': os.path.join(ALL_DATA_DIR, location, 'plots', 'rfsi') 
             for net in INTERVALS
             for location in INTERVALS[net]
             for interval in INTERVALS[net][location]
             for cell_size in CELL_SIZES[location]}

def distance_matrix(i, j, grid=None, grid_dim=None):
    '''
    returns euclidean distance matrix between i'jth coordinate and rest of grid
    '''
    assert grid is not None or grid_dim is not None
    if grid is not None:
        grid_dim = grid.shape
    dist = np.array([[np.sqrt((i-l)**2 + (j-k)**2) for k in range(grid_dim[1])] 
                     for l in range(grid_dim[0])])
    return dist

def reverse_scale_outputs_rfsi(y, out, pm_max, pm_min):
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

def df_for_rfsi(x_grid, y_grid, static_points=None, normalise_distances=True, full_df=False, avg_static_pm=False):
    '''
    creates a dataframe where
    each row is a grid coordinate
    contains information on distance to each static sensor point and
    the values of said point
    for use in random forest spatial interpolation
    if full_df, creates a dataframe for which we can do a full grid interpolation (for valid and test data only)
    '''
    if static_points is None:
        static_points = x_grid[0].nonzero()
    num_static_points = len(static_points[0])
    print(f'Number of static points: {num_static_points}')
    grid_dim = x_grid[0].shape
    
    # initialise data
    labels = ['pm','humidity','roads','landuse1','landuse2','landuse3','traffic']
    data = {'current': {'index':[], 'i_coords':[], 'j_coords':[]}}
    distance_matrices = {}
    for l in labels:
        data['current'][l] = []
    for s in range(num_static_points):
        static_point = (int(static_points[0][s]), int(static_points[1][s]))
        distance_matrices[f'static{s}'] = distance_matrix(static_point[0], static_point[1], grid_dim=grid_dim)
        data[f'static{s}'] = {'dist':[]}
        for l in labels:
            data[f'static{s}'][l] = []
    
    index = 0
    for j in range(grid_dim[1]):
        for i in range(grid_dim[0]): # loop through all cells in grid
            index += 1
            if (y_grid[0][i][j] > 0.0) or full_df: # only fill in data if we have it or if full_df is set to true
                data['current']['index'].append(index)
                data['current']['i_coords'].append(i)
                data['current']['j_coords'].append(j)
                data['current']['pm'].append(y_grid[0][i][j])
                for k in range(1,len(labels)):
                    data['current'][labels[k]].append(x_grid[k][i][j]) # spatial data for current cell

                for s in range(num_static_points):
                    static_point = (int(static_points[0][s]), int(static_points[1][s]))
                    data[f'static{s}']['dist'].append(distance_matrices[f'static{s}'][i][j]) # distance from i j th cell to static point
                    for k in range(len(labels)): # all other static point data
                        data[f'static{s}'][labels[k]].append(x_grid[k][static_point[0]][static_point[1]])

        data2 = {f'current_{c}':data['current'][c] for c in data['current']} # modify dict
        for s in range(num_static_points):
            for c in data[f'static{s}']:
                data2[f'static{s}_{c}'] = data[f'static{s}'][c]
    
    # get rid of unnecessary cols
    df = pd.DataFrame(data=data2)
    df = df.drop(columns=[f'static{s}_{c}' for s in range(num_static_points) for c in ['humidity','roads','landuse1','landuse2','landuse3','traffic']])
    
    if normalise_distances: # normalise distances
        distance_max = distance_matrix(0, 0, grid_dim=grid_dim).max()
        for s in range(num_static_points): 
            df[f'static{s}_dist'] = df[f'static{s}_dist']/distance_max
            
    # avg static pms; this removes the static cols and replaces with an average sensor reading across the grid
    if avg_static_pm:
        static_cols = [c for c in list(df.columns) if 'static' in c]
        static_pm_cols = [c for c in static_cols if 'pm' in c]
        df[f'static_pm_avg'] = df[static_pm_cols].mean(axis=1)
        df = df.drop(columns=static_cols)
    return df

def df_for_rfsi_t(seq_x_grid, y_grid, static_points=None, normalise_distances=True, full_df=False, avg_static_pm=False):
    '''
    creates df inputs for spatio-temporal rfsi
    '''
    dfs = []
    T = seq_x_grid.shape[0]
    cols = ['current_index', 'current_i_coords', 'current_j_coords', 
            'current_pm','current_humidity', 'current_roads', 'current_landuse1','current_landuse2', 'current_landuse3', 'current_traffic']
    df = df_for_rfsi(seq_x_grid[T-1], y_grid, static_points=static_points, normalise_distances=normalise_distances, full_df=full_df, avg_static_pm=avg_static_pm)
    dfs.append(df) # final timestamp first
    for t in range(T-2, -1, -1):
        df = df_for_rfsi(seq_x_grid[t], y_grid, static_points=static_points, normalise_distances=normalise_distances, full_df=full_df, avg_static_pm=avg_static_pm)
        df = df.drop(columns=cols)
        df = df.rename(columns={col:f'{col}_{t}' for col in df.columns})
        df[f'timestep_{t}'] = (t+1)/(T-1) # add (normalised) timestep column
        dfs.append(df)
    return pd.concat(dfs, axis=1)

def rfsi(train_df, valid_df=None, n_estimators=100, criterion='mse', max_depth=None, min_impurity_decrease=0.0, random_state=25):
    '''
    perform random forest interpolation on dataframe (constructed as in df_for_rfsi above)
    returns the rfsi model, plus mse and mape on validation data if valid_df supplied
    and coordinate data for reconstructing into grid
    '''
    # format data
    y_train = train_df['current_pm'].to_numpy()
    X_train_with_coords = train_df.drop(columns=['current_pm','current_index']).to_numpy()
    X_train, coords_train = X_train_with_coords[:, 2:], X_train_with_coords[:, :2]
    if valid_df is not None:
        y_valid = valid_df['current_pm'].to_numpy()
        X_valid_with_coords = valid_df.drop(columns=['current_pm','current_index']).to_numpy()
        X_valid, coords_valid = X_valid_with_coords[:, 2:], X_valid_with_coords[:, :2]
    
    # train model
    rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, min_impurity_decrease=min_impurity_decrease, random_state=random_state)
    rfr.fit(X_train, y_train)
    
    if valid_df is not None:
        errors = {}
        out = rfr.predict(X_valid)
        errors['mse'] = mse(y_valid, out)
        errors['mape'] = mape(y_valid, out)
        return rfr, coords_train, out, y_valid, coords_valid, errors
    
    return rfr, coords_train

def rfsi_predict(rfr, test_df):
    '''
    get output from prediction grid from
    input rfsi df
    '''
    y_test = test_df['current_pm'].to_numpy()
    X_test_with_coords = test_df.drop(columns=['current_pm','current_index']).to_numpy()
    X_test, coords = X_test_with_coords[:, 2:], X_test_with_coords[:, :2]
    out = rfr.predict(X_test)
    errors= {}
    errors['mse'] = mse(y_test, out)
    errors['mape'] = mape(y_test, out)
    return out, y_test, coords, errors

def reconstruct_grid_rfsi(out, y, coords, grid_dim, out_grid_filename = None, plot_filename=None, figsize=(10,7)):
    '''
    reconstructs grid(s) out of the outputs from rfsi algorithm
    if data_indices supplied then creates separate grid/plot for each datapoint index
    '''
    out_grid = np.zeros(grid_dim)
    y_grid = np.zeros(grid_dim)
    
    # create grids
    for k in range(out.shape[0]):
        i, j = int(coords[k][0]), int(coords[k][1])
        out_grid[i, j] = out[k]
        y_grid[i, j] = y[k]
        
    if out_grid_filename is not None:
        np.save(out_grid_filename, out_grid)
        
    # plot
    sns.set()
    vmin, vmax = 0, max(out_grid.max(), y_grid.max())
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize, squeeze=True)
    fig1=sns.heatmap(y_grid, ax=axes[0], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    fig2=sns.heatmap(out_grid, ax=axes[1], xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=False);
    
    plt.colorbar(fig1.get_children()[0], ax = axes,orientation = 'horizontal');
    axes[0].set_title('Target data')
    axes[1].set_title('RFSI prediction')
    
    if plot_filename is not None:
        fig.savefig(plot_filename)
    
    return out_grid, y_grid

def rfsi_full_run(model_name, eval_data='valid', rfr=None, use_static_points=True, n_estimators=100,
                  criterion='mse', max_depth=None, min_impurity_decrease=0.0, random_state=25, 
                  interpolate_full_grid=True, avg_static_pm=False):
    '''
    wrapper for rfsi model functions
    eval_data set to 'valid' or 'test'
    '''
    # import data
    data = {}
    if use_static_points:
        static_points = np.load(STATIC_POINTS[model_name])
    else:
        static_points = None
        
    data['train'] = grid_dataset(DATAMAP[model_name]['train'], normalise=True, static_points=static_points)
    data[eval_data] = grid_dataset(DATAMAP[model_name][eval_data], normalise=False, static_points=static_points)
    data[eval_data].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    
    print(f'Number of train datapoints: {len(data["train"])}')
    print(f'Number of test datapoints: {len(data[eval_data])}')    
    example_datapoint = data['train'][0]
    for x in ('x','y'):
        print(f'Example {x} datapoint is shape {example_datapoint[x].detach().numpy().shape}')
    grid_dim = example_datapoint['x'].detach().numpy().shape[-2], example_datapoint['x'].detach().numpy().shape[-1]
    
    # create dataframes
    if rfr is None:
        dsets = ('train',eval_data)
        do_plotting = True
    else:
        dsets = (eval_data,)
        do_plotting = False
        
    rfsi_dfs = {'train':[], eval_data:[], f'{eval_data}_full':[]}
    for dset in dsets:
        for datapoint in data[dset]:
            df = df_for_rfsi(datapoint['x'].detach().numpy(), datapoint['y'].detach().numpy(), static_points=static_points,avg_static_pm=avg_static_pm)
            rfsi_dfs[dset].append(df)
            if dset == eval_data:
                rfsi_dfs[f'{eval_data}_full'].append(df_for_rfsi(datapoint['x'].detach().numpy(), datapoint['y'].detach().numpy(), static_points=static_points, full_df=True,avg_static_pm=avg_static_pm))
            
        rfsi_dfs[f'{dset}_all'] = pd.concat(rfsi_dfs[dset])
    
    # run random forest
    if rfr is None:
        rfr, coords_train, out, y_valid, coords_valid, errors = rfsi(rfsi_dfs['train_all'], valid_df=rfsi_dfs[f'{eval_data}_all'], 
                                                                     n_estimators=n_estimators, criterion='mse', max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, random_state=random_state)
    
    # visualise and save outputs
    errors = []
    output_dir = OUTPUT_DIRS[model_name]
    i = 0
    if avg_static_pm:
        suffix = '_avg_pm'
    else:
        suffix = ''
        
    for datapoint in data[eval_data]:
        y_filename = os.path.join(output_dir, f'{model_name}_rfsi{suffix}_{eval_data}_{i}_y.npy')
        output_filename = os.path.join(output_dir, f'{model_name}_rfsi{suffix}_{eval_data}_{i}_out.npy')
        plot_filename = os.path.join(PLOT_DIRS[model_name], f'{model_name}_rfsi{suffix}_{eval_data}_{i}_heatmap.png')
        
        y_df = rfsi_dfs[eval_data][i]
        out_output, y_output, coords, error = rfsi_predict(rfr, y_df)
        
        # scale back up
        y_output, out_output =  reverse_scale_outputs_rfsi(y_output, out_output, data['train'].get_pm_max(), data['train'].get_pm_min())
        nonzero_points = y_output.nonzero()
        mape_loss = np.mean(np.abs(y_output[nonzero_points]-out_output[nonzero_points])/y_output[nonzero_points])
        #mape_loss = mape(y_output, out_output)
        errors.append(mape_loss)
        
        if interpolate_full_grid: # 
            y_df = rfsi_dfs[f'{eval_data}_full'][i]
            out_output, y_output, coords, _ = rfsi_predict(rfr, y_df)
            y_output, out_output =  reverse_scale_outputs_rfsi(y_output, out_output, data['train'].get_pm_max(), data['train'].get_pm_min())
        
        out_grid, y_grid = reconstruct_grid_rfsi(out_output, y_output, coords, grid_dim, out_grid_filename=output_filename, plot_filename=plot_filename)
        np.save(y_filename, y_grid)
        
        i += 1
    
    if do_plotting:
        cols = rfsi_dfs['train_all'].drop(columns=['current_pm','current_index', 'current_i_coords','current_j_coords']).columns
        print('Train data cols in order of feature importance:')
        print([c for _, c in sorted(zip(rfr.feature_importances_, list(cols)),reverse=True)])
        # plot feature importances
        sns.set()
        fig, ax = plt.subplots(figsize=(10,5));
        plot = sns.barplot([c for _, c in sorted(zip(rfr.feature_importances_, list(cols)),reverse=True)], sorted(rfr.feature_importances_, reverse=True),ax=ax);
        for tick in ax.get_xticklabels():
            tick.set_rotation(65);
        fig = plot.get_figure()
        fig.savefig(os.path.join(PLOT_DIRS[model_name], f'{model_name}_rfsi{suffix}_featureimportance.png'))
    else:
        cols = []
    
    return rfr, cols, rfr.feature_importances_, errors

def rfsi_full_run_location(location, intervals, cell_sizes,avg_static_pm=False, n_estimators=100,max_depth=None):
    fis = {}
    errors = {}
    for interval in intervals:
        errors[interval] = {}
        fis[interval] = {}
        for cell_size in cell_sizes:
            errors[interval][cell_size] = {}
            model_name = f'{location}_cnn_{interval}min_{cell_size}'
            print(model_name)
            
            rfr, cols, fi, val_errors = rfsi_full_run(model_name, eval_data='valid',avg_static_pm=avg_static_pm, n_estimators=n_estimators, max_depth=max_depth)
            fis[interval][cell_size] = list(zip(cols,fi))
            errors[interval][cell_size]['valid'] = val_errors
            
            _, _, _, test_errors = rfsi_full_run(model_name, eval_data='test', rfr=rfr,avg_static_pm=avg_static_pm)
            errors[interval][cell_size]['test'] = test_errors
            
    return errors, fis
            

def rfsi_t_full_run(model_name, seq_len, eval_data='valid', rfr=None, use_static_points=True, n_estimators=100,
                  criterion='mse', max_depth=None, min_impurity_decrease=0.0, random_state=25, 
                  interpolate_full_grid=True,avg_static_pm=False):
    '''
    wrapper for rfsi model functions
    eval_data set to 'valid' or 'test'
    '''
    # import data
    data = {}
    if use_static_points:
        static_points = np.load(STATIC_POINTS[model_name])
    else:
        static_points = None
        
    data['train'] = grid_seq_dataset(DATAMAP[model_name]['train'], seq_len, normalise=True, static_points=static_points)
    data[eval_data] = grid_seq_dataset(DATAMAP[model_name][eval_data], seq_len, normalise=False, static_points=static_points)
    data[eval_data].normalise(data['train'].get_pm_max(), data['train'].get_pm_min(), data['train'].get_humidity_max(), data['train'].get_humidity_min())
    
    print(f'Number of train datapoints: {len(data["train"])}')
    print(f'Number of test datapoints: {len(data[eval_data])}')    
    example_datapoint = data['train'][0]
    for x in ('x','y'):
        print(f'Example {x} datapoint is shape {example_datapoint[x].detach().numpy().shape}')
    grid_dim = example_datapoint['x'].detach().numpy().shape[-2], example_datapoint['x'].detach().numpy().shape[-1]
    
    # create dataframes
    if rfr is None:
        dsets = ('train',eval_data)
        do_plotting = True
    else:
        dsets = (eval_data,)
        do_plotting = False
    
    # create dataframes
    rfsi_dfs = {'train':[], eval_data:[], f'{eval_data}_full':[]}
    for dset in dsets:
        for datapoint in data[dset]:
            df = df_for_rfsi_t(datapoint['x'].detach().numpy(), datapoint['y'].detach().numpy(), static_points=static_points,avg_static_pm=avg_static_pm)
            rfsi_dfs[dset].append(df)
            if dset == eval_data:
                rfsi_dfs[f'{eval_data}_full'].append(df_for_rfsi_t(datapoint['x'].detach().numpy(), datapoint['y'].detach().numpy(), static_points=static_points, full_df=True,avg_static_pm=avg_static_pm))
            
        rfsi_dfs[f'{dset}_all'] = pd.concat(rfsi_dfs[dset])
    
    # run random forest
    if rfr is None:
        rfr, coords_train, out, y_valid, coords_valid, errors = rfsi(rfsi_dfs['train_all'], valid_df=rfsi_dfs[f'{eval_data}_all'], 
                                                                     n_estimators=n_estimators,criterion=criterion, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, random_state=random_state)
        print('Validation RFSI loss (pre scaling): \n', errors)
    
    # visualise and save outputs
    errors = []
    output_dir = OUTPUT_DIRS[model_name]
    i = 0
    if avg_static_pm:
        suffix = '_avg_pm'
    else:
        suffix = ''
        
    for datapoint in data[eval_data]:
        y_filename = os.path.join(output_dir, f'{model_name}_rfsi_t{suffix}_{eval_data}_{i}_y.npy')
        output_filename = os.path.join(output_dir, f'{model_name}_rfsi_t{suffix}_{eval_data}_{i}_out.npy')
        plot_filename = os.path.join(PLOT_DIRS[model_name], f'{model_name}_rfsi_t{suffix}_{eval_data}_{i}_heatmap.png')
        
        y_df = rfsi_dfs[eval_data][i]
        out_output, y_output, coords, error = rfsi_predict(rfr, y_df)
        
        # scale back up
        y_output, out_output =  reverse_scale_outputs_rfsi(y_output, out_output, data['train'].get_pm_max(), data['train'].get_pm_min())
        nonzero_points = y_output.nonzero()
        mape_loss = np.mean(np.abs(y_output[nonzero_points]-out_output[nonzero_points])/y_output[nonzero_points])
        #mape_loss = mape(y_output, out_output)
        print(f'MAPE for {i}th {eval_data} datapoint: {mape_loss}')
        errors.append(mape_loss)
        
        if interpolate_full_grid: # 
            y_df = rfsi_dfs[f'{eval_data}_full'][i]
            out_output, y_output, coords, _ = rfsi_predict(rfr, y_df)
            y_output, out_output =  reverse_scale_outputs_rfsi(y_output, out_output, data['train'].get_pm_max(), data['train'].get_pm_min())
        
        out_grid, y_grid = reconstruct_grid_rfsi(out_output, y_output, coords, grid_dim, out_grid_filename=output_filename, plot_filename=plot_filename)
        np.save(y_filename, y_grid)
        
        i += 1
    
    if do_plotting:
        cols = rfsi_dfs['train_all'].drop(columns=['current_pm','current_index', 'current_i_coords','current_j_coords']).columns
        print('Train data cols in order of feature importance:')
        print([c for _, c in sorted(zip(rfr.feature_importances_, list(cols)),reverse=True)])
    else:
        cols= []
    
    return rfr, cols, rfr.feature_importances_, errors

def rfsi_t_full_run_location(location, intervals, cell_sizes,avg_static_pm=False, n_estimators=100, max_depth=None):
    fis = {}
    errors = {}
    for interval in intervals:
        errors[interval] = {}
        fis[interval] = {}
        seq_len = SEQ_LENS[location][interval]
        for cell_size in cell_sizes:
            errors[interval][cell_size] = {}
            model_name = f'{location}_lstm_{interval}min_{cell_size}'
            print(model_name)
            
            rfr, cols, fi, val_errors = rfsi_t_full_run(model_name, seq_len, eval_data='valid',avg_static_pm=avg_static_pm, n_estimators=n_estimators, max_depth=max_depth)
            fis[interval][cell_size] = list(zip(cols,fi))
            errors[interval][cell_size]['valid'] = val_errors
            
            _, _, _, test_errors = rfsi_t_full_run(model_name, seq_len, eval_data='test', rfr=rfr,avg_static_pm=avg_static_pm)
            errors[interval][cell_size]['test'] = test_errors
            
    return errors, fis