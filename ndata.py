import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

from nnets import *
from constants_jo import *
from grid_system import pm_grid_from_file

from model_names import ALL_DATA_DIR, INTERVALS, CELL_SIZES

DATAMAP = {
    f'{location}_{net}_{interval}min_{cell_size}': {t: os.path.join(ALL_DATA_DIR, location, 'data', f'{location}_{net}_{interval}min_{cell_size}', t) for t in ('train','valid','test')}
    for net in INTERVALS
    for location in INTERVALS[net]
    for interval in INTERVALS[net][location]
    for cell_size in CELL_SIZES[location]
    }

STATIC_POINTS = {
    f'{location}_{net}_{interval}min_{cell_size}': os.path.join(root_data_dir, location, 'data', f'{location}_{net}_{interval}min_{cell_size}', 'static_points.npy')
    for net in INTERVALS
    for location in INTERVALS[net]
    for interval in INTERVALS[net][location]
    for cell_size in CELL_SIZES[location]
    }

def min_max_scale(array, min_val, max_val):
    denom = max_val - min_val
    if denom == 0:
        denom = 1
    return (array-min_val)/denom

def reverse_scale_outputs(x, y, out, pm_max, pm_min, humidity_max, humidity_min):
    '''
    scale outputs of model back up to original values
    y and out are converted to numpy
    assume no batch dimension
    '''
    x_scale = np.zeros(x.shape)
    y_scale = np.zeros(y.shape)
    out_scale = np.zeros(y.shape)
    for i in range(len(pm_max)): # pm data
        denom = pm_max[i] - pm_min[i]
        if denom == 0:
            denom = 1
        y_scale[i] = y[i]*denom + pm_min[i]
        out_scale[i] = out[i]*denom + pm_min[i]
        x_scale[i] = x[i]*denom + pm_min[i]
        
    i = len(pm_max)+1 # humidity
    denom = humidity_max - humidity_min
    if denom == 0:
        denom = 1
    x_scale[i] = x[i]*denom + humidity_min
    
    i = len(pm_max)+2 # roads
    x_scale[i] = x[i]*5
    
    return x_scale, y_scale, out_scale

class grid_dataset(Dataset):
    '''
    data arrays of size d x h x w
    first dim should be (x pm_cols) x (y pm_cols) x humid x road type x landuse1 x landuse2 x landuse3 x junction
    only train set should be normalise=True, then valid and test use the train data max/min scaling values
    '''
    def __init__(self, directory, pm_cols=(0,), normalise=False, static_points=None):
        self.xdata = []
        self.ydata = []
        self.pm_max = [0 for _ in range(len(pm_cols))]
        self.pm_min = [0 for _ in range(len(pm_cols))]
        
        files = sorted(os.listdir(directory))
        for f in files:
            filepath = os.path.join(directory, f)
            grid = np.load(filepath)
            xgrid = np.concatenate([grid[:len(pm_cols)], grid[len(pm_cols)*2:]])
            if np.abs(xgrid).sum() == 0:
                print(f'Warning: 0 PM values for static data in {filepath}')
                
            if static_points is not None: # mask the static pm points where necessary
                mask = np.zeros((xgrid.shape[1], xgrid.shape[2]))
                for k in range(len(static_points[0])):
                    i, j = static_points[0][k], static_points[1][k]
                    mask[i, j] = 1.0
                xgrid[0] = xgrid[0]*mask
                    
            ygrid = grid[len(pm_cols):len(pm_cols)*2]
            if np.abs(ygrid).sum() == 0:
                print(f'Warning: 0 PM values for personal data in {filepath}')
            self.xdata.append(xgrid)
            self.ydata.append(ygrid)
            
        # get max/min humidity
        self.humidity_max = max([float(self.xdata[t][self.ydata[0].shape[0]+1].max()) for t in range(len(self.xdata))])
        self.humidity_min = min([float(self.xdata[t][self.ydata[0].shape[0]+1].min()) for t in range(len(self.xdata))])
        
        # get max/min pm
        for t in range(len(self.ydata)): # loop through all datapoints
            for i in range(self.ydata[0].shape[0]): # pm values
                pm_max = max(self.pm_max[i], self.xdata[t][i].max(), self.ydata[t][i].max())
                self.pm_max[i] = pm_max
            
        if normalise: # min max scale the data
            for t in range(len(self.ydata)): # loop through all datapoints
                for i in range(self.ydata[0].shape[0]): # pm values
                    max_val = self.pm_max[i]
                    min_val = self.pm_min[i]
                    self.xdata[t][i] = min_max_scale(self.xdata[t][i], min_val, max_val)
                    self.ydata[t][i] = min_max_scale(self.ydata[t][i], min_val, max_val)
                    
                i = self.ydata[0].shape[0]+1 #humidity
                max_val = self.humidity_max
                min_val = self.humidity_min
                self.xdata[t][i] = min_max_scale(self.xdata[t][i], min_val, max_val)
                
                i = self.ydata[0].shape[0]+2 # road
                max_val = 5 # max road type
                min_val = 0
                self.xdata[t][i] = min_max_scale(self.xdata[t][i], min_val, max_val)
                
                #landuse and jn are one hot, so already within [0,1]
            
        self.xdata = [torch.tensor(grid) for grid in self.xdata]
        self.ydata = [torch.tensor(grid) for grid in self.ydata]
            
    def __len__(self):
        return len(self.xdata)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist
        sample = {'x': self.xdata[index], 'y': self.ydata[index]}
        return sample

    def get_pm_max(self):
        return self.pm_max

    def get_pm_min(self):
        return self.pm_min

    def get_humidity_max(self):
        return self.humidity_max

    def get_humidity_min(self):
        return self.humidity_min
    
    def normalise(self, pm_max, pm_min, humidity_max, humidity_min):
        '''
        scale the data using given min max scaling
        intended to scale the validation and test sets after retrieving the max/mins from training
        '''
        for t in range(len(self.ydata)): # loop through all datapoints
            for i in range(self.ydata[0].shape[0]): # pm values
                max_val = pm_max[i]
                min_val = pm_min[i]
                self.xdata[t][i] = min_max_scale(self.xdata[t][i], min_val, max_val)
                self.ydata[t][i] = min_max_scale(self.ydata[t][i], min_val, max_val)
                
            i = self.ydata[0].shape[0]+1 #humidity
            max_val = humidity_max
            min_val = humidity_min
            self.xdata[t][i] = min_max_scale(self.xdata[t][i], min_val, max_val)
            
            i = self.ydata[0].shape[0]+2 # road
            max_val = 5 # max road type
            min_val = 0
            self.xdata[t][i] = min_max_scale(self.xdata[t][i], min_val, max_val)
    
    def all_data(self):
        '''
        return the whole dataset as a numpy array
        useful for random forest
        '''
        return np.stack(self.xdata), np.stack(self.ydata)
    
    def get_static_points(self):
        min_static_points = 10000
        i = 0
        for t in range(len(self.xdata)):
            x = self.xdata[t].detach().numpy()
            static_points = x[0].nonzero()
            num_static_points = len(static_points[0])
            if num_static_points < min_static_points:
                min_static_points = num_static_points
                i = t
        print(f'Min of {min_static_points} static points at timestep {i}')
        static_points = self.xdata[i].detach().numpy()[0].nonzero()
        return static_points
            
class grid_seq_dataset(grid_dataset):
    '''
    grid sequence dataset where intead of sampling individual grids,
    we sample time series of length seq_len
    '''
    def __init__(self, directory, seq_len, pm_cols=(0,), normalise=False, static_points=None):
        super().__init__(directory, pm_cols=(0,), normalise=normalise, static_points=static_points) # call init of grid_dataset
        self.directory = directory
        self.seq_len = seq_len
        
        self.seq_xdata = []
        self.seq_ydata = []
        num_samples = len(self.xdata)
        for i in range(num_samples-seq_len):
            x = self.xdata[i:i+seq_len] # first seq_len elements of time series
            y = self.ydata[i+seq_len] # next element of time series
            self.seq_xdata.append(x)
            self.seq_ydata.append(y)
            
    def __len__(self):
        return len(self.seq_xdata)
            
    def normalise(self, pm_max, pm_min, humidity_max, humidity_min):
        '''
        scale the data using given min max scaling
        intended to scale the validation and test sets after retrieving the max/mins from training
        has to re-form seqeuence data
        '''
        super().normalise(pm_max, pm_min, humidity_max, humidity_min) # call normalisation on self.xdata and self.ydata from grid_dataset class

        self.seq_xdata = []
        self.seq_ydata = []
        num_samples = len(self.xdata)
        seq_len = self.seq_len
        for i in range(num_samples-seq_len):
            x = self.xdata[i:i+seq_len] # first seq_len elements of time series
            y = self.ydata[i+seq_len] # next element of time series
            self.seq_xdata.append(x)
            self.seq_ydata.append(y)
    
    def __getitem__(self, index):
        '''
        returns tensors of size 
        T x c x h x w (x)
        c x h x w (y)
        '''
        if torch.is_tensor(index):
            index = index.tolist
        sample = {'x': torch.stack(self.seq_xdata[index]), 'y': self.seq_ydata[index]}
        return sample
    
def grid_dataset_wrapper(location, cell_size, interval, normalise=True):
    '''
    wrapper for retrieving datasets
    '''
    train_data = grid_dataset(DATAMAP[f'{location}_cnn_{interval}min_{cell_size}']['train'], normalise=normalise)
    valid_data = grid_dataset(DATAMAP[f'{location}_cnn_{interval}min_{cell_size}']['valid'], normalise=False)
    test_data = grid_dataset(DATAMAP[f'{location}_cnn_{interval}min_{cell_size}']['test'], normalise=False)
    
    if normalise:
        valid_data.normalise(train_data.get_pm_max(), train_data.get_pm_min(), train_data.get_humidity_max(), train_data.get_humidity_min())
        test_data.normalise(train_data.get_pm_max(), train_data.get_pm_min(), train_data.get_humidity_max(), train_data.get_humidity_min())
    
    print(f'Train data has {len(train_data)} datapoints.')
    print(f'Train data has {len(valid_data)} datapoints.')
    print(f'Train data has {len(test_data)} datapoints.')
    
    return train_data, valid_data, test_data

def grid_seq_dataset_wrapper(location, cell_size, interval, seq_len, normalise=True):
    '''
    wrapper for retrieving datasets
    '''
    train_data = grid_seq_dataset(DATAMAP[f'{location}_lstm_{interval}min_{cell_size}']['train'], seq_len, normalise=normalise)
    valid_data = grid_seq_dataset(DATAMAP[f'{location}_lstm_{interval}min_{cell_size}']['valid'], seq_len, normalise=False)
    test_data = grid_seq_dataset(DATAMAP[f'{location}_lstm_{interval}min_{cell_size}']['test'], seq_len, normalise=False)
    
    if normalise:
        valid_data.normalise(train_data.get_pm_max(), train_data.get_pm_min(), train_data.get_humidity_max(), train_data.get_humidity_min())
        test_data.normalise(train_data.get_pm_max(), train_data.get_pm_min(), train_data.get_humidity_max(), train_data.get_humidity_min())
    
    print(f'Train data has {len(train_data)} datapoints.')
    print(f'Train data has {len(valid_data)} datapoints.')
    print(f'Train data has {len(test_data)} datapoints.')
    
    return train_data, valid_data, test_data