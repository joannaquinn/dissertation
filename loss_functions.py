import os
import datetime as dt
import csv
from chardet import detect
import random
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def mse(true, predicted):
    '''
    mean square error
    '''
    return np.mean((true-predicted)**2)

def rmse(true, predicted):
    '''
    root mean square error
    '''
    return np.sqrt(mse(predicted, true))

def mae(true, predicted):
    '''
    mean absolute error
    '''
    return np.mean(np.abs(predicted-true))

def mape(true, predicted):
    '''
    mean absolute percentage error
    multiple by 100 to get percentage value
    warning: cannot handle zero values!
    '''
    return np.mean(np.abs((predicted-true)/true))

def loss_masked(target, pred):
    '''
    mse loss function which masks out points for which we
    don't have data
    uses MSE
    inputs are size b x k x h x w
    '''
    losses = []
    for b in range(target.size(0)): # batch item
        loss_grid = []
        for i in range(target.size(1)): # grid
            loss = 0
            target_grid = target[b][i].cpu().detach().numpy()
            points = target_grid.nonzero()
            for j in range(len(points[0])):
                coord = (int(points[0][j]), int(points[1][j])) # nonzero coordinate
                losses.append((target[b][i][coord[0]][coord[1]] - pred[b][i][coord[0]][coord[1]])**2)
    losses= torch.stack(losses)
    return losses.mean()

def mape_masked(target, pred, return_avg = True):
    target2 = target.cpu().detach().numpy()
    pred2 = pred.cpu().detach().numpy()
    losses = []
    for b in range(target2.shape[0]): # batch item
        loss_grid = []
        for i in range(target2.shape[1]): # grid
            loss = 0
            target_grid = target2[b][i]
            points = target_grid.nonzero()
            for j in range(len(points[0])):
                coord = (int(points[0][j]), int(points[1][j])) # nonzero coordinate
                losses.append(np.abs(target2[b][i][coord[0]][coord[1]] - pred2[b][i][coord[0]][coord[1]])/target2[b][i][coord[0]][coord[1]])
    losses = np.stack(losses)
    if return_avg:
        return losses.mean()
    else:
        return list(losses.flatten())
    
def mae_masked(target, pred, return_avg = True):
    target2 = target.cpu().detach().numpy()
    pred2 = pred.cpu().detach().numpy()
    losses = []
    for b in range(target2.shape[0]): # batch item
        loss_grid = []
        for i in range(target2.shape[1]): # grid
            loss = 0
            target_grid = target2[b][i]
            points = target_grid.nonzero()
            for j in range(len(points[0])):
                coord = (int(points[0][j]), int(points[1][j])) # nonzero coordinate
                losses.append(np.abs(target2[b][i][coord[0]][coord[1]] - pred2[b][i][coord[0]][coord[1]]))
    losses = np.stack(losses)
    if return_avg:
        return losses.mean()
    else:
        return list(losses.flatten())

def mape_masked_unbatch(target, pred, return_avg = True):
    '''
    masked mape calcualtion for unbatched numpy data
    ie shape 1 x h x w
    '''
    losses = []
    for i in range(target.shape[0]): # grid
        target_grid = target[i]
        points = target_grid.nonzero()
        for j in range(len(points[0])):
            coord = (int(points[0][j]), int(points[1][j])) # nonzero coordinate
            losses.append(np.abs(target[i][coord[0]][coord[1]] - pred[i][coord[0]][coord[1]])/target[i][coord[0]][coord[1]])
    losses = np.stack(losses)
    if return_avg:
        return losses.mean()
    else:
        return list(losses.flatten())

def mae_masked_unbatch(target, pred, return_avg = True):
    '''
    masked mae calcualtion for unbatched numpy data
    ie shape 1 x h x w
    '''
    losses = []
    for i in range(target.shape[0]): # grid
        target_grid = target[i]
        points = target_grid.nonzero()
        for j in range(len(points[0])):
            coord = (int(points[0][j]), int(points[1][j])) # nonzero coordinate
            losses.append(np.abs(target[i][coord[0]][coord[1]] - pred[i][coord[0]][coord[1]]))
    losses = np.stack(losses)
    if return_avg:
        return losses.mean()
    else:
        return list(losses.flatten())
    
# def loss_masked(target, pred, loss_fn = nn.MSELoss(reduction='mean')):
#     '''
#     mse loss function which masks out points for which we
#     don't have data
#     inputs are size b x k x h x w
#     '''
#     loss = 0
#     nonzero = target.nonzero()
#     for coord in nonzero:
#         print(target[coord])
#         loss += nn.MSELoss(target[coord], pred[coord])
#     loss = loss/len(coord)
#     return loss