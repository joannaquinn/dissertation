import torch.nn as nn
import torch
import numpy as np

import os

from nnets import *
from constants_jo import *
from ndata import DATAMAP, grid_dataset

##################################################################
######## space for constructing various model architectures ######
##################################################################

class NNModel():
    '''
    '''
    def __init__(self, model_name, submodel_name):
        '''
        '''
        self.model_name = model_name
        self.submodel_name = submodel_name
        self.data = DATAMAP(model_name)