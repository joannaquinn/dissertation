import torch.nn as nn
import torch
import numpy as np

# constants
ACTIVATIONS = {'relu': nn.ReLU,
               'sigmoid': nn.Sigmoid,
               'tanh': nn.Tanh}

POOLINGS = {'max':nn.MaxPool2d,
            'average':nn.AvgPool2d}

def calculate_output_size_conv(height, width, in_channels, out_channels, kernel_size=3, padding=1):
    '''
    calculate the size of output tensor from convolutional layer
    b x c x h x w
    '''
    x = torch.ones(size=(1, in_channels, height, width))
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    out = conv(x)
    return out.size()

def calculate_output_size_pool(height, width, channels, pool_size=2):
    '''
    calculate the size of output tensor from pooling layer
    b x c x h x w
    '''
    x = torch.ones(size=(1, channels, height, width))
    pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    out = pool(x)
    return out.size()

def padding_from_kernel_size(kernel_size):
    '''
    calculate the correct padding value for same convolutions
    '''
    return int(kernel_size//2)

class ConvBlockEncode(nn.Module):
    '''
    Convolutional block
    For use in CNN/CRNN
    inputs will have size b x c x h x w
    Downsampler
    '''
    def __init__(self, in_channels, out_channels, conv_layers=2, kernel_size=3, 
                 pooling='average', pool_size=2, activation='relu', batch_norm=False, skip=False):
        '''
        2d Convolutional layers followed by activation and downsample/pooling
        Parameters
        height, width: int
            height and width of grid tensor
        in_channels: int
            number of input channels of the input tensor
        out_channels: int
            desired output channels
        conv_layers: int >= 1
            number of convolutional layers before pooling
        kernel_size: int
            kernel size of convolution, should be odd unless you want weird padding
        pooling_type: string; 'average' or 'max'
            which pooling function to use
        pool_size: int
            the kernel and stride of pool operation
        activation: string; 'relu' or 'tanh' or 'sigmoid'
            activation function, called after every convolutional layer
        batch_norm: boolean
            whether to apply batch normalisation, applied after activation before pooling
        skip: boolean
            whether to use residual layer/skip connection
        '''
        super(ConvBlockEncode, self).__init__()
        # inputs are size b x c x h x w4
        
        activation = ACTIVATIONS[activation]
        pooling = POOLINGS[pooling]
        padding = padding_from_kernel_size(kernel_size)
        
        self.skip = skip
        self.conv_layers = conv_layers
        self.conv = nn.ModuleList()
        
        # convolutional layers
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding), 
            activation()))
        for i in range(1, conv_layers):
            self.conv.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                activation()))
        
        # batch norm
        if batch_norm:
            self.BN = nn.BatchNorm2d(out_channels)
        else:
            self.BN = nn.Sequential()
        
        # pooling layer
        self.pool = pooling(kernel_size=pool_size, stride=pool_size)
        
    def forward(self, x):
        '''
        forward pass
        x is tensor of size b x c x h x w
        '''
        out = x
        out = self.conv[0](out)
        res = out.clone() # residual
        
        for i in range(1, self.conv_layers):
            out = self.conv[i](out)
            
        if self.skip: # skip connection
            out += res
            
        out = self.BN(out)
        out = self.pool(out)
        return out
    
class ConvBlockDecode(nn.Module):
    '''
    Convolutional block
    For use in CNN/CRNN
    inputs will have size b x c x l
    Upsampler
    '''
    def __init__(self, in_channels, out_channels, conv_layers=2, kernel_size=3, 
                 scale_factor=2, activation='relu', batch_norm=False, skip=False):
        '''
        1d Convolutional layers followed by activation and upsample
        Parameters
        in_channels: int
            number of input channels of the input tensor
        out_channels: int
            desired output channels
        conv_layers: int >= 1
            number of convolutional layers before upsampling
        kernel_size: int
            kernel size of convolution, should be odd unless you want weird padding
        scale_factor: int
            scale factor to upsample by
        activation: string; 'relu' or 'tanh' or 'sigmoid'
            activation function, called after every convolutional layer
        batch_norm: boolean
            whether to apply batch normalisation, applied after activation before pooling
        skip: boolean
            whether to use residual layer/skip connection
        '''
        super(ConvBlockDecode, self).__init__()
        
        activation = ACTIVATIONS[activation]
        padding = padding_from_kernel_size(kernel_size)
        
        self.skip = skip
        self.conv_layers = conv_layers
        self.conv = nn.ModuleList()
        
        # convolutional layers
        self.conv.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding), 
            activation()))
        for i in range(1, conv_layers):
            self.conv.append(nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                activation()))
        
        # batch norm
        if batch_norm:
            self.BN = nn.BatchNorm1d(out_channels)
        else:
            self.BN = nn.Sequential()
        
        # pooling layer
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        
    def forward(self, x):
        '''
        forward pass
        x is tensor of size b x l
        '''
        out = x
        out = self.conv[0](out)
        res = out.clone() # residual
        
        for i in range(1, self.conv_layers):
            out = self.conv[i](out)
            
        if self.skip: # skip connection
            out += res
            
        out = self.BN(out)
        out = self.upsample(out)
        return out

class FCBlock(nn.Module):
    '''
    Fully connected block
    For use in CNN/CRNN
    inputs can be 1d or 2d; size b x c x k or size b x c x h x w
    outputs will be flattened to size b x l
    '''
    def __init__(self, input_size, output_size, fc_layers=1, fc_neurons=1000, activation = 'relu', dropout=0.0):
        '''
        Fully connected layers
        Parameters
        input_size: int
            should be integer, if input is 2d then should be channels x height x width
            if input is 1d then should be channels x length
        output_size:
            size of output tensor (b x output_size)
        fc_layers: int >= 1
            number of fully connected layers. block will have fc_layers + 1 layers
        fc_neurons: int
            number of neurons in hidden layers
        activation: string; 'relu' or 'tanh' or 'sigmoid'
            activation function, called after every fc layer
        dropout: float (0 to 1)
            dropout rate for training
        '''
        super(FCBlock, self).__init__()
        
        activation = ACTIVATIONS[activation]
        self.input_size = input_size
        self.fc_layers = fc_layers
        self.fc = nn.ModuleList()
        
        # fully connected layers
        self.fc.append(nn.Sequential(
            nn.Linear(self.input_size, fc_neurons),
            activation(),
            nn.Dropout(dropout)))
        
        for i in range(1, fc_layers):
            self.fc.append(nn.Sequential(
                nn.Linear(fc_neurons, fc_neurons),
                activation(),
                nn.Dropout(dropout)))
        
        self.final = nn.Sequential(
            nn.Linear(fc_neurons, output_size),
            nn.Identity())
        
    def forward(self, x):
        '''
        forward pass
        x is tensor of size b x c x h x w
        '''
        out = x.view(-1, self.input_size) # flatten
        for i in range(self.fc_layers):
            out = self.fc[i](out)
        out = self.final(out)
        return out

class CNNEncode(nn.Module):
    '''
    Convolutional Neural Network
    No temporal element; inputs are of size b x c x h x w
    Encoder style; 2d grid to 1d vector
    '''
    def __init__(self, height, width, output_size,
                 channels, conv_layers=2, kernel_sizes=None, pooling='average', pool_sizes=None, conv_activation='relu', batch_norm=False, skip=False,
                 fc_layers=1, fc_neurons=1000, fc_activation = 'relu', dropout=0.0):
        '''
        Parameters (that aren't specified in ConvBlock and FCBlock above)
        conv_layers: int
            number of conv layers per block, this is static
        channels: list(int)
            list of channel sizes (ie [3, 16, 32])
            length of list should equal number of blocks + 1
        kernel_sizes: list(int)
            list of kernel sizes
            length of list should be equal to number of blocks
        fc layers: int
        '''
        super(CNNEncode, self).__init__()
        
        if kernel_sizes is None: # default kernel sizes to 3
            kernel_sizes = [3 for _ in range(len(channels)-1)]
        if pool_sizes is None: # default pool size is 2
            pool_sizes = [2 for _ in range(len(channels)-1)]
        
        self.conv_blocks = len(kernel_sizes)
        assert self.conv_blocks == len(channels)-1, 'number of kernel sizes should be same as number of conv blocks'
        assert self.conv_blocks == len(pool_sizes), 'number of pool sizes should be same as number of conv blocks'
        
        self.conv = nn.ModuleList()
        
        for i in range(self.conv_blocks):
            # add convolutional block
            in_channels = channels[i]
            out_channels = channels[i+1]
            kernel_size = kernel_sizes[i]
            pool_size = pool_sizes[i]
            self.conv.append(ConvBlockEncode(in_channels, out_channels, conv_layers, kernel_size, 
                                                pooling, pool_size, conv_activation, batch_norm, skip))
            # calculate size of output
            out = calculate_output_size_pool(height, width, out_channels, pool_size=pool_size)
            height, width = out[2], out[3]
        
        input_size = out_channels*height*width
        self.fc = FCBlock(input_size, output_size, 
                                    fc_layers=fc_layers, fc_neurons=fc_neurons, activation=fc_activation, dropout=dropout)
        
    def forward(self, x):
        out = x
        for i in range(self.conv_blocks):
            out = self.conv[i](out)
        out = self.fc(out)
        return out

class CNNDecode(nn.Module):
    '''
    Convolutional Neural Network
    No temporal element; inputs are of size b x l
    Decoder style; 1d vector to 2d grid
    '''
    def __init__(self, input_size, height, width, channels, out_channels=3, conv_layers=2, kernel_sizes=None, upsample_scales=None, conv_activation='relu', batch_norm=False, skip=False,
                 fc_layers=1, fc_neurons=1000, fc_activation = 'relu', dropout=0.0):
        '''
        Parameters (that aren't specified in ConvBlock and FCBlock above)
        conv_layers: int
            number of conv layers per block, this is static
        channels: list(int)
            list of channel sizes (ie [1, 16, 3])
            length of list should equal number of blocks + 1
        kernel_sizes: list(int)
            list of kernel sizes
            length of list should be equal to number of blocks
        fc layers: int
        '''
        super(CNNDecode, self).__init__()
        
        if channels is None: # default decoder channels 32, conv blocks 2
            channels = [32 for _ in range(3)]
        self.conv_blocks = len(channels)-1
        
        if kernel_sizes is None: # default kernel sizes to 3
            kernel_sizes = [3 for _ in range(self.conv_blocks)]
        if upsample_scales is None: # default upsample scale is 2
            upsample_scales = [2 for _ in range(self.conv_blocks)]

        self.height= height
        self.width = width
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = height*width*self.out_channels
        
        assert self.conv_blocks == len(channels)-1, 'number of kernel sizes should be same as number of conv blocks'
        assert self.conv_blocks == len(upsample_scales), 'number of pool sizes should be same as number of conv blocks'
        assert channels[0] == 1, 'first channel size should be 1'
        
        self.conv = nn.ModuleList()
        
        for i in range(self.conv_blocks):
            # add convolutional block
            in_channels = channels[i]
            out_channels = channels[i+1]
            kernel_size = kernel_sizes[i]
            scale_factor = upsample_scales[i]
            self.conv.append(ConvBlockDecode(in_channels, out_channels, conv_layers=conv_layers, kernel_size=kernel_size, scale_factor=scale_factor, 
                                             activation=conv_activation, batch_norm = batch_norm, skip=skip))
            
        input_size = channels[-1]*np.product(upsample_scales)*self.input_size
        self.fc = FCBlock(input_size, self.output_size, fc_layers=fc_layers, fc_neurons=fc_neurons, 
                                    activation=fc_activation, dropout=dropout)
    def forward(self, x):
        '''
        forward pass
        x is tensor of size b x l
        '''
        out = x.unsqueeze(1) # change to shape b x 1 x l
        for i in range(self.conv_blocks):
            out = self.conv[i](out)
        out = self.fc(out)
        out = out.view(-1, self.out_channels, self.height, self.width)
        return out

class CNNAutoEncoder(nn.Module):
    '''
    Convolutional Neural Network  (Autoencoder)
    '''
    def __init__(self, height, width, in_channels, out_channels,  encoded_size,
                 encode_channels, encode_conv_layers=2, encode_kernel_sizes=None, pooling='average', pool_sizes=None, encode_conv_activation='relu', encode_batch_norm=False, encode_skip=False,
                 encode_fc_layers=1, encode_fc_neurons=1000, encode_fc_activation = 'relu', encode_dropout=0.0,
                 decode_channels=None, decode_conv_layers=2, decode_kernel_sizes=None, upsample_scales=None, decode_conv_activation='relu', decode_batch_norm=False, decode_skip=False,
                 decode_fc_layers=1, decode_fc_neurons=1000, decode_fc_activation = 'relu', decode_dropout=0.0):
        '''
        inputs are 2d grids size b x in_channels x height x width
        encoded to b x encoded_size
        decoded to b x out_channels x height x width
        '''
        super(CNNAutoEncoder, self).__init__()
        
        self.encoder = CNNEncode(height, width, encoded_size,
                                 encode_channels, conv_layers=encode_conv_layers, kernel_sizes=encode_kernel_sizes, pooling=pooling, pool_sizes=pool_sizes, conv_activation=encode_conv_activation, batch_norm=encode_batch_norm, skip=encode_skip,
                                 fc_layers=encode_fc_layers, fc_neurons=encode_fc_neurons, fc_activation=encode_fc_activation, dropout=encode_dropout)
        self.decoder = CNNDecode(encoded_size, height, width, 
                                 decode_channels, out_channels=out_channels, conv_layers=decode_conv_layers, kernel_sizes=decode_kernel_sizes, upsample_scales=upsample_scales, conv_activation=decode_conv_activation, batch_norm=decode_batch_norm, skip=decode_skip,
                                 fc_layers=decode_fc_layers, fc_neurons=decode_fc_neurons, fc_activation=decode_fc_activation, dropout=decode_dropout)
        
    def forward(self, x):
        out = x
        out = self.encoder(out)
        out = self.decoder(out)
        return out
    
class LSTMBlock(nn.Module):
    '''
    Long Short Term Memory block
    for use in CRNN
    inputs will be times series of (encoded) vectors, of size T x B x L
    '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=True):
        '''
        Parameters
        input_size: int
            size of the encoded input tensor
        hidden_size: int
            number of features to use in hidden state
        num_layers: int
            number of recurrent layers (these stack)
        bias: bool
            whether to use bias weights
        dropout: float
            dropout rate for each LSTM layer
        bidirectional: bool
            whether to use bidirectional LSTM
        '''
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
        
    def forward(self, x):
        out = x
        out = self.lstm(out)
        return out

class CRNN(nn.Module):
    '''
    Convolutional Recurrent Neural Network
    Structure: CNN encoder -> LSTM -> CNN decoder
    Inputs to CNN encoder: T x b x c x h x w
    one by one encoded to b x l, then stacked to T x b x l for
    input into LSTM (num_dircetions*lstm_hidden_size)
    then upsampled one by one into decoder to output b x d x h x w
    '''
    def __init__(self, height, width, in_channels, out_channels, encoded_size,
                 encode_channels, encode_conv_layers=2, encode_kernel_sizes=None, pooling='average', pool_sizes=None, encode_conv_activation='relu', encode_batch_norm=False, encode_skip=False,
                 encode_fc_layers=1, encode_fc_neurons=1000, encode_fc_activation = 'relu', encode_dropout=0.0,
                 decode_channels=None, decode_conv_layers=2, decode_kernel_sizes=None, upsample_scales=None, decode_conv_activation='relu', decode_batch_norm=False, decode_skip=False,
                 decode_fc_layers=1, decode_fc_neurons=1000, decode_fc_activation = 'relu', decode_dropout=0.0,
                 lstm_hidden_size=1000, lstm_layers=1, lstm_bias=True, lstm_dropout=0.0, bidirectional=True):
        '''
        Parameters same as autoencoder & for LSTM block
        '''
        super(CRNN, self).__init__()
        
        if bidirectional:
            num_directions=2
        else:
            num_directions=1
        lstm_output_size = num_directions*lstm_hidden_size
        
        self.encoder = CNNEncode(height, width, encoded_size,
                                 encode_channels, conv_layers=encode_conv_layers, kernel_sizes=encode_kernel_sizes, pooling=pooling, pool_sizes=pool_sizes, conv_activation=encode_conv_activation, batch_norm=encode_batch_norm, skip=encode_skip,
                                 fc_layers=encode_fc_layers, fc_neurons=encode_fc_neurons, fc_activation=encode_fc_activation, dropout=encode_dropout)
        
        self.lstm = LSTMBlock(input_size=encoded_size, hidden_size=lstm_hidden_size, num_layers=lstm_layers, bias=lstm_bias, dropout=lstm_dropout, bidirectional=bidirectional)
        
        self.decoder = CNNDecode(lstm_output_size, height, width, 
                                 decode_channels, out_channels=out_channels, conv_layers=decode_conv_layers, kernel_sizes=decode_kernel_sizes, upsample_scales=upsample_scales, conv_activation=decode_conv_activation, batch_norm=decode_batch_norm, skip=decode_skip,
                                 fc_layers=decode_fc_layers, fc_neurons=decode_fc_neurons, fc_activation=decode_fc_activation, dropout=decode_dropout)
        
    def forward(self, x):
        out = x # size T x b x c x h x w
        T = out.size(0)
        outputs = []
        for t in range(T): # encoder
            outputs.append(self.encoder(out[t]))
        out = torch.stack(outputs) # size T x b x encode_size
        out, state = self.lstm(out) # size T x b x encoded_size, total output only
        self.hidden_state, self.cell_state = state[0], state[1] # store for debugging
        out = out[-1] # last timestep only
        out = self.decoder(out)
        return out