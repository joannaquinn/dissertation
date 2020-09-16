import argparse
import os

def parse_arg(arg):
    if type(arg) == str:
        return parse_string(arg)
    else:
        return arg

def parse_string(string):
    string = string.lower()
    if string == '':
        return string
    if string == 't' or string == 'true':
        return True
    elif string == 'f' or string == 'false':
        return False
    elif string[0] == '[':
        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.replace(' ', '')
        return [int(x) for x in string.split(',')]
    elif string == 'none':
        return None
    else:
        return string
    
class CNNRunParameters():
    '''
    import model parameters for training NNs
    height, width, in_channels, out_channels,  encoded_size=100,
    encode_channels=[in_channels, 16, 32], encode_conv_layers=2, encode_kernel_sizes=None, pooling='average', pool_sizes=None, encode_conv_activation='relu', encode_batch_norm=True, encode_skip=False,
    encode_fc_layers=1, encode_fc_neurons=1000, encode_fc_activation = 'relu', encode_dropout=0.0,
    decode_channels=[1, 3, 3], decode_conv_layers=2, decode_kernel_sizes=None, upsample_scales=None, decode_conv_activation='relu', decode_batch_norm=True, decode_skip=False,
    decode_fc_layers=1, decode_fc_neurons=1000, decode_fc_activation = 'relu', decode_dropout=0.0
    lstm_hidden_size=1000, lstm_layers=1, lstm_bias=True, lstm_dropout=0.0, bidirectional=True
    '''
    def __init__(self):
        parser = argparse.ArgumentParser(description='Arguments for running model.')
        
        # run arguments
        parser.add_argument('--model_name', nargs="?", type=str, default='edinburgh_2018_cnn_48', help='Model name')
        parser.add_argument('--optimiser', nargs="?", type=str, default='adam', help='Which optimiser (adam or rmsprop)')
        parser.add_argument('--learning_rate', nargs="?", type=float, default=0.0001, help='Learning rate for optimiser')
        parser.add_argument('--weight_decay', nargs="?", type=float, default=0.0001, help='Weight decay rate for optimiser')
        parser.add_argument('--num_epochs', nargs="?", type=int, default=10, help='Max epochs')
        parser.add_argument('--batch_size', nargs="?", type=int, default=1, help='Batch size')
        parser.add_argument('--save', nargs="?", type=str, default=False, help='Whether to save model weights')
        parser.add_argument('--save_suffix', nargs="?", type=str, default='', help='Suffix to add to .pth file')
        parser.add_argument('--normalise', nargs="?", type=str, default='True', help='Apply min max scaling to input data')
        parser.add_argument('--start_epoch', nargs="?", type=int, default=-1, help='Epoch to load from (if -1 start from fresh)')
        parser.add_argument('--patience', nargs="?", type=int, default=20, help='Patience for early stopping')
        parser.add_argument('--print_epochs', nargs="?", type=str, default='True', help='Whether to print epochs')
        
        # encoder arguments
        parser.add_argument('--encoded_size', nargs="?", type=int, default=100, help='Size of encoded tensor')
        parser.add_argument('--encode_channels', nargs="?", type=str, default='[7, 16, 32]', help='Encoder channel sizes')
        parser.add_argument('--encode_conv_layers', nargs="?", type=int, default=2, help='Number of layers in each encoder conv block')
        parser.add_argument('--encode_kernel_sizes', nargs="?", type=str, default='None', help='Size of encoded tensor')
        parser.add_argument('--pooling', nargs="?", type=str, default='average', help='Pooling type (average or max)')
        parser.add_argument('--pool_sizes', nargs="?", type=str, default='None', help='Pooling kernel sizes to use for each encoder conv block')
        parser.add_argument('--encode_conv_activation', nargs="?", type=str, default='relu', help='Activation function for encoder conv layers (relu, sigmoid or tanh')
        parser.add_argument('--encode_batch_norm', nargs="?", type=str, default='True', help='Use batch normalisation on encoder conv blocks')
        parser.add_argument('--encode_skip', nargs="?", type=str, default='False', help='Use skip connections on encoder conv blocks')
        parser.add_argument('--encode_fc_layers', nargs="?", type=int, default=1, help='Number of fc layers for encoder network')
        parser.add_argument('--encode_fc_neurons', nargs="?", type=int, default=1000, help='Number of neurons on encoder fc hidden layers')
        parser.add_argument('--encode_fc_activation', nargs="?", type=str, default='relu', help='Activation function for encoder fc layers')
        parser.add_argument('--encode_dropout', nargs="?", type=float, default=0.0, help='Dropout rate for encoder fc layers')
        
        # decoder arguments
        parser.add_argument('--decode_channels', nargs="?", type=str, default='[1, 1, 1]', help='decoder channel sizes')
        parser.add_argument('--decode_conv_layers', nargs="?", type=int, default=2, help='Number of layers in each decoder conv block')
        parser.add_argument('--decode_kernel_sizes', nargs="?", type=str, default='None', help='Size of decoded tensor')
        parser.add_argument('--upsample_scales', nargs="?", type=str, default='None', help='Upsample scale for each decoder conv block')
        parser.add_argument('--decode_conv_activation', nargs="?", type=str, default='relu', help='Activation function for decoder conv layers (relu, sigmoid or tanh')
        parser.add_argument('--decode_batch_norm', nargs="?", type=str, default='True', help='Use batch normalisation on decoder conv blocks')
        parser.add_argument('--decode_skip', nargs="?", type=str, default='False', help='Use skip connections on decoder conv blocks')
        parser.add_argument('--decode_fc_layers', nargs="?", type=int, default=1, help='Number of fc layers for decoder network')
        parser.add_argument('--decode_fc_neurons', nargs="?", type=int, default=1000, help='Number of neurons on decoder fc hidden layers')
        parser.add_argument('--decode_fc_activation', nargs="?", type=str, default='relu', help='Activation function for decoder fc layers')
        parser.add_argument('--decode_dropout', nargs="?", type=float, default=0.0, help='Dropout rate for decoder fc layers')
        
        # RNN arguments (only relevant for LSTM models)
        parser.add_argument('--seq_len', nargs="?", type=int, default=5, help='Sequence length for LSTM')
        parser.add_argument('--lstm_hidden_size', nargs="?", type=int, default=1000, help='Size of hidden layers for LSTM')
        parser.add_argument('--lstm_layers', nargs="?", type=int, default=1, help='Number of LSTM layers')
        parser.add_argument('--lstm_bias', nargs="?", type=str, default=True, help='Whether to use bias vector in LSTM')
        parser.add_argument('--lstm_dropout', nargs="?", type=float, default=0.0, help='Dropout rate for LSTM')
        parser.add_argument('--bidirectional', nargs="?", type=str, default=True, help='Whether to use a bidirectional LSTM')
        
        args = parser.parse_args()
        self.arg_dict = {key:parse_arg(value) for (key, value) in vars(args).items()}
    
    @property
    def args(self):
        return self.arg_dict
    
class TestParameters():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Arguments for running model.')
        
        # evaluation model arguments
        parser.add_argument('--model_name', nargs="?", type=str, default='edinburgh_2018_cnn_48', help='Model name')
        parser.add_argument('--epoch', nargs="?", type=int, default=9, help='Epoch number')
        parser.add_argument('--save_suffix', nargs="?", type=str, default='', help='Suffix for .pth file')
        parser.add_argument('--data', nargs="?", type=str, default='valid', help='dataset to evaluate (valid or test)')
        parser.add_argument('--plot', nargs="?", type=str, default='True', help='Whether to plot outputs')
        
        args = parser.parse_args()
        self.arg_dict = {key:parse_arg(value) for (key, value) in vars(args).items()}
        
    @property
    def args(self):
        return self.arg_dict
    
if __name__ == '__main__':
    params = CNNRunParameters()
    print(params.args)