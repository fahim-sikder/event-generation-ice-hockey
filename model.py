import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CommonGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, mode = 'GRU', activation_fn = torch.sigmoid):
        
        super(CommonGRU, self).__init__()
        
        if mode == 'GRU':
            
        
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            
            
        else:
            
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)
        #self.linear = nn.DataParallel(self.linear) # parallel GPU
        
        self.relu = nn.LeakyReLU(0.2)
        
        self.activation_fn = activation_fn
        
    def forward(self, x):
        
        output, _ = self.rnn(x)
        
#         output = self.relu(output)
        
        output = self.linear(output)
        
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        
        
        return output