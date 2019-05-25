import torch.nn.functional as F
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn 

class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class NetSharedLayer(torch.nn.Module):
    def __init__(self, hidden_size, dropout_prob, non_linear):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NetSharedLayer, self).__init__()
        self.dropout_prob = dropout_prob 
        self.input_size = 729
        self.output_size = 729
        self.hidden_size = 9*int(hidden_size/9)   
        self.non_linear = non_linear
        
        self.ops = []
        #self.drop = torch.nn.Dropout(p=self.dropout_prob)
        self.conv = torch.nn.Linear(int(self.input_size / 9), int(self.hidden_size/9))
        
        if self.non_linear:
            self.final_layer = torch.nn.Linear(int(self.hidden_size), int(self.output_size))
        
        
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        
        #xx = self.drop(x)
   
        deep_arr = []
        for v in range(9):
            #deep_x = x[:,81*v:81*v+81]
            conv_x = x.narrow(1, v*81, 81)
            conv_x = self.conv(conv_x)
            
            if self.non_linear:
                conv_x = F.leaky_relu(conv_x)

                conv_x = F.dropout(conv_x, p=self.dropout_prob, training=self.training)
            
            deep_arr.append(conv_x)
            
        deep_x = torch.cat(deep_arr, 1)    
        
        if self.non_linear:
            deep_x = self.final_layer(deep_x)
                                
        return deep_x 
    
    def __str__(self):
        return '{} {} {} {} {} {}'.format(
            type(self).__name__, 
            self.dropout_prob, 
            self.input_size, 
            self.hidden_size,
            self.output_size,
            self.non_linear
        )
    
class BigConvNet(torch.nn.Module):
    def __init__(self, num_layers, num_filters, hidden_non_linear, dropout_prob):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(BigConvNet, self).__init__()
        
        
        self.num_layers = num_layers
        #self.n_hidden = n_hidden
        self.num_filters = num_filters
        self.output_channels = 9
        self.output_size = (9)**3
        self.dropout_prob = dropout_prob
        
        self.hidden_non_linear = hidden_non_linear() if hidden_non_linear else None
        
        self.layer_size = self.num_filters * (9)**2
        
        self.ops = []
        #dropbout after relu, batch after linear (conv)
        
        self.ops.append(torch.nn.Conv2d(9, self.num_filters, kernel_size=(9,9), stride=1, padding=4))
        self.ops.append(torch.nn.BatchNorm2d(self.num_filters))
        self.ops.append(hidden_non_linear())
        
        for i in range(self.num_layers-1):
            if dropout_prob:
                self.ops.append(torch.nn.Dropout2d(p=self.dropout_prob))
            self.ops.append(torch.nn.Conv2d(self.num_filters, self.num_filters, 
                                         kernel_size=(3,3), stride=1, padding=1))
            self.ops.append(torch.nn.BatchNorm2d( self.num_filters))
            self.ops.append(hidden_non_linear())
            
        self.ops = ListModule(*self.ops)
        
        self.combined_linear = torch.nn.Linear(
                self.layer_size,  self.output_size)
        
        

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        
        deep_x = x.reshape(-1, 9, 9, 9)

        for op in self.ops:
            deep_x = op(deep_x)

        deep_x = deep_x.reshape(-1, self.layer_size)
            
        
        deep_x = self.combined_linear(deep_x)
        
        return deep_x
        
    
    def __str__(self):
        return '{} {} {} dropout: {}'.format(
            type(self).__name__, 
            self.num_layers, self.num_filters,
            self.hidden_non_linear, 
            self.dropout_prob
        )    

class ResNet(torch.nn.Module):
    def __init__(self, first_net, second_net, stack):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        
        
        self.first_net = first_net
        self.second_net = second_net
        self.stack = stack
        
        
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        
        first_output = self.first_net(x)
        
        if self.stack:
            second_output = self.second_net(first_output)
        else:
            second_output = self.second_net(x)
                                
        return first_output + second_output 
    
    def __str__(self):
        return '{} Stack: {} First Net: {} Second Net: {}'.format(
            type(self).__name__, 
            self.stack, 
            self.first_net, 
            self.second_net
        )












class SharedConvNet(torch.nn.Module):
    def __init__(self, num_layers, num_filters, dropout_prob, linear_last):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SharedConvNet, self).__init__()
        self.dropout_prob = dropout_prob 
        self.input_size = 729
        self.num_layers = num_layers
        self.linear_last = linear_last
        #self.n_hidden = n_hidden
        self.num_filters = num_filters
        self.output_channels = 9
        self.output_size = (9)**3
        self.layer_size = self.num_filters * (9)**2
        
        self.ops = []
        #dropbout after relu, batch after linear (conv)
        
        self.conv = torch.nn.Linear(int(self.input_size / 9), int(self.input_size/9))
        
        self.ops.append(torch.nn.Conv2d(9, self.num_filters, kernel_size=(3,3), stride=1, padding=1))
        self.ops.append(torch.nn.BatchNorm2d(self.num_filters))
        self.ops.append(torch.nn.LeakyReLU())
        
        for i in range(self.num_layers-1):
            self.ops.append(torch.nn.Dropout2d(p=self.dropout_prob))
            self.ops.append(torch.nn.Conv2d(self.num_filters, self.num_filters, 
                                         kernel_size=(3,3), stride=1, padding=1))
            self.ops.append(torch.nn.BatchNorm2d( self.num_filters))
            self.ops.append(torch.nn.LeakyReLU())
            
        
        if self.linear_last:
            self.combined_linear = torch.nn.Linear(self.layer_size,  self.output_size)
        else:
            self.ops.append(torch.nn.Conv2d(self.num_filters, self.output_channels, 
                                         kernel_size=(3,3), stride=1, padding=1))
            
        self.ops = ListModule(*self.ops)
        #self.combined_linear = torch.nn.Linear(self.conv_out,  output_size)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        
        

        deep_arr = []
        for v in range(9):
            #deep_x = x[:,81*v:81*v+81]
            conv_x = x.narrow(1, v*81, 81)
            conv_x = self.conv(conv_x)
            # torch.nn.BatchNorm1d(hidden_size)
            conv_x = F.leaky_relu(conv_x)

            conv_x = F.dropout(conv_x, p=self.dropout_prob, training=self.training)
            
            deep_arr.append(conv_x)
            
        deep_x = torch.cat(deep_arr, 1)   
        
        deep_x = x.reshape(-1, 9, 9, 9)
        
        for op in self.ops:
            deep_x = op(deep_x)
        
        #y_pred = self.combined_linear()  
        
        
        
        if self.linear_last:
            deep_x = deep_x.reshape(-1, self.layer_size)
            return self.combined_linear(deep_x)
        else:
            deep_x = deep_x.reshape(-1, self.output_size)
            return deep_x
    
    def __str__(self):
        return '{} {} {} {} {}'.format(
            type(self).__name__, 
            self.dropout_prob, 
            self.num_layers, self.num_filters, self.linear_last)   
    
    
class ComplexNetShared(torch.nn.Module):
    def __init__(self, hidden_size, dropout_prob, n_hidden=1):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ComplexNetShared, self).__init__()
        self.dropout_prob = dropout_prob 
        self.input_size = 729
        self.output_size = 729
        self.hidden_size = 9*int(hidden_size/9)   
        self.n_hidden = n_hidden
        
        self.ops = []
        #self.drop = torch.nn.Dropout(p=self.dropout_prob)
        self.conv_channels = torch.nn.Linear(int(self.input_size / 9), int(self.hidden_size/9/3))
        
        self.conv_rows = torch.nn.Linear(int(self.input_size / 9), int(self.hidden_size/9/3))
        
        self.conv_cols = torch.nn.Linear(int(self.input_size / 9), int(self.hidden_size/9/3))
        
        
        for i in range(n_hidden):
            self.ops.append(torch.nn.Linear(int(self.hidden_size), int(self.hidden_size)))
            self.ops.append(torch.nn.BatchNorm1d(int(self.hidden_size)))
            self.ops.append(torch.nn.LeakyReLU())
            self.ops.append(torch.nn.Dropout(p=self.dropout_prob))
        
        self.ops.append(torch.nn.Linear(int(self.hidden_size), int(self.output_size)))
        
        self.ops = ListModule(*self.ops)
        
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        
        #xx = self.drop(x)
   
        deep_arr = []
    
        # channels
        for v in range(9):
            conv_x = x.narrow(1, v*81, 81)
            conv_x = self.conv_channels(conv_x)
            conv_x = F.leaky_relu(conv_x)
            conv_x = F.dropout(conv_x, p=self.dropout_prob, training=self.training)
            
            deep_arr.append(conv_x)
            
        # rows    
        
        deep_x = x.reshape(-1, 9, 9, 9)
        deep_x = deep_x.permute(-1, 1, 0, 2).reshape(-1, 9 * 9 * 9)
        for v in range(9):
            conv_x = deep_x.narrow(1, v*81, 81)
            conv_x = self.conv_rows(conv_x)
            conv_x = F.leaky_relu(conv_x)
            conv_x = F.dropout(conv_x, p=self.dropout_prob, training=self.training)
            
            deep_arr.append(conv_x)
            
        deep_x = x.reshape(-1, 9, 9, 9)
        deep_x = deep_x.permute(-1, 2, 0, 1).reshape(-1, 9 * 9 * 9)    
        for v in range(9):
            conv_x = x.narrow(1, v*81, 81)
            conv_x = self.conv_cols(conv_x)
            conv_x = F.leaky_relu(conv_x)
            conv_x = F.dropout(conv_x, p=self.dropout_prob, training=self.training)
            
            deep_arr.append(conv_x)    
            
            
        deep_x = torch.cat(deep_arr, 1)               
                                
        for op in self.ops:
            deep_x = op(deep_x)
                                
        return deep_x 
    
    def __str__(self):
        return '{} {} {} {} {} {}'.format(
            type(self).__name__, 
            self.dropout_prob, 
            self.input_size, 
            self.hidden_size,
            self.output_size,
            self.n_hidden
        )
class ConvNet(torch.nn.Module):
    def __init__(self, num_layers, num_filters, dropout_prob, linear_last):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ConvNet, self).__init__()
        self.dropout_prob = dropout_prob 
        
        self.num_layers = num_layers
        self.linear_last = linear_last
        #self.n_hidden = n_hidden
        self.num_filters = num_filters
        self.output_channels = 9
        self.output_size = (9)**3
        self.layer_size = self.num_filters * (9)**2
        
        self.ops = []
        #dropbout after relu, batch after linear (conv)
        
        self.ops.append(torch.nn.Conv2d(9, self.num_filters, kernel_size=(3,3), stride=1, padding=1))
        self.ops.append(torch.nn.BatchNorm2d(self.num_filters))
        self.ops.append(torch.nn.LeakyReLU())
        
        for i in range(self.num_layers-1):
            self.ops.append(torch.nn.Dropout2d(p=self.dropout_prob))
            self.ops.append(torch.nn.Conv2d(self.num_filters, self.num_filters, 
                                         kernel_size=(3,3), stride=1, padding=1))
            self.ops.append(torch.nn.BatchNorm2d( self.num_filters))
            self.ops.append(torch.nn.LeakyReLU())
            
        
        if self.linear_last:
            self.combined_linear = torch.nn.Linear(self.layer_size,  self.output_size)
        else:
            self.ops.append(torch.nn.Conv2d(self.num_filters, self.output_channels, 
                                         kernel_size=(3,3), stride=1, padding=1))
            
        self.ops = ListModule(*self.ops)
        #self.combined_linear = torch.nn.Linear(self.conv_out,  output_size)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        deep_x = x.reshape(-1, 9, 9, 9)
        
        for op in self.ops:
            deep_x = op(deep_x)
        
        #y_pred = self.combined_linear()  
        
        
        
        if self.linear_last:
            deep_x = deep_x.reshape(-1, self.layer_size)
            return self.combined_linear(deep_x)
        else:
            deep_x = deep_x.reshape(-1, self.output_size)
            return deep_x
    
    def __str__(self):
        return '{} {} {} {} {}'.format(
            type(self).__name__, 
            self.dropout_prob, 
            self.num_layers, self.num_filters, self.linear_last)
        