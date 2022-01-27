import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters


class DNN(nn.Module):
    """
    Simple fully conected network with 7 layers of 30 neurons
    """
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()                                                  # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(2, 30))                     # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())                              # First activation Layer

        for num in range(2, 7):                                                     # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))       # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                 # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(30, 3))                 # Output Layer
        self.num_layers = 7
    # Forward Feed
    def forward(self, x):
        return self.net(x)
    
    
#TODO: implement other networks

class DNN2(nn.Module):
    def __init__(self, num_neurons, output):
        """
            Defined model
        Input
        -----
        num_neurons: number of desired neurons
        output: number of variables
        
        Output
        ------
        z: shape [# points, # variables]
        """
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(2, num_neurons)
        self.hidden = nn.Linear(num_neurons,num_neurons)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(num_neurons, output)
        
        # Define sigmoid activation and softmax output 
        self.tanh = nn.Tanh()
        
    def net(self, X):
        # Pass the input tensor through each of our operations

        x = self.input(X)
        h = self.tanh(x)
        
        u = self.input(X)
        u = self.tanh(u)
        
        v = self.input(X)
        v = self.tanh(v)
        
        z = self.hidden(h)
        z = self.tanh(z)
        
        for i in range(2, 7):
            # ones of shape z
            o = torch.ones_like(z)

            # h = [(o-z) * u] + z * v
            h = torch.add(torch.mul((torch.sub(o,z)),u),torch.mul(z,v))
            
            z = self.hidden(h)
            z = self.tanh(z)

        z = self.output(z)

        return z

    def forward(self,x):
        return self.net(x)
        
class tTanh(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, beta = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(tTanh,self).__init__()
        #self.in_features = in_features

        # initialize alpha
        if beta == None:
            self.beta = Parameter(torch.tensor(1.0)) # create a tensor out of alpha
        else:
            self.beta = Parameter(torch.tensor(beta)) # create a tensor out of alpha
            
        self.tanh = nn.Tanh() # Tanh needs to be defined here 

  
        self.beta.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        x = (self.beta*x)
        return self.tanh(x)


class DNN3(nn.Module):
    def __init__(self,num_neurons,output):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(2, num_neurons) #p and rho
        self.hidden = nn.Linear(num_neurons,num_neurons)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(num_neurons, output-1)
        self.output2 = nn.Linear(num_neurons, 1)
        # Define sigmoid activation and softmax output 
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        #self.tanh = tTanh()
        
    def net(self, X):
        # Pass the input tensor through each of our operations
        
        
        x1 = self.input(X) #rho and p
        h1 = self.relu(x1)
        
        x2 = self.input(X) #v
        h2 = self.tanh(x2)
        
        for i in range(2, 10):
            h1 = self.hidden(h1)
            h1 = self.relu(h1)
            h2 = self.hidden(h2)
            h2 = self.tanh(h2)
        
        #z = torch.add(h1,h2)
        z1 = self.output(h1)
        z2 = self.output2(h2)
        z = torch.cat([z1,z2],1)
   
        return z

    def forward(self,x):
        return self.net(x)