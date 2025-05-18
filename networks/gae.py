import torch.nn as nn
import torch
import torch.nn.functional as F
class content_graph_conv(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(content_graph_conv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return torch.mm(adj, x)
class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list,dim_latent: int, act_fn: object = nn.LeakyReLU):
        """Encoder.

        Args:
           dim_in : Number of input channels of the image. For CIFAR, this parameter is 3
           dim_hidden : List of integers representing the number of neurons in each middle layer
           dim_latent : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()
        self.net = self._build_layers(dim_in, dim_hidden, dim_latent)
    def _build_layers(self, dim_in, dim_hidden, dim_out):
        # Base case: If dim_hidden is empty, connect to output layer
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        
        # Recursive case: Build the current layer and the remaining layers
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),  # First hidden layer
            nn.ReLU(),  # Activation function
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out)  # Recursively build the rest
        )
    def forward(self, x):
        return self.net(x)
class Decoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: list, dim_out: int, act_fn: object = nn.ReLU):
        """Decoder.

        Args:
           dim_in : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           dim_hidden : List of integers representing the number of neurons in each middle layer
           dim_out : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()
        self.act_fn = nn.ReLU()
        self.net = self._build_layers(dim_in, dim_hidden, dim_out)
        self.act = act_fn()

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        # Base case: If dim_hidden is empty, connect to output layer
        if len(dim_hidden) == 0:
            return nn.Sequential(nn.Linear(dim_in, dim_out))
        
        # Recursive case: Build the current layer and the remaining layers
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),  # First hidden layer
            nn.ReLU(),  # Activation function
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out)  # Recursively build the rest
        )
    def forward(self, x):
        x = self.net(x)
        return self.act_fn(x)

class GAE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_latent, in_logarithm=True, act_fn=nn.LeakyReLU):
        super(GAE, self).__init__()
        self.gcn_layer = content_graph_conv(dim_in,dim_hidden[0])
        self.encoder = Encoder(dim_hidden[0], dim_hidden[1:], dim_latent, act_fn)
        self.decoder = Decoder(dim_latent, dim_hidden[::-1], dim_in, act_fn) # reverse the dims
        self.in_logarithm = in_logarithm
        self.BN = nn.BatchNorm1d(dim_hidden[0])

    def forward(self, x,adj,idx):
        if self.in_logarithm:
            x = torch.log(1+x) 
        h = F.relu(self.BN(self.gcn_layer(x,adj)))
        h = h[idx,:]
        z = self.encoder(h)
        out = self.decoder(z)
        return out, z   

    def generate(self,x,adj,idx):
        y, z = self.forward(x, adj, idx)
        return y, z
    def forward_loss(self, x, adj, idx):
        y, z = self.forward(x, adj, idx)
        x = x[idx,:]
        loss = F.mse_loss(y, x)
        return loss, y, z


if __name__ == "__main__":
    model = GAE(dim_in=24898, dim_hidden=1024, dim_latent=128).cuda()
    x = torch.ones([100, 24898]).cuda()
    z = model.get_latent_representation(x)
    y = model.generate(x, sample_shape=1)
    print(z.shape, y.shape)

