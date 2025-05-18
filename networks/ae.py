import torch.nn as nn
import torch
import torch.nn.functional as F

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
    def __init__(self, dim_in: int, dim_hidden: list, dim_out: int, act_fn: object = nn.LeakyReLU):
        """Decoder.

        Args:
           dim_in : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           dim_hidden : List of integers representing the number of neurons in each middle layer
           dim_out : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()
    
        self.net = self._build_layers(dim_in, dim_hidden, dim_out)

    def _build_layers(self, dim_in, dim_hidden, dim_out):
        # Base case: If dim_hidden is empty, connect to output layer
        if len(dim_hidden) == 0:
            return nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.ReLU()
                )
        
        # Recursive case: Build the current layer and the remaining layers
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden[0]),  # First hidden layer
            nn.ReLU(),  # Activation function
            self._build_layers(dim_hidden[0], dim_hidden[1:], dim_out)  # Recursively build the rest
        )
    def forward(self, x):
        return self.net(x)
    
class AE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_latent, in_logarithm=True, act_fn=nn.LeakyReLU):
        super(AE, self).__init__()
        self.encoder = Encoder(dim_in, dim_hidden, dim_latent, act_fn)
        self.decoder = Decoder(dim_latent, dim_hidden[::-1], dim_in, act_fn) # reverse the dims
        self.in_logarithm = in_logarithm
    
    def forward(self, x):
        if self.in_logarithm:
            x = torch.log(1+x) # suggested by yumin
        z = self.encoder(x)
        return self.decoder(z), z
    
    def forward_loss(self, x):
        y, z = self.forward(x)
        loss = F.mse_loss(y, x)
        return loss, y, z

if __name__ == "__main__":

    model = AE(dim_in=24898, dim_hidden=1024, dim_latent=128).cuda()
    x = torch.ones([100, 24898]).cuda()
    z = model.get_latent_representation(x)
    y = model.generate(x, sample_shape=1)
    print(z.shape, y.shape)

