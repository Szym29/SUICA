import torch
import torch.nn as nn
import numpy as np

class GaussianEncoding(nn.Module):
    """
    Given an input of size [batches, num_input_channels],
     returns a tensor of size [batches, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.register_buffer('B', self._B)

    def forward(self, x):
        x = x @ self.B.to(x.device)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class FourierFeatureNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, final_activation):
        super().__init__()
        self.transform = GaussianEncoding(dim_in, 256)

        final_activation = getattr(nn, final_activation)()

        self.layers = nn.Sequential(
                        nn.Linear(512, dim_hidden),
                        nn.ReLU(),
                        *[nn.Linear(dim_hidden, dim_hidden) for _ in range(num_layers)],
                        nn.Linear(dim_hidden, dim_out),
                        final_activation
                    )
        
    
    def forward(self, x):
        ff = self.transform(x)
        y = self.layers(ff)
        return y


if __name__ == "__main__":
    transform = FourierFeatureNet(dim_in=2, dim_hidden=8, dim_out=64, num_layers=2, final_activation="Identity")
    x = torch.ones([100, 2])
    y = transform(x)
    print(y.shape)
