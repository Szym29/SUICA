# import tinycudann as tcnn
import torch.nn as nn

class NGP(nn.Module):
    def __init__(self, dim_in=2, dim_hidden=64, dim_out=32, num_layers=2, final_activation="Identity"):
        super().__init__()
        
        if final_activation == "Identity":
            final_activation = "None" 
        # Define the neural network with the input encoding
        # self.network = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=dim_in,
        #     n_output_dims=dim_out,
        #     encoding_config={
        #         "otype": "Frequency",
        #         "n_frequencies": 8
        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": final_activation,
        #         "n_neurons": dim_hidden,
        #         "n_hidden_layers": num_layers
        #     }
        # )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    import torch
    ngp = NGP().cuda()
    x = torch.ones([10000, 2]).cuda()
    y = ngp(x)
    print(y.shape)
