import torch.nn as nn

class TWCNN(nn.Module):
    """ Class Summary
    This class is used for 4-d data convolution. We do 3d convo-
    lution considering time as channels. When taking inputs, we 
    assume that the inputs are ordered in (Type, Month, axis_x, 
    axis_y), where type is ordered in (temperature, wind, press-
    ure, precipitation, cloud_cover, humidity, ocean). The kern-
    el design is (1, 3, 7) in first layer by default. Different 
    from TCNN, we deal with data channel-wise, then concatnate 
    at the last layer.
    The output size after each layer is:
    1  (14, 12, 13, 20)
    2  (28, 6, 11, 18)
    3  (56, 3, 9, 16)
    4  (112, 1, 3, 4)
    5  (output_dim, 1, 1, 1)
    
    Parameters:
    - in_channel: number of input type. 7 by default.
    - output_dim: output_dim for latent features. 224 by fefault
    """
    
    def __init__(self, in_channel = 7, output_dim = 224, **args):
        super(TWCNN, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Conv3d(in_channels = in_channel, out_channels = 14, kernel_size = (1, 3, 7), stride = (1, 1, 3), padding = 0, groups = 7),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels = 14, out_channels = 28, kernel_size = (2, 3, 3), stride = (2, 1, 1), padding = 0, groups = 7),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels = 28, out_channels = 56, kernel_size = (2, 3, 3), stride = (2, 1, 1), padding = 0, groups = 7),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels = 56, out_channels = 112, kernel_size = (3, 3, 4), stride = (1, 3, 4), padding = 0, groups = 7),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels = 112, out_channels = output_dim, kernel_size = (1, 3, 4), stride = 1, padding = 0, groups = 7),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        
    def forward(self, x):
        return self.model(x)