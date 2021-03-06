import torch.nn as nn


class HCNN(nn.Module):
    """ Class Summary
    This class is used for horizontal-moving only CNN kernels to  
    form time-related data, with 5 hidden layers after input.
    
    Parameters:
    - in_channel: input data channels
    - output_dim: latent size of linear layer
    """

    def __init__(self, in_channel=5, output_dim=256, **args):
        super(HCNN, self).__init__()
        assert "vertical_dim" in args
        self.args = args
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(self.args["vertical_dim"], 1), stride=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=output_dim, kernel_size=(1, 3), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)
