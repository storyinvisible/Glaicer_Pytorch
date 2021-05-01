import torch.nn as nn


class VCNN(nn.Module):
    """ Class Summary
    This class is used for vertical-moving only CNN kernels to  
    form time-related data.
    
    Parameters:
    - in_channel: input data channels
    - output_dim: latent size of linear layer
    """

    def __init__(self, in_channel=5, output_dim=256, **args):
        super(VCNN, self).__init__()
        self.args = args
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 12), stride=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32 * 35, output_dim)
        )

    def forward(self, x):
        out =self.model(x)
        print(out.shape)
        return out
