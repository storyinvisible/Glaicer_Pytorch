import torch.nn as nn


class VInvertedResidual(nn.Module):
    """ Class Summary
    This class works as an extractor of features in our dataset,
    with one input CNN layer inside, including extra two invert-
    ed residual blocks from MobileNetV2 as feature extrator. On-
    ly x parameters are taken in. Input kernel size are set to 
    (289, 1) without padding and stride (1, 1) to form first out-
    put. 
    
    Parameters:
    - param: To be filled if any
    """

    def __init__(self, in_channel=5, output_dim=256, **args):
        super(VInvertedResidual, self).__init__()
        self.args = args
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=(289, 1), stride=(1, 1), padding=(0, 0)),
            nn.LeakyReLU(0.2),
            VInvertedBlock(in_channel=16, out_channel=64, stride=(2, 1)),
            VInvertedBlock(in_channel=64, out_channel=32, stride=(1, 1)),
            nn.Flatten(),
            nn.Linear(32 * 5, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class VInvertedBlock(nn.Module):
    """ Class Summary
    This class works as a specific block which helps to constru-
    ct the inverted residual extractor. If there's downsampling,
    no shortcut will be used. However, if there's no downsampl-
    ing, a shortcut will be applied.
    
    Parameters:
    - expansion: expansion rate used among dimension expansion
    and depthwise layer.
    - out_channel: Used to decide output channel after linear b-
    ottleneck.
    """

    def __init__(self, in_channel, out_channel, expansion=4, stride=(1, 1)):
        super(VInvertedBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel * expansion, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=in_channel * expansion, out_channels=in_channel * expansion, kernel_size=(1, 3),
                      stride=stride, padding=(0, 1), groups=in_channel * expansion),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=in_channel * expansion, out_channels=out_channel, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0)),
        )

        if in_channel != out_channel and stride == (1, 1):
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1),
                                      stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        out = self.block(x)
        if self.in_channel != self.out_channel and self.stride == (1, 1):
            return out + self.shortcut(x)
        else:
            return out
