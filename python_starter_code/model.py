import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """

        # change this obviously!
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=0, bias=False),
            nn.LayerNorm(normalized_shape=[106, 106]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2)
        )

        self.depthwise_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=2, padding=0, groups=8, bias=False),
            nn.LayerNorm(normalized_shape=[24, 24]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.depthwise_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, groups=8, bias=False),
            nn.LayerNorm(normalized_shape=[6, 6]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, stride=1, padding=0, bias=True)
        )
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
        x = self.conv(x)
        x = self.depthwise_conv1(x)
        x = self.depthwise_conv2(x)
        out = self.fc(x)

        # change this obviously!
        
        return out