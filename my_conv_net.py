import torch
import torch.nn as nn


class MyConvNet(nn.Module):

    def __init__(self):
        super(MyConvNet, self).__init__()

        self.conv = nn.Conv2d(1, 8, kernel_size=5, bias=False)
        self.relu = nn.ReLU()
        self.avg = nn.AvgPool2d(kernel_size=4)

        self.fc1 = nn.Linear(288, 10)

        self.current_fms = None

    def forward(self, x):
        out = self.conv(x)
        self.current_fms = out

        out = self.relu(out)
        out= self.avg(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)

        return out