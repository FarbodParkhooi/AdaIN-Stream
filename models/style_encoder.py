from torch import nn
import torch

class style_encoder(nn.Module):
    def __init__(self):
        super(style_encoder, self).__init__()

        self.cnv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False) # image_size/2
        self.nrm1 = nn.InstanceNorm2d(num_features=64)

        self.cnv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False) # image_size/4
        self.nrm2 = nn.InstanceNorm2d(num_features=128)

        self.cnv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False) # image_size/8
        self.nrm3 = nn.InstanceNorm2d(num_features=256)

        self.cnv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False) # image_size/16
        self.nrm4 = nn.InstanceNorm2d(num_features=512)

        self.head_1 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.head_2 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.head_3 = nn.Linear(in_features=512, out_features=128, bias=True)

        self.pol1 = nn.AdaptiveAvgPool2d((1,1))
        self.flt1 = nn.Flatten(start_dim=1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, image):
        x = self.cnv1(image)
        x = self.nrm1(x)
        x = self.relu(x)

        x = self.cnv2(x)
        x = self.nrm2(x)
        x = self.relu(x)

        x = self.cnv3(x)
        x = self.nrm3(x)
        x = self.relu(x)

        x = self.cnv4(x)
        x = self.nrm4(x)
        x = self.relu(x)
        
        x = self.pol1(x)
        x = self.flt1(x)

        h1 = self.head_1(x); gamma1, beta1 = torch.chunk(h1, 2, dim=1)
        h2 = self.head_2(x); gamma2, beta2 = torch.chunk(h2, 2, dim=1)
        h3 = self.head_3(x); gamma3, beta3 = torch.chunk(h3, 2, dim=1)

        return ((gamma1, beta1), (gamma2, beta2), (gamma3, beta3))
        