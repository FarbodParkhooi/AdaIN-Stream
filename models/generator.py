from utilities.adain import AdaIN
from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Encoder
        self.enc_cnv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_nrm1 = nn.InstanceNorm2d(num_features=64) 

        self.enc_cnv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False) # image_size/2
        self.enc_nrm2 = nn.InstanceNorm2d(num_features=128)

        self.enc_cnv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False) # image_size/4
        self.enc_nrm3 = nn.InstanceNorm2d(num_features=256)

        self.enc_cnv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False) # image_size/8
        self.enc_nrm4 = nn.InstanceNorm2d(num_features=512)

        self.enc_cnv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False) # image_size/8
        self.enc_nrm5 = nn.InstanceNorm2d(num_features=512)

        self.enc_cnv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False) # image_size/8
        self.enc_nrm6 = nn.InstanceNorm2d(num_features=512)

        # Decoder
        self.dec_ups1 = nn.Upsample(scale_factor=2, mode='bilinear') # image_size/4
        self.dec_cnv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        # Fusion 1
        self.dec_fcv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec_fnm1 = nn.InstanceNorm2d(num_features=256, affine=True)
        # Skip processing 1
        self.skip_cv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False)

        self.dec_ups2 = nn.Upsample(scale_factor=2, mode='bilinear') # image_size/2
        self.dec_cnv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        # Fusion 2
        self.dec_fcv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec_fnm2 = nn.InstanceNorm2d(num_features=128, affine=True)
        # Skip processing 1
        self.skip_cv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, bias=False)

        self.dec_ups3 = nn.Upsample(scale_factor=2, mode='bilinear') # image_size/1
        self.dec_cnv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # Fusion 3
        self.dec_fcv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec_fnm3 = nn.InstanceNorm2d(num_features=64, affine=True)
        # Skip processing 1
        self.skip_cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False)

        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, image, style):
        skips = []

        # Encoder
        x = self.enc_cnv1(image)
        x = self.enc_nrm1(x)
        x = self.relu(x)
        skips.append(x)

        x = self.enc_cnv2(x)
        x = self.enc_nrm2(x)
        x = self.relu(x)
        skips.append(x)

        x = self.enc_cnv3(x)
        x = self.enc_nrm3(x)
        x = self.relu(x)
        skips.append(x)

        x = self.enc_cnv4(x)
        x = self.enc_nrm4(x)
        x = self.relu(x)
        identity = x

        x = self.enc_cnv5(identity)
        x = self.enc_nrm5(x)
        x = self.relu(x)

        x = self.enc_cnv6(x)
        x = self.enc_nrm6(x)
        x = x + identity
        x = self.relu(x)

        # Decoder
        x = self.dec_ups1(x)
        x = self.dec_cnv1(x)
        x = AdaIN(x, style[0])
        # Skip processing
        skip = self.skip_cv1(skips.pop())
        x = torch.cat([x, skip], dim=1)
        # Fusion 1
        x = self.dec_fcv1(x)
        x = self.dec_fnm1(x)
        x = self.relu(x)

        x = self.dec_ups2(x)
        x = self.dec_cnv2(x)
        x = AdaIN(x, style[1])
        # Skip processing
        skip = self.skip_cv2(skips.pop())
        x = torch.cat([x, skip], dim=1)
        # Fusion 2
        x = self.dec_fcv2(x)
        x = self.dec_fnm2(x)
        x = self.relu(x)

        x = self.dec_ups3(x)
        x = self.dec_cnv3(x)
        x = AdaIN(x, style[2])
        # Skip processing
        skip = self.skip_cv3(skips.pop())
        x = torch.cat([x, skip], dim=1)
        # Fusion 3
        x = self.dec_fcv3(x)
        x = self.dec_fnm3(x)
        x = self.relu(x)

        x = self.output(x)
