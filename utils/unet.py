import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

def conv_block(in_ch, out_ch, k=3, p=1):
  """2 3D convolutions, activation function is LeakyReLU"""
  return nn.Sequential(
    nn.Conv3d(in_ch, out_ch, k, padding = p, bias = False),
    nn.InstanceNorm3d(out_ch, affine = True),
    nn.LeakyReLU(0.01, inplace = True),
    nn.Conv3d(out_ch, out_ch, k, padding = p, bias = False),
    nn.InstanceNorm3d(out_ch, affine = True),
    nn.LeakyReLU(0.01, inplace = True),
  )
    
class UNet3D(nn.Module):
  def __init__(self, in_ch=4, out_ch=4, base =32):
    super().__init__()
    ## Down path
    self.enc1 = conv_block(in_ch, base)
    self.down1 = nn.Conv3d(base, base*2, 2, 2) # stride=2
    self.enc2 = conv_block(base*2, base*2)
    self.down2 = nn.Conv3d(base*2, base*4, 2, 2)
    self.enc3 = conv_block(base*4, base*4)
    self.down3 = nn.Conv3d(base*4, base*8, 2, 2)
    self.enc4 = conv_block(base*8, base*8)

    ## Up path
    self.up3  = nn.ConvTranspose3d(base*8, base*4, 2, 2)
    self.dec3 = conv_block(base*8,  base*4)
    self.up2  = nn.ConvTranspose3d(base*4, base*2, 2, 2)
    self.dec2 = conv_block(base*4,  base*2)
    self.up1  = nn.ConvTranspose3d(base*2, base,   2, 2)
    self.dec1 = conv_block(base*2,  base)

    self.out = nn.Conv3d(base, out_ch, 1)

  def forward(self,x):
    e1 = self.enc1(x)
    e2 = self.enc2(self.down1(e1))
    e3 = self.enc3(self.down2(e2))
    e4 = self.enc4(self.down3(e3))

    d3 = self.up3(e4)
    d3 = torch.cat([d3, e3], dim=1)
    d3 = self.dec3(d3)

    d2 = self.up2(d3)
    d2 = torch.cat([d2, e2], dim=1)
    d2 = self.dec2(d2)

    d1 = self.up1(d2)
    d1 = torch.cat([d1, e1], dim=1)
    d1 = self.dec1(d1)

    return self.out(d1)
  


