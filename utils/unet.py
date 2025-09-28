import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

def conv_block(in_ch, out_ch, k=3, p=1, act_slope=0.01):
  """2 3D convolutions, activation function is LeakyReLU"""
  return nn.Sequential(
    nn.Conv3d(in_ch, out_ch, k, padding = p, bias = False),
    nn.InstanceNorm3d(out_ch, affine = True),
    nn.LeakyReLU(act_slope, inplace = True),
    nn.Conv3d(out_ch, out_ch, k, padding = p, bias = False),
    nn.InstanceNorm3d(out_ch, affine = True),
    nn.LeakyReLU(act_slope, inplace = True),
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
  


# #####OPTIMIZING UNET!

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, act_slope=0.01, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(act_slope, inplace=True),
            nn.Conv3d(out_ch, out_ch, k, padding=p, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.act  = nn.LeakyReLU(act_slope, inplace=True)

    def forward(self, x):
        out = self.main(x) + self.skip(x)
        out = self.drop(out)
        return self.act(out)



class UNet3D_optimized(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, base=32, dropout=0.1):
        super().__init__()
        # Encoder --  Down path
        self.enc1 = ResidualConvBlock(in_ch, base, dropout=dropout)
        self.down1 = nn.Conv3d(base, base*2, 2, 2)
        self.enc2 = ResidualConvBlock(base*2, base*2, dropout=dropout)
        self.down2 = nn.Conv3d(base*2, base*4, 2, 2)
        self.enc3 = ResidualConvBlock(base*4, base*4, dropout=dropout)
        self.down3 = nn.Conv3d(base*4, base*8, 2, 2)
        self.enc4 = ResidualConvBlock(base*8, base*8, dropout=dropout)

        # Decoder ---  Up path
        self.up3  = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.dec3 = ResidualConvBlock(base*8, base*4, dropout=dropout)
        self.up2  = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.dec2 = ResidualConvBlock(base*4, base*2, dropout=dropout)
        self.up1  = nn.ConvTranspose3d(base*2, base,   2, 2)
        self.dec1 = ResidualConvBlock(base*2, base, dropout=dropout)

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
  


# # Residual Convolutional Block with GroupNorm
# # ----------------------------------------
# class ResidualConvBlockGN(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, p=1, act_slope=0.01, dropout=0.1, gn_groups=8):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
#             nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch, affine=True),
#             nn.LeakyReLU(act_slope, inplace=True),
#             nn.Conv3d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
#             nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch, affine=True)
#         )
#         self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)
#         self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
#         self.act  = nn.LeakyReLU(act_slope, inplace=True) #Final Activation

#     def forward(self, x):
#         out = self.main(x) + self.skip(x)
#         out = self.drop(out)
#         return self.act(out)

# # 3D Attention Gate
# # ----------------------------------------
# class AttentionGate3D(nn.Module):
#     def __init__(self, F_g, F_l, F_int):

#         '''
#  F_g = channels in gating (decoder) signal
# F_l = channels in encoder feature
# F_int = intermediate channels for computing attention

        
#         '''
#         super().__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv3d(F_g, F_int, kernel_size=1, bias=True),
#             nn.GroupNorm(num_groups=1, num_channels=F_int)
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv3d(F_l, F_int, kernel_size=1, bias=True),
#             nn.GroupNorm(num_groups=1, num_channels=F_int)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, g):
#         # x = encoder feature map, g = gating signal from decoder
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# # ----------------------------------------
# # 3D UNet with base=64, 5 levels, GroupNorm, Deep Supervision & Attention Gates
# # ----------------------------------------
# class UNet3D_optimized2(nn.Module):
#     def __init__(self, in_ch=4, out_ch=4, base=64, dropout=0, gn_groups=8):
#         super().__init__()
#         # Encoder
#         self.enc1  = ResidualConvBlockGN(in_ch,     base,   dropout=dropout, gn_groups=gn_groups)
#         self.down1 = nn.Conv3d(base,    base*2, 2, 2)
#         self.enc2  = ResidualConvBlockGN(base*2,  base*2, dropout=dropout, gn_groups=gn_groups)
#         self.down2 = nn.Conv3d(base*2,  base*4, 2, 2)
#         self.enc3  = ResidualConvBlockGN(base*4,  base*4, dropout=dropout, gn_groups=gn_groups)
#         self.down3 = nn.Conv3d(base*4,  base*8, 2, 2)
#         self.enc4  = ResidualConvBlockGN(base*8,  base*8, dropout=dropout, gn_groups=gn_groups)
#         self.down4 = nn.Conv3d(base*8,  base*16,2, 2)
#         self.enc5  = ResidualConvBlockGN(base*16, base*16,dropout=dropout, gn_groups=gn_groups)

#         # Attention gates
#         self.att4 = AttentionGate3D(F_g=base*8, F_l=base*8, F_int=base*4)
#         self.att3 = AttentionGate3D(F_g=base*4, F_l=base*4, F_int=base*2)
#         self.att2 = AttentionGate3D(F_g=base*2, F_l=base*2, F_int=base)
#         self.att1 = AttentionGate3D(F_g=base,   F_l=base,   F_int=base//2)

#         # Decoder + deep supervision heads
#         self.up4  = nn.ConvTranspose3d(base*16, base*8, 2, 2)
#         self.dec4 = ResidualConvBlockGN(base*16, base*8,  dropout=dropout, gn_groups=gn_groups)
#         self.ds4  = nn.Conv3d(base*8,  out_ch, 1)

#         self.up3  = nn.ConvTranspose3d(base*8,  base*4, 2, 2)
#         self.dec3 = ResidualConvBlockGN(base*8,  base*4,  dropout=dropout, gn_groups=gn_groups)
#         self.ds3  = nn.Conv3d(base*4,  out_ch, 1)

#         self.up2  = nn.ConvTranspose3d(base*4,  base*2, 2, 2)
#         self.dec2 = ResidualConvBlockGN(base*4,  base*2,  dropout=dropout, gn_groups=gn_groups)
#         self.ds2  = nn.Conv3d(base*2,  out_ch, 1)

#         self.up1  = nn.ConvTranspose3d(base*2,  base,   2, 2)
#         self.dec1 = ResidualConvBlockGN(base*2,  base,   dropout=dropout, gn_groups=gn_groups)
#         self.out  = nn.Conv3d(base,    out_ch, 1)

#     def forward(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.down1(e1))
#         e3 = self.enc3(self.down2(e2))
#         e4 = self.enc4(self.down3(e3))
#         e5 = self.enc5(self.down4(e4))

#         # Decoder with Attention
#         d4   = self.up4(e5)
#         e4_a = self.att4(e4, d4)
#         d4   = self.dec4(torch.cat([d4, e4_a], dim=1))

#         d3   = self.up3(d4)
#         e3_a = self.att3(e3, d3)
#         d3   = self.dec3(torch.cat([d3, e3_a], dim=1))

#         d2   = self.up2(d3)
#         e2_a = self.att2(e2, d2)
#         d2   = self.dec2(torch.cat([d2, e2_a], dim=1))

#         d1   = self.up1(d2)
#         e1_a = self.att1(e1, d1)
#         d1   = self.dec1(torch.cat([d1, e1_a], dim=1))

#         # Deep supervision outputs
#         out_main = self.out(d1)
#         out_ds2  = F.interpolate(self.ds2(d2), size=out_main.shape[2:], mode="trilinear", align_corners=False)
#         out_ds3  = F.interpolate(self.ds3(d3), size=out_main.shape[2:], mode="trilinear", align_corners=False)
#         out_ds4  = F.interpolate(self.ds4(d4), size=out_main.shape[2:], mode="trilinear", align_corners=False)

#         return [out_main, out_ds2, out_ds3, out_ds4]



