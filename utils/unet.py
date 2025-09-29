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
  


# #####OPTIMIZING UNET! Model 1

# =====================================================================

# class ResidualConvBlock(nn.Module):
#     """
#     3×3 Residual block with InstanceNorm, LeakyReLU and Dropout3d.
#     Keeps the same in_ch→out_ch signature and uses a 1×1 conv for the skip if needed.
#     """
#     def __init__(self, in_ch, out_ch, dropout=0.2):
#         super().__init__()
#         self.conv1    = nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
#         self.norm1    = nn.InstanceNorm3d(out_ch, affine=True)
#         self.conv2    = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
#         self.norm2    = nn.InstanceNorm3d(out_ch, affine=True)
#         self.act      = nn.LeakyReLU(1e-2, inplace=True)
#         self.dropout  = nn.Dropout3d(dropout)
#         self.res_conv = (nn.Conv3d(in_ch, out_ch, 1, bias=False)
#                          if in_ch != out_ch else nn.Identity())

#     def forward(self, x):
#         res = self.res_conv(x)
#         out = self.act(self.norm1(self.conv1(x)))
#         out = self.dropout(out)
#         out = self.act(self.norm2(self.conv2(out)))
#         out = self.dropout(out)
#         return self.act(out + res)


# class AttentionGate(nn.Module):
#     """
#     3D Attention Gate with on‐the‐fly upsampling of the gating signal.
#     """
#     def __init__(self, g_ch, x_ch, inter_ch):
#         super().__init__()
#         self.Wg  = nn.Sequential(
#             nn.Conv3d(g_ch,    inter_ch, 1, bias=False),
#             nn.InstanceNorm3d(inter_ch)
#         )
#         self.Wx  = nn.Sequential(
#             nn.Conv3d(x_ch,    inter_ch, 1, bias=False),
#             nn.InstanceNorm3d(inter_ch)
#         )
#         self.psi = nn.Sequential(
#             nn.Conv3d(inter_ch, 1,       1, bias=False),
#             nn.InstanceNorm3d(1),
#             nn.Sigmoid()
#         )
#         self.act = nn.LeakyReLU(1e-2, inplace=True)

#     def forward(self, g, x):
#         # project
#         g1 = self.Wg(g)    # may be coarser spatially
#         x1 = self.Wx(x)    # skip is finer spatially

#         # up‐sample g1 to x1's spatial size
#         if g1.shape[2:] != x1.shape[2:]:
#             g1 = F.interpolate(
#                 g1,
#                 size=x1.shape[2:],
#                 mode="trilinear",
#                 align_corners=False
#             )

#         # compute attention
#         psi = self.act(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi



# class Up(nn.Module):
#     """
#     Upsample by transpose‐conv, concatenate skip, then ResidualConvBlock.
#     """
#     def __init__(self, in_ch, skip_ch, out_ch, dropout=0.2):
#         super().__init__()
#         self.up    = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.block = ResidualConvBlock(out_ch + skip_ch, out_ch, dropout)

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         return self.block(x)


# class UNet3D_Optimized_1(nn.Module):
#     """
#       - in_ch, out_ch, dropout, deep_supervision arguments
#       - ResidualConvBlocks
#       - AttentionGates
#       - Deep supervision heads
#     """
#     def __init__(self,in_ch= 4,out_ch= 4,base_ch = 32,dropout= 0.2,deep_supervision = True):
#         super().__init__()
#         self.deep_supervision = deep_supervision

#         # Encoder (3 levels)
#         self.enc1   = ResidualConvBlock(in_ch,    base_ch,     dropout)
#         self.pool1  = nn.MaxPool3d(2)
#         self.enc2   = ResidualConvBlock(base_ch,  base_ch * 2, dropout)
#         self.pool2  = nn.MaxPool3d(2)
#         self.enc3   = ResidualConvBlock(base_ch*2, base_ch * 4, dropout)
#         self.pool3  = nn.MaxPool3d(2)

#         # Bottleneck
#         self.bottleneck = ResidualConvBlock(base_ch*4, base_ch*8, dropout)

#         # Decoder + Attention
#         self.att3 = AttentionGate(g_ch=base_ch*8, x_ch=base_ch*4, inter_ch=base_ch*4)
#         self.up3  = Up(in_ch=base_ch*8, skip_ch=base_ch*4, out_ch=base_ch*4, dropout=dropout)

#         self.att2 = AttentionGate(g_ch=base_ch*4, x_ch=base_ch*2, inter_ch=base_ch*2)
#         self.up2  = Up(in_ch=base_ch*4, skip_ch=base_ch*2, out_ch=base_ch*2, dropout=dropout)

#         self.att1 = AttentionGate(g_ch=base_ch*2, x_ch=base_ch,   inter_ch=base_ch)
#         self.up1  = Up(in_ch=base_ch*2, skip_ch=base_ch,   out_ch=base_ch,   dropout=dropout)

#         # Final classifier
#         self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)

#         # Deep supervision heads
#         if self.deep_supervision:
#             self.ds3 = nn.Conv3d(base_ch*4, out_ch, kernel_size=1)
#             self.ds2 = nn.Conv3d(base_ch*2, out_ch, kernel_size=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2))
#         b  = self.bottleneck(self.pool3(e3))

#         a3 = self.att3(b, e3)
#         d3 = self.up3(b, a3)

#         a2 = self.att2(d3, e2)
#         d2 = self.up2(d3, a2)

#         a1 = self.att1(d2, e1)
#         d1 = self.up1(d2, a1)

#         out = self.out(d1)
#         if not self.deep_supervision:
#             return out

#         # upsample deep‐sup heads to match final resolution
#         ds3 = F.interpolate(
#             self.ds3(d3),
#             scale_factor=4,
#             mode="trilinear",
#             align_corners=False
#         )
#         ds2 = F.interpolate(
#             self.ds2(d2),
#             scale_factor=2,
#             mode="trilinear",
#             align_corners=False
#         )
#         return [out, ds2, ds3]







# #####OPTIMIZING UNET! Model 2

# =====================================================================


# -----------------------------------------------------------------------------
#   Squeeze-and-Excitation block
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Conv3d(channels, mid,    kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(mid,      channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B,C,D,H,W)
        s = x.mean(dim=(2,3,4), keepdim=True)   # global avg pool
        s = self.act(self.fc1(s))
        s = self.sig(self.fc2(s))
        return x * s


# -----------------------------------------------------------------------------
#   Residual block with GroupNorm, Dropout, and SE
# -----------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2, gn_groups=8, se_reduction=16):
        super().__init__()
        self.conv1    = nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1      = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.conv2    = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2      = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.act      = nn.LeakyReLU(1e-2, inplace=True)
        self.drop     = nn.Dropout3d(dropout)
        self.res_conv = (nn.Conv3d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())
        self.se       = SEBlock(out_ch, reduction=se_reduction)

    def forward(self, x):
        res = self.res_conv(x)

        out = self.act(self.gn1(self.conv1(x)))
        out = self.drop(out)

        out = self.act(self.gn2(self.conv2(out)))
        out = self.drop(out)

        out = out + res
        out = self.act(out)

        # channel-wise recalibration
        out = self.se(out)
        return out



#   Attention gate (unchanged, but we ensure GroupNorm too)
class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch, gn_groups=8):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv3d(g_ch,    inter_ch, 1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=inter_ch)
        )
        self.Wx = nn.Sequential(
            nn.Conv3d(x_ch,    inter_ch, 1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="trilinear", align_corners=False
            )
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# -----------------------------------------------------------------------------
#   Upsample + concat + ResidualConvBlock
# -----------------------------------------------------------------------------
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.2, gn_groups=8, se_reduction=16):
        super().__init__()
        # trilinear upsample + 1×1 conv to reduce aliasing
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch),
            nn.LeakyReLU(1e-2, inplace=True),
        )
        self.block = ResidualConvBlock(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
            dropout=dropout,
            gn_groups=gn_groups,
            se_reduction=se_reduction
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# -----------------------------------------------------------------------------
#   Final UNet3D with SE + GroupNorm + Deep Supervision
# -----------------------------------------------------------------------------
class UNet3D_Optimized_2(nn.Module):
    def __init__(
        self,
        in_ch: int = 4,
        out_ch: int = 4,
        base_ch: int = 32,
        dropout: float = 0.2,
        deep_supervision: bool = True
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # encoder (3 levels)
        self.enc1   = ResidualConvBlock(in_ch,    base_ch,     dropout)
        self.pool1  = nn.MaxPool3d(2)
        self.enc2   = ResidualConvBlock(base_ch,  base_ch*2,   dropout)
        self.pool2  = nn.MaxPool3d(2)
        self.enc3   = ResidualConvBlock(base_ch*2, base_ch*4,   dropout)
        self.pool3  = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = ResidualConvBlock(base_ch*4, base_ch*8, dropout)

        # decoder + attention
        self.att3 = AttentionGate(g_ch=base_ch*8, x_ch=base_ch*4, inter_ch=base_ch*4)
        self.up3  = Up(in_ch=base_ch*8, skip_ch=base_ch*4, out_ch=base_ch*4, dropout=dropout)

        self.att2 = AttentionGate(g_ch=base_ch*4, x_ch=base_ch*2, inter_ch=base_ch*2)
        self.up2  = Up(in_ch=base_ch*4, skip_ch=base_ch*2, out_ch=base_ch*2, dropout=dropout)

        self.att1 = AttentionGate(g_ch=base_ch*2, x_ch=base_ch,   inter_ch=base_ch)
        self.up1  = Up(in_ch=base_ch*2, skip_ch=base_ch,   out_ch=base_ch,   dropout=dropout)

        # final classifier
        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)

        # deep‐supervision heads
        if self.deep_supervision:
            self.ds3 = nn.Conv3d(base_ch*4, out_ch, kernel_size=1)
            self.ds2 = nn.Conv3d(base_ch*2, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))

        a3 = self.att3(b, e3)
        d3 = self.up3(b, a3)

        a2 = self.att2(d3, e2)
        d2 = self.up2(d3, a2)

        a1 = self.att1(d2, e1)
        d1 = self.up1(d2, a1)

        out = self.out(d1)
        if not self.deep_supervision:
            return out

        # deep‐supervised logits (upsample to final size)
        ds3 = F.interpolate(self.ds3(d3), scale_factor=4,
                            mode="trilinear", align_corners=False)
        ds2 = F.interpolate(self.ds2(d2), scale_factor=2,
                            mode="trilinear", align_corners=False)
        return [out, ds2, ds3]




























