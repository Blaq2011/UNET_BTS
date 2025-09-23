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
  

#####OPTIMIZING UNET!
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
        # Encoder
        self.enc1 = ResidualConvBlock(in_ch, base, dropout=dropout)
        self.down1 = nn.Conv3d(base, base*2, 2, 2)
        self.enc2 = ResidualConvBlock(base*2, base*2, dropout=dropout)
        self.down2 = nn.Conv3d(base*2, base*4, 2, 2)
        self.enc3 = ResidualConvBlock(base*4, base*4, dropout=dropout)
        self.down3 = nn.Conv3d(base*4, base*8, 2, 2)
        self.enc4 = ResidualConvBlock(base*8, base*8, dropout=dropout)

        # Decoder
        self.up3  = nn.ConvTranspose3d(base*8, base*4, 2, 2)
        self.dec3 = ResidualConvBlock(base*8, base*4, dropout=dropout)
        self.up2  = nn.ConvTranspose3d(base*4, base*2, 2, 2)
        self.dec2 = ResidualConvBlock(base*4, base*2, dropout=dropout)
        self.up1  = nn.ConvTranspose3d(base*2, base,   2, 2)
        self.dec1 = ResidualConvBlock(base*2, base, dropout=dropout)

        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)





















































# Building blocks
# ---------------------------

# class ResidualConvBlock(nn.Module):
#     """
#     Residual block:
#       Conv3d -> IN -> LReLU -> Conv3d -> IN + skip -> LReLU
#     """
#     def __init__(self, in_ch, out_ch, k=3, p=1, act_slope=0.01):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
#             nn.InstanceNorm3d(out_ch, affine=True),
#             nn.LeakyReLU(act_slope, inplace=True),
#             nn.Conv3d(out_ch, out_ch, k, padding=p, bias=False),
#             nn.InstanceNorm3d(out_ch, affine=True),
#         )
#         self.act = nn.LeakyReLU(act_slope, inplace=True)
#         self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, 1, bias=False)

#     def forward(self, x):
#         return self.act(self.conv(x) + self.skip(x))


# class Downsample(nn.Module):
#     """
#     Learned downsampling via stride-2 Conv3d.
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.down = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

#     def forward(self, x):
#         return self.down(x)


# class UpsampleBlock(nn.Module):
#     """
#     Trilinear upsample + Conv3d (safer than ConvTranspose for artifacts).
#     Optionally applies an AttentionGate on the skip connection.
#     """
#     def __init__(self, in_ch, out_ch, use_attention=False):
#         super().__init__()
#         self.use_attention = use_attention
#         self.conv1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)  # reduce channels after upsample
#         if use_attention:
#             self.att = AttentionGate(g_ch=out_ch, x_ch=out_ch)  # gating on aligned channels
#         else:
#             self.att = None

#     def forward(self, x, skip):
#         # x: decoder feature (low-res), skip: encoder feature (high-res)
#         x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
#         x = self.conv1x1(x)  # channel align

#         if self.att is not None:
#             skip = self.att(g=x, x=skip)

#         return torch.cat([x, skip], dim=1)


# class AttentionGate(nn.Module):
#     """
#     Simple additive Attention Gate (Attention U-Net style).
#     g: decoder gating signal, x: encoder skip features.
#     Outputs attention-weighted skip.
#     """
#     def __init__(self, g_ch, x_ch, inter_ch=None):
#         super().__init__()
#         if inter_ch is None:
#             inter_ch = max(1, x_ch // 2)

#         self.W_g = nn.Sequential(
#             nn.Conv3d(g_ch, inter_ch, kernel_size=1, bias=True),
#             nn.InstanceNorm3d(inter_ch, affine=True),
#         )
#         self.W_x = nn.Sequential(
#             nn.Conv3d(x_ch, inter_ch, kernel_size=1, bias=True),
#             nn.InstanceNorm3d(inter_ch, affine=True),
#         )
#         self.psi = nn.Sequential(
#             nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, g, x):
#         att = F.leaky_relu(self.W_g(g) + self.W_x(x), negative_slope=0.01, inplace=True)
#         alpha = self.psi(att)  # [B,1,D,H,W]
#         return x * alpha


# # ---------------------------
# # UNet3D (Residual, optional Attention & Deep Supervision)
# # ---------------------------

# class UNet3D_optimized(nn.Module):
#     """
#     Residual 3D U-Net with:
#       - InstanceNorm3d + LeakyReLU
#       - Residual blocks
#       - Trilinear upsample + 1x1 conv for channel alignment
#       - Optional Attention Gates on skip connections
#       - Optional Dropout in the bottleneck
#       - Optional Deep Supervision (aux heads at decoder stages)

#     Default behavior matches your previous interface:
#       UNet3D(in_ch=4, out_ch=4) and forward(x) -> logits of shape (B, C, D, H, W)

#     If deep_supervision=True, you can request auxiliary logits:
#       forward(x, return_aux=True) -> (main_logits, [aux2, aux3])
#     """
#     def __init__(
#         self,
#         in_ch: int = 4,
#         out_ch: int = 4,
#         base: int = 32,
#         depth: int = 4,
#         use_attention: bool = False,
#         dropout: float = 0.0,
#         deep_supervision: bool = False,
#         act_slope: float = 0.01,
#     ):
#         """
#         Args:
#             in_ch: input channels (BraTS modalities = 4)
#             out_ch: number of classes (BraTS = 4)
#             base: base number of feature maps (32 default)
#             depth: number of down/upsampling levels (4 is typical)
#             use_attention: if True, apply attention gate on skip connections
#             dropout: dropout prob in the bottleneck (0.0 to disable)
#             deep_supervision: if True, create aux heads in decoder
#             act_slope: negative slope for LeakyReLU
#         """
#         super().__init__()
#         assert depth == 4, "This implementation is configured for depth=4 (common for 128^3 / 96^3 patches)."

#         self.deep_supervision = deep_supervision
#         widths = [base, base*2, base*4, base*8]  # encoder widths; bottleneck = base*8

#         # Encoder
#         self.enc1 = ResidualConvBlock(in_ch, widths[0], act_slope=act_slope)
#         self.down1 = Downsample(widths[0], widths[1])
#         self.enc2 = ResidualConvBlock(widths[1], widths[1], act_slope=act_slope)
#         self.down2 = Downsample(widths[1], widths[2])
#         self.enc3 = ResidualConvBlock(widths[2], widths[2], act_slope=act_slope)
#         self.down3 = Downsample(widths[2], widths[3])
#         self.enc4 = ResidualConvBlock(widths[3], widths[3], act_slope=act_slope)

#         self.drop = nn.Dropout3d(dropout) if dropout and dropout > 0.0 else nn.Identity()

#         # Decoder
#         # up from enc4 -> enc3
#         self.up3  = UpsampleBlock(in_ch=widths[3], out_ch=widths[2], use_attention=use_attention)
#         self.dec3 = ResidualConvBlock(widths[2] + widths[2], widths[2], act_slope=act_slope)

#         # up from dec3 -> enc2
#         self.up2  = UpsampleBlock(in_ch=widths[2], out_ch=widths[1], use_attention=use_attention)
#         self.dec2 = ResidualConvBlock(widths[1] + widths[1], widths[1], act_slope=act_slope)

#         # up from dec2 -> enc1
#         self.up1  = UpsampleBlock(in_ch=widths[1], out_ch=widths[0], use_attention=use_attention)
#         self.dec1 = ResidualConvBlock(widths[0] + widths[0], widths[0], act_slope=act_slope)

#         # Heads
#         self.out_head = nn.Conv3d(widths[0], out_ch, kernel_size=1)

#         if self.deep_supervision:
#             # Auxiliary heads at dec2 and dec3 (weights e.g. 0.5, 0.25 in your loss)
#             self.aux2 = nn.Conv3d(widths[1], out_ch, kernel_size=1)
#             self.aux3 = nn.Conv3d(widths[2], out_ch, kernel_size=1)
#         else:
#             self.aux2 = None
#             self.aux3 = None

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#                 nn.init.kaiming_normal_(m.weight, a=0.01)
#                 if getattr(m, "bias", None) is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x, return_aux: bool = False):
#         # Encoder
#         e1 = self.enc1(x)            # base
#         e2 = self.enc2(self.down1(e1))  # base*2
#         e3 = self.enc3(self.down2(e2))  # base*4
#         e4 = self.enc4(self.down3(e3))  # base*8

#         # Bottleneck regularization if enabled
#         e4 = self.drop(e4)

#         # Decoder
#         d3 = self.up3(e4, e3)        # concat([up(e4)->base*4, e3->base*4]) => 2*base*4
#         d3 = self.dec3(d3)           # base*4

#         d2 = self.up2(d3, e2)        # concat([up(d3)->base*2, e2->base*2]) => 2*base*2
#         d2 = self.dec2(d2)           # base*2

#         d1 = self.up1(d2, e1)        # concat([up(d2)->base, e1->base]) => 2*base
#         d1 = self.dec1(d1)           # base

#         logits = self.out_head(d1)   # (B, out_ch, D, H, W)

#         if self.deep_supervision and return_aux:
#             # produce aux logits upsampled to the final resolution
#             aux2 = self.aux2(d2)
#             aux3 = self.aux3(d3)
#             aux2 = F.interpolate(aux2, size=logits.shape[2:], mode='trilinear', align_corners=False)
#             aux3 = F.interpolate(aux3, size=logits.shape[2:], mode='trilinear', align_corners=False)
#             return logits, [aux2, aux3]

#         return logits

























