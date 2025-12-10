import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorConv3d(nn.Module):
    """
    (2+1)D decomposition 3D convolution: 1xHxW spatial convolution -> Swish -> Tx1x1 temporal convolution
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride: int = 1,
                 dilation: int = 1):
        super().__init__()

        if isinstance(kernel_size, int):
            k_t, k_h, k_w = kernel_size, kernel_size, kernel_size
        else:
            k_t, k_h, k_w = kernel_size

        pad_t  = (k_t - 1) * dilation // 2
        pad_hw = (k_h - 1) * dilation // 2

        self.spatial = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(1, k_h, k_w),
            stride=(1, stride, stride),
            padding=(0, pad_hw, pad_hw),
            dilation=(1, dilation, dilation),
            groups=in_channels,
            bias=False
        )

        self.temporal = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(k_t, 1, 1),
            stride=(stride, 1, 1),
            padding=(pad_t, 0, 0),
            dilation=(dilation, 1, 1),
            bias=True
        )

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.spatial(x)
        x = self.act(x)
        x = self.temporal(x)
        return x


class LayerNorm2D(nn.Module):
    """
    LayerNorm over C for a 4-D tensor (B, C, H, W)
    """
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)        # (B, 1, H, W)
        var  = x.var (dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class PoseRefNetNoBNV3(nn.Module):
    def __init__(self,
                 in_channels_c: int,
                 in_channels_x: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.nhead = num_heads

        self.proj_p = nn.Conv2d(in_channels_c, hidden_dim, kernel_size=1)
        self.proj_r = nn.Conv2d(in_channels_x, hidden_dim, kernel_size=1)

        self.proj_p_back = nn.Conv2d(hidden_dim, in_channels_c, kernel_size=1)

        self.cross_attn = nn.MultiheadAttention(hidden_dim,
                                                num_heads=num_heads,
                                                dropout=dropout)

        self.ffn_pose = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.norm1 = LayerNorm2D(hidden_dim)
        self.norm2 = LayerNorm2D(hidden_dim)

    def forward(self, pose, ref, mask=None):
        """
        pose : (B, C1, T, H, W)
        ref  : (B, C2, T, H, W)
        mask : (B, T*H*W) optional key_padding_mask
        return: (B, d_model, T, H, W)
        """
        B, _, T, H, W = pose.shape
        L = H * W

        p_trans = pose.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)
        r_trans = ref.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)

        p_trans = self.proj_p(p_trans)
        r_trans = self.proj_r(r_trans)

        p_trans = p_trans.flatten(2).transpose(1, 2)
        r_trans = r_trans.flatten(2).transpose(1, 2)

        # MultiheadAttention expects (L, N, E) if batch_first=False (default)
        p_trans = p_trans.transpose(0, 1)
        r_trans = r_trans.transpose(0, 1)

        out = self.cross_attn(query=r_trans,
                              key=p_trans,
                              value=p_trans,
                              key_padding_mask=mask)[0]

        out = out.transpose(0, 1) # (N, L, E)
        out = out.transpose(1, 2).contiguous().view(B*T, -1, H, W)
        out = self.norm1(out)

        ffn_out = self.ffn_pose(out)
        out = out + ffn_out
        out = self.norm2(out)
        out = self.proj_p_back(out)
        out = out.view(B, T, -1, H, W).contiguous().transpose(1, 2)

        return out


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class DYModule(nn.Module):
    def __init__(self, inp, oup, fc_squeeze=8):
        super(DYModule, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        if inp < oup:
            self.mul = 4
            reduction = 8
            self.avg_pool = nn.AdaptiveAvgPool2d(2)
        else:
            self.mul = 1
            reduction = 2
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dim = min((inp * self.mul) // reduction, oup // reduction)
        while self.dim ** 2 > inp * self.mul * 2:
            reduction *= 2
            self.dim = min((inp * self.mul) // reduction, oup // reduction)
        if self.dim < 4:
            self.dim = 4

        squeeze = max(inp * self.mul, self.dim ** 2) // fc_squeeze
        if squeeze < 4:
            squeeze = 4
        self.conv_q = nn.Conv2d(inp, self.dim, 1, 1, 0, bias=False)

        self.fc = nn.Sequential(
            nn.Linear(inp * self.mul, squeeze, bias=False),
            SEModule_small(squeeze),
        )
        self.fc_phi = nn.Linear(squeeze, self.dim ** 2, bias=False)
        self.fc_scale = nn.Linear(squeeze, oup, bias=False)
        self.hs = Hsigmoid()
        self.conv_p = nn.Conv2d(self.dim, oup, 1, 1, 0, bias=False)
        
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=self.dim)
        self.bn2 = nn.GroupNorm(num_groups=4, num_channels=self.dim)

    def forward(self, x):
        r = self.conv(x)

        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c * self.mul)
        y = self.fc(y)
        dy_phi = self.fc_phi(y).view(b, self.dim, self.dim)
        dy_scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1)
        r = dy_scale.expand_as(r) * r

        x = self.conv_q(x)
        x = self.bn1(x)
        x = x.view(b, -1, h * w)
        x = self.bn2(torch.matmul(dy_phi, x)) + x
        x = x.view(b, -1, h, w)
        x = self.conv_p(x)
        return x + r
