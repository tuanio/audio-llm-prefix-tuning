import torch.nn as nn


class LoRDepthWiseSepConv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, rank=16, bias=False):
        super(LoRDepthWiseSepConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            nin, rank, kernel_size=kernel_size, padding=padding, groups=rank, bias=bias
        )
        self.pointwise = nn.Conv1d(rank, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class LoRDepthWiseSepConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, rank=16, bias=False):
        super(LoRDepthWiseSepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, rank, kernel_size=kernel_size, padding=padding, groups=rank, bias=bias
        )
        self.pointwise = nn.Conv2d(rank, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=0)
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU()

        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TransformerModule(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            activation=nn.SiLU(),
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.models = nn.TransformersEncoder(encoder, num_layers)

    def forward(self, x):
        return self.models(x)
