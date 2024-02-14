import torch
from torch import nn
from .modules import TransformerModule, ConvModule


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvTransformerMappingNetworks(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        prefix_len: int = 50,
        num_heads: int = 4,
        num_layers: int = 4,
        is_temporal: bool = True,
    ):
        super().__init__()

        self.is_temporal = is_temporal

        self.cnn = ConvModule(in_dim, out_dim, kernel_size, stride)
        self.transformers = TransformerModule(out_dim, num_heads, num_layers)

        if is_temporal:
            self.pos_encoding = PositionalEncoding(d_model=out_dim)

        self.prefix_const = nn.Parameters(
            torch.rand(prefix_len, out_dim), requires_grad=True
        )
        torch.nn.init.kaiming_uniform_(self.prefix_const)

    def forward(self, x):
        if not self.is_temporal:
            # (bs, 527)
            x = x.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, 527)
        else:
            # x (bs, 2048, len, 2)
            x = x.permute(0, 1, 3, 2)  # (bs, 2048, 2, len)

        x = self.cnn(x)
        # temporal -> (bs, out_dim, len, 1)
        # global -> (bs, out_dim, len, 1)

        x = x.squeeze(3)

        if self.is_temporal:
            x = self.pos_encoding(x)

        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, -self.prefix_length :]
        return out


class LocalGlobalTemporalMappingNetworks(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        temporal_prefix_len: int = 60,
        global_prefix_len=20,
        num_heads=8,
        num_layers=8,
    ):
        super().__init__()
        self.temporal_map_net = ConvTransformerMappingNetworks(
            in_dim=in_dim,
            out_dim=out_dim,
            kernel_size=(2, 1),
            stride=(1, 1),
            prefix_len=temporal_prefix_len,
            num_heads=num_heads,
            num_layers=num_layers,
            is_temporal=True,
        )
        self.global_map_net = ConvTransformerMappingNetworks(
            in_dim=in_dim,
            out_dim=out_dim,
            kernel_size=(1, 48),
            stride=(1, 48),
            prefix_len=global_prefix_len,
            num_heads=num_heads,
            num_layers=num_layers,
            is_temporal=False,
        )

    def forward(self, temporal_embeds, global_embeds):
        temporal_prefix_vectors = self.temporal_map_net(temporal_embeds)
        global_prefix_vectors = self.global_map_net(global_embeds)

        # concat by time
        out = torch.cat([temporal_prefix_vectors, global_prefix_vectors], dim=1)

        return out
