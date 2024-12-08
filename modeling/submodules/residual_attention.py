import torch.nn as nn


class ResidualAttention(nn.Module):
    def __init__(self, layers=1, heads=1, d_model=512):
        nn.Module.__init__(self)

        self.activation = nn.LeakyReLU(negative_slope=0.02)

        attention_layers = []
        for _ in range(layers):
            attention_layers.append(
                nn.MultiheadAttention(d_model, heads, batch_first=True)
            )
        self.attention_layers = nn.ModuleList(attention_layers)

        feedforward_layers = []
        for _ in range(layers):
            # feedforward_layers.append(nn.Linear(d_model, d_model))
            feedforward_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.LeakyReLU(negative_slope=0.02),
                    nn.Linear(d_model * 4, d_model),
                )
            )
        self.feedforward_layers = nn.ModuleList(feedforward_layers)

        attn_norm_layers = []
        for _ in range(layers):
            attn_norm_layers.append(nn.LayerNorm(d_model))
        self.attn_norm_layers = nn.ModuleList(attn_norm_layers)

        ff_norm_layers = []
        for _ in range(layers):
            ff_norm_layers.append(nn.LayerNorm(d_model))
        self.ff_norm_layers = nn.ModuleList(ff_norm_layers)

    def forward(self, q, kv):
        for i in range(len(self.attention_layers)):

            q_out, _ = self.attention_layers[i](q, kv, kv)
            q = self.attn_norm_layers[i](q_out + q)

            q_out = self.activation(self.feedforward_layers[i](q))
            q = self.ff_norm_layers[i](q_out + q)
        return q


class CrossSequenceAttention(nn.Module):
    def __init__(self, layers=1, heads=1, d_model=512):
        nn.Module.__init__(self)

        self.q_attention = ResidualAttention(layers, heads, d_model)
        self.kv_attention = ResidualAttention(layers, heads, d_model)
        self.cross_attention = ResidualAttention(layers, heads, d_model)

    def forward(self, q, kv):
        q_out = self.q_attention(q, q)
        kv_out = self.kv_attention(kv, kv)
        cross_out = self.cross_attention(q_out, kv_out)

        return cross_out, q_out, kv_out
