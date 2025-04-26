import torch
import torch.nn as nn
import torch.nn.functional as F

# 改进的自注意力机制，结合轻量化卷积
class OptimizedAttention(nn.Module):
    def __init__(self, dim, num_heads=1, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        # 引入轻量化卷积来增强局部特征处理能力
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 使用卷积增强局部特征
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.conv(x)  # [B, C, N]
        x = x.permute(0, 2, 1)  # [B, N, C]

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 改进的 Transformer 编码层，增强层归一化和残差连接
class OptimizedTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.self_attn = OptimizedAttention(dim=dim, num_heads=num_heads, attention_dropout=attention_dropout)

        # 提前使用 LayerNorm，以改善特征归一化效果
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 简化的前馈网络，保持模型的轻量化
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # 改进的残差连接和 LayerNorm 结合注意力机制
        src2 = self.self_attn(self.norm1(src))
        src = src + self.dropout(src2)

        src2 = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(src)))))
        src = src + self.dropout(src2)

        return src


# 改进的 Transformer 模型，支持可学习的位置嵌入
class OptimizedTransformer(nn.Module):
    def __init__(self, sequence_length=25, dim=640, num_layers=1, num_heads=1, dropout=0.1, attention_dropout=0.1, positional_embedding='learnable'):
        super().__init__()

        self.sequence_length = sequence_length
        self.embedding_dim = dim

        # 改为可学习的位置嵌入
        if positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_emb, std=0.02)
        else:
            self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, dim), requires_grad=False)

        self.layers = nn.ModuleList([
            OptimizedTransformerEncoderLayer(dim=dim, num_heads=num_heads, dropout=dropout, attention_dropout=attention_dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x += self.positional_emb

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    @staticmethod
    def sinusoidal_embedding(sequence_length, dim):
        pe = torch.zeros(sequence_length, dim)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# 改进的 FSRM 主模型，保留轻量化并增强局部特征处理
class OptimizedFSRM(nn.Module):
    def __init__(self, sequence_length=25, embedding_dim=640, num_layers=1, num_heads=1, dropout=0.1, attention_dropout=0.1):
        super(OptimizedFSRM, self).__init__()
        self.transformer = OptimizedTransformer(sequence_length=sequence_length, dim=embedding_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout, attention_dropout=attention_dropout)
        self.flattener = nn.Flatten(2, 3)

    def forward(self, x):  # 输入 [200, 640, 5, 5]
        x = self.flattener(x).transpose(-2, -1)  # 展平后 [200, 25, 640]
        x = self.transformer(x)  # 通过 Transformer 处理后的输出 [200, 25, 640]
        return x
