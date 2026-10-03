"""最小化的多模态 Model 层组件，用于学习和 shape tracing。"""

import torch
from torch import nn


class LlavaStyleVisionProjector(nn.Module):
    """把视觉塔输出的 patch token 投影到语言模型 hidden size。"""

    def __init__(self, vision_dim: int, language_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, language_dim * 2),
            nn.GELU(),
            nn.Linear(language_dim * 2, language_dim),
        )

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        # [B, N_vision, D_vision] -> [B, N_vision, D_language]
        return self.projector(vision_tokens)


class PerceiverCrossAttention(nn.Module):
    """用少量 query 从视觉 token 中读取信息，压缩模态序列长度。"""

    def __init__(self, dim: int, num_queries: int, num_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self, vision_tokens: torch.Tensor, vision_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = vision_tokens.size(0)
        queries = self.queries.expand(batch_size, -1, -1)
        attended, _ = self.attention(
            self.norm_q(queries),
            self.norm_kv(vision_tokens),
            self.norm_kv(vision_tokens),
            key_padding_mask=vision_padding_mask,
            need_weights=False,
        )
        compressed = queries + attended
        return compressed + self.ffn(compressed)


class TinyMultimodalDecoder(nn.Module):
    """用于验证视觉 token 与文本 embedding 拼接后的统一序列建模。"""

    def __init__(self, vocab_size: int, dim: int, num_layers: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        text = self.token_embedding(input_ids)
        sequence = torch.cat([visual_tokens, text], dim=1)
        return self.lm_head(self.decoder(sequence))
