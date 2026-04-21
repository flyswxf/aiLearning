import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding)
    使用正弦和余弦函数为序列中的每个位置生成固定的编码
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # position : 
        # [[0],
        # [1],
        # [2],
        # [3],
        # [4]
        # ...]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 0::2和1::2的结果要相同, 这要求d_model必须为偶数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch_size 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 注册为 buffer，这样它不会被视为模型参数进行更新，但会随模型保存
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 的线性映射
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 最后的输出线性映射
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 将 mask 中为 0 的位置替换为极小值，使得 softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
        # 最后两维代表一个句子中的一个词对其他位置的词的关注度(权重)
        attn_weights = F.softmax(scores, dim=-1)
        # 行*列, 对于一个位置的词, 用它对于其他位置的词的关注度作为权重, 对其他位置的词的特征做加权和. 
        # 比如从第一行从左到右, 因为是同一个词, 关注度(权重)是相同的, 依次对别的词的第一维, 第二维...特征做加权和, 最终形成这个词新的特征
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性映射并分头 (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            # 计算注意力
            output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        else:
            output, attn_weights = self.scaled_dot_product_attention(q, k, v)

        # 拼接多头 (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最后经过一次线性映射
        return self.W_o(output)


class PositionwiseFeedForward(nn.Module):
    """
    基于位置的前馈神经网络 (Position-wise Feed-Forward Network)
    因为它只对一个词内部的维度做线性变换, 而不关注词与词之间的关系
    """

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi-Head Self Attention + Add & Norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Feed Forward + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1. Masked Multi-Head Self Attention + Add & Norm
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2. Cross Multi-Head Attention + Add & Norm (Query 来源 Decoder，Key/Value 来源 Encoder)
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 3. Feed Forward + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=5000,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # 编码器堆叠
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # 解码器堆叠
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # 输出映射层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_mask(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        """
        生成 Source Mask 和 Target Mask
        """
        # src_mask 仅掩码 padding: (batch_size, 1, 1, src_len)
        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

        # tgt_mask 需要掩码 padding 和 未来的信息
        tgt_seq_len = tgt.size(1)
        # 构造下三角矩阵，防止看到未来信息 (1, seq_len, seq_len)
        nopeak_mask = torch.tril(
            torch.ones((1, tgt_seq_len, tgt_seq_len), device=tgt.device)
        ).bool()
        # padding 掩码
        tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # 结合两者
        tgt_mask = tgt_pad_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        # 1. 生成 masks
        src_mask, tgt_mask = self.generate_mask(src, tgt, src_pad_idx, tgt_pad_idx)

        # 2. Encoder
        enc_out = self.dropout(
            self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        )
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        # 3. Decoder
        dec_out = self.dropout(
            self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        )
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        # 4. Output
        out = self.fc_out(dec_out)
        return out


if __name__ == "__main__":
    # 测试参数设置
    src_vocab_size = 1000
    tgt_vocab_size = 1500
    d_model = 256
    num_heads = 8
    num_layers = 4
    batch_size = 2
    src_len = 10
    tgt_len = 12

    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # 构造假数据 (随机生成 token 索引)
    # 假设 pad_idx 为 0
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    # 故意加入一些 pad token 用于测试 mask
    src[0, 8:] = 0
    tgt[0, 10:] = 0

    print("Source Shape:", src.shape)
    print("Target Shape:", tgt.shape)

    # 前向传播
    output = model(src, tgt)

    print("Output Shape:", output.shape)
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size), "输出维度不匹配！"
    print("模型测试通过！")
