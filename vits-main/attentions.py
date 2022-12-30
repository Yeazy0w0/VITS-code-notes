import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from modules import LayerNorm
   
# Encoder 类继承自 PyTorch 的 nn.Module 类，表示它是一个神经网络模块。它实现了一个编码器，可以将输入序列编码成隐藏状态。在初始化时，该类接受了若干参数，例如隐藏通道数、过滤通道数、头数和层数。在 forward 方法中，它会对输入序列执行多层注意力和前馈网络，并返回编码后的隐藏状态。
class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x

# Decoder 类也继承自 nn.Module 类，表示它是一个神经网络模块。它实现了一个解码器，可以根据编码后的隐藏状态和输入序列，生成输出序列。在初始化时，该类接受了若干参数，例如隐藏通道数、过滤通道数、头数和层数。在 forward 方法中，它会对输入序列和编码后的隐藏状态执行多层注意力和前馈网络，并返回生成的输出序列。
class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, h, h_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
      
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x

    
# 这两个模块类可以被用来构建一个序列到序列模型，其中输入序列通过编码器编码成隐藏状态，再通过解码器生成输出序列。
# 在 Encoder 类中，有一个成员变量 attn_layers，是一个 PyTorch 的 nn.ModuleList 类型，用于存储多层注意力层。
# 每一层注意力层都是由一个 MultiHeadAttention 类来实现的。norm_layers_1 是一个 nn.ModuleList 类型，用于存储多层归一化层，每一层归一化层都是由一个 LayerNorm 类来实现的。
# ffn_layers 是一个 nn.ModuleList 类型，用于存储多层前馈网络层，每一层前馈网络层都是由一个 FFN 类来实现的。norm_layers_2 是一个 nn.ModuleList 类型，用于存储多层归一化层，每一层归一化层都是由一个 LayerNorm 类来实现的。
# 在 forward 方法中，首先计算注意力矩阵的遮挡矩阵 attn_mask，然后对输入序列执行多层注意力和前馈网络。每一层注意力层都会调用 MultiHeadAttention 类的 forward 方法，每一层前馈网络层都会调用 FFN 类的 forward 方法。
# 在每一层注意力和前馈网络之后，都会调用一层归一化层来归一化输入和输出。最后，返回编码后的隐藏状态。
# Decoder 类的实现方式与 Encoder 类似，也有多层注意力层、归一化层和前馈网络层。不同的是，Decoder 类还包含了自注意力层和编码器解码器注意力层。自注意力层用于对解码器自身的隐藏状态进行注意力。编码器解码器注意力层用于在解码过程中，将编码后的隐藏状态与解码器自身的隐藏状态融合在一起。
# 在 forward 方法中，首先计算注意力矩阵的遮挡矩阵 attn_mask，然后对解码器的输入和隐藏状态执行多层注意力和前馈网络。每一层注意力层都会调用对应的 MultiHeadAttention 类的 forward 方法，每一层前馈网络层都会调用对应的 FFN 类的 forward 方法。在每一层注意力和前馈网络之后，都会调用一层归一化层来归一化输入和输出。最后，返回生成的输出序列。

# 多头注意力机制 (multi-head attention) 是 Transformer 模型中用于自注意力机制 (self-attention) 的一种改进方法，它将注意力机制分成多个独立的头 (heads)，每个头都在输入上进行注意力机制，然后将各个头的输出拼接在一起作为整个多头注意力机制的输出。这样做的优点是可以提高自注意力机制的表示能力，并且可以使得每个头注意到的信息不同，从而使得模型更加灵活。
# 该模型有三个输入：输入张量 x、上下文张量 c 和注意力掩码 attn_mask，并输出一个张量 x。
# 在初始化函数中，定义了模型的超参数，包括输入和输出张量的通道数，注意力头数量，以及注意力机制的一些辅助参数，如 dropout 率、窗口大小、是否共享注意力头等。然后创建了四个 1D 卷积层，分别用于处理输入张量、上下文张量的查询、键和值。如果定义了窗口大小，则还定义了两个参数张量，用于在注意力机制中使用相对位置信息。
# 在前向传播函数中，使用四个 1D 卷积层分别处理输入张量和上下文张量的查询、键和值，然后调用 attention 函数进行注意力机制的计算，最后使用最后一个 1D 卷积层将注意力机制的输出转换为最终输出 x。
# attention 函数中，使用矩阵乘法将查询张量和键张量相乘得到注意力权重，然后根据是否使用了相对位置信息，使用不同的方法计算最终的注意力权重。
class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

# 这段代码实现了一个叫做 FFN（Feed Forward Network）的模型。
# FFN 模型主要用于在序列中提取线性关系。它通常用于多头注意力模型（例如 Transformer 模型）中，用于将多头注意力模型提取的信息融合到输出中。
# FFN 模型包含两个卷积层。输入数据通过这两个卷积层处理，然后得到最终输出。两个卷积层之间还包含一个激活函数（默认使用 ReLU），以及一个 Dropout 层（默认保留率为 0）。
# 这段代码的参数列表如下：
# in_channels：输入数据的通道数。
# out_channels：输出数据的通道数。
# filter_channels：卷积层的通道数。
# kernel_size：卷积核的大小。
# p_dropout：Dropout 层的保留率。
# activation：激活函数的名称。
# causal：是否使用防止滞后的填充。
# 在调用模型时，你可以传入一个输入数据和一个输入数据的掩码（mask）。在输入数据通过两个卷积层时，掩码会被乘到输入数据上，以便控制哪些位置的数据需要被忽略。
class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    if causal:
      self.padding = self._causal_padding
    else:
      self.padding = self._same_padding

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(self.padding(x * x_mask))
    return x * x_mask
  
  def _causal_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = self.kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

  def _same_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x
