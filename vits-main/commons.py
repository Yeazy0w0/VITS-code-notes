import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# init_weights 是一个辅助函数，用于初始化模型中的权重。
def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)

# 计算卷积核的填充大小
def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)

# 转换填充形状
def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

# 将列表中的每个元素之间插入一个指定元素
def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

# 计算 KL 散度
def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q)**2)) * torch.exp(-2. * logs_q)
  return kl

# 从 Gumbel 分布中采样
def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
  return -torch.log(-torch.log(uniform_samples))

# 返回与输入张量相似的随机 Gumbel 张量
def rand_gumbel_like(x):
  g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
  return g

# 根据指定的起始位置，将序列分割成指定大小的片段
def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret

# 随机分割序列
def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str

# 获取 1D 时序信号
def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(
      torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  return signal

# 将 1D 时序信号添加到序列中
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal.to(dtype=x.dtype, device=x.device)

# 用于将输入张量x和一个1维时序信号拼接在一起，并返回新的张量。时序信号是一个增强了输入张量的时间信息的维度，可以加快训练过程并提高模型的性能。
# 函数的输入包括：
# x：一个三维张量，其中包含输入信息。它应该具有形状(batch_size, channels, length)。
# min_timescale：一个浮点数，表示每个信道使用的最小时间尺度。
# max_timescale：一个浮点数，表示每个信道使用的最大时间尺度。
# axis：一个整数，表示x和时序信号的拼接位置。
# 函数会返回一个四维张量，其中第三维为输入张量x的第三维和时序信号的第一维，其余维度与x相同。
# 例如，假设我们有一个具有形状(batch_size=2, channels=3, length=4)的输入张量x，以及min_timescale=1.0和max_timescale=1.0e4。在这种情况下，时序信号将具有形状(3, 4)，并且cat_timing_signal_1d函数将返回一个形状为(batch_size=2, channels=6, length=4)的四维张量。
def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

# subsequent_mask 函数定义了一个三角形矩阵，其中最上面的行全部填充 1，其余部分填充 0。结果会被扩展为一个形状为 (1, 1, length, length) 的张量。这个函数可能被用来生成掩码，以便在自注意力机制中将位置后面的输入标记为无效。
def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask

# 这段代码定义了一个名为fused_add_tanh_sigmoid_multiply的函数。
# 这个函数接受三个输入：
# input_a: 一个形状为 (batch_size, n_channels, length) 的张量。
# input_b: 一个形状为 (batch_size, n_channels, length) 的张量。
# n_channels: 一个形状为 (1,) 的张量，表示n_channels的数值。
# 这个函数会返回一个形状为 (batch_size, n_channels, length) 的张量，它是以下几个操作的结果：
# 对输入的input_a和input_b进行元素级加法。
# 对求和的结果进行tanh操作。
# 对求和的结果进行sigmoid操作。
# 将tanh的结果和sigmoid的结果进行元素级乘法。
# 注意，这个函数是用@torch.jit.script装饰的，这意味着它是用TorchScript编写的，可以在TorchScript中使用，并且可以在PyTorch的许多平台上运行。
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts

# 这段代码定义了一个函数 convert_pad_shape，这个函数用于将一个给定的矩形填充列表转换成一维列表。
# 矩形填充列表是用于指定矩形填充的元素的顺序和数量的三维列表，其中第一维是填充的轴，第二维是填充的左边和右边，第三维是填充的数量。例如，[[1, 2], [3, 4], [5, 6]] 表示在第一个轴上的填充分别为 1 和 2，在第二个轴上的填充分别为 3 和 4，在第三个轴上的填充分别为 5 和 6。
# 函数 convert_pad_shape 首先将矩形填充列表翻转（将第一维移动到最后一维，将第二维移动到倒数第二维等），然后使用列表推导式将矩形填充列表扁平化成一维列表。最后，函数返回转换后的一维填充列表。
def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape

# shift_1d 函数实现了在一维上对数据进行移位。具体来说，首先使用 PyTorch 的 F.pad 函数对数据进行 padding，然后使用切片操作将最后一个维度的最后一个元素去掉，即右移一个位置。在这里，数据的最后一维代表时间步，因此对数据进行移位就是在时间步上进行移位。
def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x

# 这是一个生成序列掩码的函数。它接受一个长度张量并返回一个形状为 (len(length), max_length) 的掩码矩阵，该矩阵将包含所有序列的掩码。 序列掩码是一个二元矩阵，其中为长度小于等于序列的位置填充 1，否则填充 0。
# 这个函数的主要用途是为序列模型的输入设置掩码，以便在计算损失时忽略无效位置的贡献。在使用序列掩码时，通常会为模型的输出和目标序列使用相同的掩码。
def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)

# 这段代码实现了生成路径的功能。其中，duration是一个三维张量，表示序列的每一个元素的持续时间；mask是一个四维张量，表示源序列和目标序列的关系，mask[b, 1, t_y, t_x]的值为1表示源序列的第b个元素被映射到目标序列的第t_y个位置。
# 具体来说，首先使用torch.cumsum()函数计算出cum_duration，它的值为[b, 1, t_x]的张量，表示序列的每一个元素的累计持续时间。然后使用sequence_mask()函数创建一个矩阵，其中包含了序列中每一个元素的位置。最后，将这个矩阵转化为一个四维张量，并且与mask相乘，得到结果path。
def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device
  
  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2,3) * mask
  return path

# 这是一个 PyTorch 计算图中的辅助函数，它将被用于在训练模型时约束梯度的大小。
# 具体来说，它会在传入的参数列表中查找每个有梯度的张量，然后对这些张量的梯度使用 PyTorch 的 clamp_ 函数进行裁剪，以使其的每个元素的值都在 -clip_value 和 clip_value 之间。这种裁剪是为了防止梯度爆炸或消失，从而更好地训练模型。
# 此外，该函数还会计算所有被裁剪的梯度的范数，并返回这些范数的和。范数的类型由参数 norm_type 指定，默认情况下是二范数。
# 在函数名中的下划线表示这是一个原地修改操作，也就是说它会直接修改传入的参数列表中的张量，而不是返回一个新的张量。
def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm

# 这段代码定义的函数主要是一些通用的函数，包含了一些常用的计算和处理张量的方法。
# 它们可以用来解决在深度学习模型开发中经常遇到的一些公共问题，比如权重初始化、计算 KL 散度、获取时序信号等。
# 这些函数是用来支持模型的开发和训练的，而不是实际执行模型的计算的。
