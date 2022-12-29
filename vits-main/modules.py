import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import init_weights, get_padding
from transforms import piecewise_rational_quadratic_transform


LRELU_SLOPE = 0.1

# LayerNorm 是 PyTorch 中的一种常用的层归一化方法。它的作用是对输入的每一个通道的数据分别进行归一化，使得每个通道的数据的均值为 0，方差为 1。
# 这个类有两个参数：
# channels：需要归一化的通道数。
# eps：为了避免除以 0 的情况，在计算均值和方差时会加上一个小常数 $\epsilon$。
# 具体来说，对于输入的每一个通道，我们计算出它的均值 $\mu$ 和方差 $\sigma^2$，然后将它的每个元素 $x$ 通过以下方式进行归一化：
# $$\frac{x - \mu}{\sigma}$$
# 在实现中，我们可以使用 PyTorch 的 F.layer_norm 函数来实现上述过程。这个函数接受一个输入 x 和两个参数 $\gamma$ 和 $\beta$，其中 $\gamma$ 和 $\beta$ 分别是 $\frac{1}{\sigma}$ 和 $-\frac{\mu}{\sigma}$，然后对输入进行归一化并返回结果。
# 在这个类中，我们将 $\gamma$ 和 $\beta$ 分别视为可学习的参数，并初始化为全 1 和全 0。最后，我们通过调用 F.layer_norm 函数来实现层归一化。
class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)

# 这个类定义了一个用于计算 1D 卷积的神经网络模块。它包含了一系列的卷积层、归一化层、ReLU 激活函数和 Dropout 正则化，具体的实现细节如下：
# 定义了几个超参数：输入通道数、隐藏层通道数、输出通道数、卷积核大小、层数、Dropout 比例。
# 创建了几个空的模块列表，用于保存卷积层和归一化层。
# 创建第一个卷积层和归一化层，并将它们加入模块列表。
# 创建一个包含 ReLU 和 Dropout 两个子模块的模块。
# 使用循环创建剩余的卷积层和归一化层，并将它们加入模块列表。
# 创建一个 1x1 卷积层，用于将隐藏层映射到输出层，并将其权重和偏置初始化为 0。
# 定义 forward 函数，用于计算输入的 1D 卷积。首先定义一个副本 x_org 用于保存输入，然后使用循环依次计算每一层的输出，最后将副本加上最后一层的输出并返回结果。
class ConvReluNorm(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    assert n_layers > 1, "Number of layers should be larger than 0."

    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    for _ in range(n_layers-1):
      self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x, x_mask):
    x_org = x
    for i in range(self.n_layers):
      x = self.conv_layers[i](x * x_mask)
      x = self.norm_layers[i](x)
      x = self.relu_drop(x)
    x = x_org + self.proj(x)
    return x * x_mask

# 这是一个用于自然语言处理任务的卷积神经网络模块。它包含一些卷积层，批标准化层和 ReLU 激活函数，并使用了扩张卷积和深度可分离卷积来进行特征提取。
# 此外，还使用了 dropout 正则化来防止过拟合。该模块的输入是原始输入和一个可选的网络间跳跃（g），其输出是提取的特征。
class DDSConv(nn.Module):
  """
  Dialted and Depth-Separable Convolution
  """
  def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
    super().__init__()
    self.channels = channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.convs_sep = nn.ModuleList()
    self.convs_1x1 = nn.ModuleList()
    self.norms_1 = nn.ModuleList()
    self.norms_2 = nn.ModuleList()
    for i in range(n_layers):
      dilation = kernel_size ** i
      padding = (kernel_size * dilation - dilation) // 2
      self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, 
          groups=channels, dilation=dilation, padding=padding
      ))
      self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
      self.norms_1.append(LayerNorm(channels))
      self.norms_2.append(LayerNorm(channels))

  def forward(self, x, x_mask, g=None):
    if g is not None:
      x = x + g
    for i in range(self.n_layers):
      y = self.convs_sep[i](x * x_mask)
      y = self.norms_1[i](y)
      y = F.gelu(y)
      y = self.convs_1x1[i](y)
      y = self.norms_2[i](y)
      y = F.gelu(y)
      y = self.drop(y)
      x = x + y
    return x * x_mask

# 这是一个 PyTorch 实现的 Wavenet 模型的一部分。Wavenet 是一种生成模型，常用于生成语音或音乐。这部分代码实现了 Wavenet 中的一个组件，称为 Dilated Convolution（扩张卷积）。这种卷积方式通过膨胀卷积核来增加模型的感受野，从而能够捕捉较大范围的信息。
# 另外，这个组件还使用了权值归一化（Weight Normalization）来控制模型的复杂度
class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      padding = int((kernel_size * dilation - dilation) / 2)
      in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, padding=padding)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = commons.fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)

# 这两个类定义了 PyTorch 中的模块，即神经网络的一个部分。它们都是扩展自 torch.nn.Module 类，并定义了向前传递（forward）函数，该函数定义了在进行反向传播时如何计算输入的输出。
# ResBlock1 类定义了一个包含两个部分的残差块。第一部分包含三个带膨胀卷积（dilated convolution）的卷积层，而第二部分包含另外三个卷积层。每个部分中的卷积层的膨胀系数（dilation factor）分别在构造函数中指定。另外，这两个部分中的卷积层都使用了权重归一化（weight normalization），这是一种用于解决深度学习中的梯度消失/爆炸问题的方法。
# ResBlock2 类定义了一个包含一个部分的残差块，该部分包含两个带膨胀卷积的卷积层。与 ResBlock1 类相似，这两个卷积层都使用了权重归一化。
# 两个类都定义了一个名为 remove_weight_norm 的方法，该方法用于在训练过程结束后移除卷积层中的权重归一化，以便在推断（inference）时更快地进行前向计算。
class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

# Log 类定义了一个 PyTorch 模块，该模块对输入的张量进行对数变换。
# 具体来说，该模块的输入是一个张量 x 和一个用于掩码（mask）的张量 x_mask。如果 reverse 参数设置为 False，则该模块会将 x 中的所有元素取对数，并返回变换后的张量和对数行列式的值。如果 reverse 参数设置为 True，则该模块会将变换后的张量取指数，并返回反向变换后的张量。
class Log(nn.Module):
  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
      logdet = torch.sum(-y, [1, 2])
      return y, logdet
    else:
      x = torch.exp(x) * x_mask
      return x
    
# Flip 类定义了一个 PyTorch 模块，该模块的作用是翻转输入的张量。
# 具体来说，该模块的输入是一个张量 x。如果 reverse 参数设置为 False，则该模块会将 x 的第一维（第一个维度）翻转，并返回翻转后的张量和行列式的值。如果 reverse 参数设置为 True，则该模块会返回翻转前的张量。
class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x

# ElementwiseAffine 类定义了一个 PyTorch 模块，该模块可以对输入的张量的每个元素进行仿射变换，即进行加法和乘法。
# 具体来说，该模块有两个参数 m 和 logs，分别表示加法和乘法的参数。在向前传递（forward）时，如果 reverse 参数设置为 False，则对输入张量 x 进行仿射变换，并返回变换后的结果和变换所产生的对数行偏移量。如果 reverse 设置为 True，则反向进行仿射变换，并返回变换后的结果。
# 这个模块还有一个输入参数 x_mask，表示一个用于掩码（mask）的张量。它用于指定哪些元素应该被计算，哪些应该被忽略。这在模型的训练过程中很有用，因为它可以避免在序列的结尾部分计算不必要的元素。
class ElementwiseAffine(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.channels = channels
    self.m = nn.Parameter(torch.zeros(channels,1))
    self.logs = nn.Parameter(torch.zeros(channels,1))

  def forward(self, x, x_mask, reverse=False, **kwargs):
    if not reverse:
      y = self.m + torch.exp(self.logs) * x
      y = y * x_mask
      logdet = torch.sum(self.logs * x_mask, [1,2])
      return y, logdet
    else:
      x = (x - self.m) * torch.exp(-self.logs) * x_mask
      return x

# forward中对x进行分割，分割成两部分，得到x0（上面的）和x1（下面的）。x0会经过一个pre层，再经过encoder层得到新的h，再让h经过post层，得到stats（一个统计量）。
# 如果设置的mean_only是false的话，就会输出均值和标准差，如果是true，就直接预测平移量，不会预测缩放量。
# x0没变过，基于x0对x1进行变换，得到新的x1。
# 如果不是reverse就会返回logdet。如果mean_only等于true的话，这里logdet就等于0。
class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x

# ConvFlow 类定义了一个 PyTorch 模块，该模块可以执行分段极端值函数（piecewise rational quadratic function）的变换，即对输入的张量进行分离变换，并使用一个可以学习的卷积网络进行编码。
# 这个模块的输入是一个张量 x 和一个掩码张量 x_mask，它可以通过设置 reverse 参数来进行反向变换。它还有一个可选的输入 g，用于指定变换中使用的参数（如果提供的话）。
# 在向前传递（forward）时，如果 reverse 参数设置为 False，则模块会对输入张量进行分段极端值函数的变换。首先，它会将输入张量按通道数分成两部分。然后使用 pre 将其中一部分缩小为过滤通道，再使用 convs 对这部分进行编码，最后使用 proj 将编码结果扩展为三倍的通道数。
# 接下来，模块会对另一部分进行分段极端值函数的变换。如果设置了 g，则会在 piecewise_rational_quadratic_transform 中使用 g 作为变换参数；如果未设置 g，则使用从编码中学习的参数。然后，将两部分合并为一个张量，并返回变换后的结果
class ConvFlow(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
    super().__init__()
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.num_bins = num_bins
    self.tail_bound = tail_bound
    self.half_channels = in_channels // 2

    self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
    self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.)
    self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0)
    h = self.convs(h, x_mask, g=g)
    h = self.proj(h) * x_mask

    b, c, t = x0.shape
    h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2) # [b, cx?, t] -> [b, c, t, ?]

    unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
    unnormalized_derivatives = h[..., 2 * self.num_bins:]

    x1, logabsdet = piecewise_rational_quadratic_transform(x1,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=reverse,
        tails='linear',
        tail_bound=self.tail_bound
    )

    x = torch.cat([x0, x1], 1) * x_mask
    logdet = torch.sum(logabsdet * x_mask, [1,2])
    if not reverse:
        return x, logdet
    else:
        return x
