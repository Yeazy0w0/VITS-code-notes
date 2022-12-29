import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding


# 这个随机时长预测器是根据上一步单调对齐搜索，也就是动态规划的搜索中得到的对齐矩阵，根据矩阵对频谱那个维度进行求和得到每个文本持续的时长。基于这个时长，可以知道StochasticDurationPredictor的训练。
# 之所以用随机时长预测器就是因为这种做法可以得到表现力更强的模型，比如一个人不同的时候说话表现出不同语速和节奏。
# 随机时长其实是预测每个因素的时长分布，从这个分布中采样，就能得到这个因素的时长，这样来体现随机性。具体的采样是基于冲参数化采样的。
# sdp（随机时长预测器）是基于Flow设计的，基于最大似然估计做了优化。
# 这里的x是文本编码状态，w是时长，g是条件。
# x = torch.detach(x)进行分离，因为这个时长模型梯度更新的时候，不能去影响文本编码器，因此在这里对x做一个截断。
# 然后x经过一个pre层，这个pre层是一个一维卷积，得到卷积后的x。
# 如果是多说话人的模型，要把speaker_Embedding加进来。
# x = self.convs(x, x_mask)和x = self.proj(x) * x_mask对x进行预处理，得到新的x，就是论文中的c。
# 训练的时候走not reverse这个逻辑，首先得到flows，就是先验的flow。
# h_w = self.post_pre(w)、h_w = self.post_convs(h_w, x_mask)和h_w = self.post_proj(h_w) * x_mask是对时长进行预处理，得到h_w，h_w和x作为后验分布的条件。
# 下一行代码会生成一个随机噪声，这个噪声和w的形状相关。因为这里用的是变分增广的flow，引入了另外一个随机量，所以特征维度是2不是1。
# z_q分别经过很多个flow进行遍历，每次遍历都会把上一个flow的输出作为下一个flow的输入，同时引入条件g。
# 每次经过flow之后，它的对数似然的变化量logdet_q也能返回。同时flow变换后的z变量也能返回。
# 然后对z_q进行分割，分割成z_u和z1，再得到u。现在就可以算出后验分布的对数似然了。
class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      # 下面两行就是后验分布的对数似然。
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q
      # 以上是后验分布（训练部分），下面是先验分布的部分。

      # z0就是预测的时长，z0经过log_flow，再把z0和辅助的变量z1拼起来，得到新的z。
      # nll是先验分布的对数似然。
      # 先验分布的对数似然nll和后验分布的对数似然logq加起来，表示的是我们要优化的负对数似然的上界。
      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    # 推理阶段，先对先验分布的flow翻转一下，再将中间无用的flow过滤掉，然后会生成一个随机噪声z。
    # 对反转过的flow进行一次次的经过变换，z就从一个简单的分布变成了复杂的分布。
    # 对z进行split得到z0和z1，z0是log时长。
    # 返回logw(log时长)，主函数中会对logw做一个计算得到最终的w，再经过一个int操作就可以取整，得到整数的时长。
    #得到整数的时长之后，就可以对引变量进行拓展，再送入解码器中，就可以得到预测的完整的波形。
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

# x表示单词的索引，单词先转化成Embedding，再缩放一下，送入基于transformer的encoder里面，再对它进行映射，得到两个统计量，一个均值和标准差的对数，是先验编码器预测的两个分布的参数。
class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


# 这个flow主要由两个模块构成的，第一个叫ResidualCouplingLayer，是一个耦合Flow，第二个叫Flip。
class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

# 这个后验编码器主要是由一维的卷积所构成的，x就是线性谱，g是global condition，后验编码器会接受说话人的身份作为条件。
class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask

# 这是一个生成器模型，用于生成波形数据。它包含了若干个反卷积层和残差块。
# 初始化时，需要传入的参数有：
# initial_channel：生成器的输入数据的通道数。
# resblock：残差块的类型，可以为'1'或'2'。
# resblock_kernel_sizes：残差块使用的卷积核大小的列表。
# resblock_dilation_sizes：残差块使用的膨胀率的列表。
# upsample_rates：反卷积层使用的上采样率的列表。
# upsample_initial_channel：反卷积层的输入通道数。
# upsample_kernel_sizes：反卷积层使用的卷积核大小的列表。
# gin_channels：输入的附加信息的通道数，默认为0，即不使用附加信息。
# 在进行前向传播时，生成器的输入为输入的波形数据 x 和可选的附加信息 g。生成器会对 x 和 g 进行处理，然后通过反卷积和残差块得到输出的波形数据。最后会将输出的波形数据通过 tanh 函数转换为 $[-1, 1]$ 区间内的值。
class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

# 这是一个 PyTorch 神经网络模型的类定义。这个模型被命名为 DiscriminatorP，它继承了 PyTorch 的 torch.nn.Module 类，表示它是一个 PyTorch 模型。
# 在这个类的构造函数中，定义了几个参数：
# period: 一个整数，表示周期。
# kernel_size: 卷积核的大小，默认值为 5。
# stride: 卷积的步幅，默认值为 3。
# use_spectral_norm: 是否使用光谱归一化，默认值为 False。
# 在构造函数中，调用了 super() 函数来初始化父类的构造函数。然后定义了几个成员变量：
# self.period: 保存周期的值。
# self.use_spectral_norm: 保存是否使用光谱归一化的布尔值。
# self.convs: 一个 PyTorch 的 ModuleList，包含多个卷积层。
# self.conv_post: 一个卷积层。
# 在构造函数的最后，定义了一个名为 forward 的方法，它定义了这个模型的前向计算过程。这个方法有一个输入参数 x，是输入的张量。这个方法通过调用 self.convs 中的卷积层并应用激活函数，对输入进行多次卷积，最后返回卷积后的结果。
# 在方法的开头，使用了一个名为 fmap 的列表来保存卷积层的输出。然后，对于输入的张量 x，将其从 1D 张量转换为 2D 张量。为了进行这种转换，需要获取输入张量的形状（通过调用 x.shape），然后使用 PyTorch 的 view 函数将其转换为 2D 张量。
# 然后，使用一个循环遍历 self.convs 中的每一个卷积层。对于每一个卷积层，使用 x 作为输入，调用卷积层的前向计算，并将结果保存在 x 中。接着，使用 PyTorch 的 F.leaky_relu 函数对结果进行激活，并将激活后的结果添加到 fmap 列表中。
# 最后，调用 self.conv_post 卷积层，并将结果保存在 x 中。然后，使用 PyTorch 的 torch.flatten 函数将结果压缩成一维张量，并返回结果和 fmap 列表。
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# 这是另一个 PyTorch 神经网络模型的类定义。这个模型被命名为 DiscriminatorS，它同样继承了 PyTorch 的 torch.nn.Module 类，表示它是一个 PyTorch 模型。
# 在这个类的构造函数中，定义了一个参数：
# use_spectral_norm: 是否使用光谱归一化，默认值为 False。
# 在构造函数中，调用了 super() 函数来初始化父类的构造函数。然后定义了一个成员变量：
# self.convs: 一个 PyTorch 的 ModuleList，包含多个卷积层。
# self.conv_post: 一个卷积层。
# 在构造函数的最后，定义了一个名为 forward 的方法，它定义了这个模型的前向计算过程。这个方法的实现方式与 DiscriminatorP 类中的 forward 方法非常相似，都是遍历 self.convs 中的卷积层，并调用卷积层的前向计算，最后返回卷积后的结果。
# 唯一的区别在于，这个模型中使用的是 1D 卷积层，而不是 2D 卷积层。 1D 卷积层在处理序列数据时非常有用，例如时间序列数据。
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# MultiPeriodDiscriminator，它同样继承了 PyTorch 的 torch.nn.Module 类，表示它是一个 PyTorch 模型。
# 在这个类的构造函数中，定义了一个参数：
# use_spectral_norm: 是否使用光谱归一化，默认值为 False。
# 在构造函数中，调用了 super() 函数来初始化父类的构造函数。然后定义了一个名为 periods 的列表，包含整数 2、3、5、7 和 11。
# 接着，使用这个列表创建了一个 DiscriminatorP 对象的列表。这些对象的周期分别为 2、3、5、7 和 11。然后，将这些对象包装在一个 PyTorch 的 ModuleList 中，并将它保存在 self.discriminators 中。
# 对于每一个 DiscriminatorP 对象，调用其 forward 方法时，使用 y 和 y_hat 分别作为输入。然后将每一次调用的结果分别保存在以下四个列表中：
# y_d_rs: 包含对 y 进行每一次前向计算的结果。
# y_d_gs: 包含对 y_hat 进行每一次前向计算的结果。
# fmap_rs: 包含每一次对 y 进行前向计算时的中间结果。
# fmap_gs: 包含每一次对 y_hat 进行前向计算时的中间结果。
# 最后，方法返回四个列表。这些列表中包含的数据可能用于进一步的计算或可视化。
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



# 这个就是generator，整个VITS从文本到波形都是在这个类里面实现的。
class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  # init函数对一些模型的参数进行指定，比如通道数、隐含层个数、注意力的头数、层数等。
  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.use_sdp = use_sdp

    # 文本被送入TextEncoder（文本编码器），以文本作为输入，将孤立的文本经过文本编码器表征成上下文相关的状态向量。得到文本的先验分布。
    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    # 波形生成器，将后验分布得到的z送入解码器中，得到波形。
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    # 后验编码器。
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    # flow，先验分布后加flow可以提高先验分布的表达能力。
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    # 这里定义了一个随机时长预测器（说话的韵律节奏）。
    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    # 说话人的身份
    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  # forward函数如何做前向运算的，如何基于VAE、Flow和GAN从文本生成波形
  # x是文本，x_lengths是长度，将这两个送入到先验编码器中得到编码后的x以及先验编码器出来的分布的均值和对数标准差，还有文本的mask。
  # m_p,是先验的均值，logs_p先验分布的对数标准差，对应论文中的fθ(z)的均值和标准差
  # enc_q是后验分布，g是说话人的身份，y是频谱，y_lengths是频谱的长度，都是从真实的音频中通过傅里叶变换得到的。将这些送入后验编码器得到z，m_q, logs_q以及y_mask。
  # m_q和logs_q是后验分布（高斯分布）的均值和对数标准差,z是以m_q和logs_q的高斯分布中采样得到的引变量。z的维度和y的时间长度是一致的，也就是说y有多少个，z就有多少个。
  # 后验分布和条件有关，先验分布和条件无关，只和文本有关。
  # z_p用在train_ms.py的loss_kl中。
  def forward(self, x, x_lengths, y, y_lengths, sid=None):

    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)

    # 这里做了一个动态规划，把z_p和先验分布对齐。因为z_p是频谱长度可能有几百帧，但是文本的m_p和logs_p可能只有一句话或者几个单词，因此需要将文本和频谱对齐。
    # 这部分对应论文中的单调对齐搜索。具体的算法在monotonic_align的core.pyx中。
    # 因为m_p和logs_p的维度是不一样的，所以这里用矩阵乘法计算。
    # 得到attn矩阵是一个零一矩阵，维度是[b,1,T_t,T_s]，w = attn.sum(2)将矩阵求和，这里的维度是[batch_size,1,T_t],其中T_t是text的长度。
    # 得到w之后，训练的时候是有后验分布的，但是推理的时候是没有后验分布的，因此需要对先验分布进行拓展，拓展的一局需要一个额外的时长模型，就是随机的时长预测器sdp。
    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    # 这里意思是，如果我们使用随机时长预测器或确定时长，要走不同的逻辑。
    # 确定时长是将编码后的x送入卷积网络之中，让它去预测每个每个x对应的时长是多少，然后把时长和w计算l1_loss。
    # 这里的l_length就是loss_length的意思。
    # 确定时长模型，就是把预测的logw和真实的logw_做一个差得到loss_length。
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.dp(x, x_mask, g=g)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 

    # 指导attn矩阵之后，可以对m_p和logs_p进行拓展。
    # m_p的形状是[b,feat_dim,T_t],logs_p的形状是[b,feat_dim,T_t],这是对齐之前的形状。
    # 拓展以后m_p和logs_p的形状是[b,feat_dim,T_s]。这时候均值和方差从T_t这个时间长度扩展到了T_s的时间长度，即频谱的长度。
    # 这时候长度一致了才能计算kl_loss，所以m_p最终会返回出来送入kl_loss。
    # z_slice, ids_slice和o = self.dec(z_slice, g=g)是从后验引变量得到波形，是解码器，由反卷积和一些残差模块构成的。
    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  # 额外的小功能，“变声器"。
  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

