import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

# 这是一个定义了一个函数 load_checkpoint 的代码段。
# 这个函数的作用是加载模型的保存的训练状态。它的输入有 checkpoint_path、model 和可选的 optimizer，checkpoint_path 是保存训练状态的文件路径，model 是保存状态的模型，optimizer 是优化器。
# 它的输出有 model、optimizer、learning_rate 和 iteration，model 和 optimizer 分别表示加载状态后的模型和优化器，learning_rate 和 iteration 分别表示保存的学习率和训练的迭代次数。
# 这个函数首先会加载 checkpoint_path 文件中保存的状态字典，然后将模型和优化器的状态更新为保存的状态。最后，它会输出加载的信息。
def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration

# save_checkpoint 函数的作用是保存模型和优化器的训练状态。
# 它的输入有 model、optimizer、learning_rate、iteration 和 checkpoint_path。model 和 optimizer 分别表示要保存状态的模型和优化器，learning_rate 和 iteration 分别表示当前的学习率和训练的迭代次数，checkpoint_path 是保存状态的文件路径。
# 这个函数会将 model 和 optimizer 的状态保存在 checkpoint_path 文件中。
def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

# 这是一个一个用于汇总数据以供可视化和分析的函数。该
# 函数接受一个 writer 对象，一个 global_step 整数，以及几个数据字典：scalars，histograms，images 和 audios。scalars 字典包含可以表示为单个标量值的数据，histograms 字典包含可以表示为直方图的数据，images 字典包含可以表示为图像的数据，audios 字典包含可以表示为音频的数据。
# 然后，函数遍历每个字典，并在字典中的每个项目上调用 writer 对象的相应方法，将键和值作为参数传递。例如，对于 scalars 字典中的每个项目，函数调用 writer.add_scalar(k, v, global_step)，其中 k 是键，v 是值。这将添加标量数据到 writer 对象，然后可以使用这些数据以某种方式可视化数据。
# audio_sampling_rate 参数指定添加到 writer 中的音频数据的采样率（以赫兹为单位）。默认值为 22050 Hz，这是常用的音频数据采样率。
def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)

# 找到最新的checkpoint
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

# 这是一个用来将频谱图转化为numpy数组的函数，使用 matplotlib 库将频谱图绘制出来，并将图像转化为RGB数组。
# 频谱图是一种图像，用来表示声音或其他信号的频率分布，通常纵坐标表示频率，横坐标表示时间。
def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

# 创建对齐矩阵图并将其作为 NumPy 数组返回的实用函数。函数接受对齐矩阵和可选的 info 字符串作为输入参数。
# 该函数首先导入必要的模块，包括 matplotlib 和 numpy。然后使用 matplotlib 的 subplots 函数创建一个图形和一个轴。它使用 imshow 函数在轴上绘制对齐矩阵的转置，并向图形添加一个颜色条。然后使用 plt.xlabel 和 plt.ylabel 设置 x 轴标签和 y 轴标签。使用 plt.tight_layout 调整布局，使用 fig.canvas.draw 绘制画布。
# 最后，该函数使用 np.fromstring 和 tostring_rgb 将画布转换为 NumPy 数组，使用图形的宽度和高度重塑数组，并返回结果数组。它还使用 plt.close 关闭图形。
def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data

# 加载 wav 格式的音频文件
def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

# 加载文件路径和文本内容
def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

# 从配置文件中读取超参数
def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams

# get_hparams_from_dir 函数用于从模型目录中读取 "config.json" 文件，并将其解析为 Python 字典。
# 然后使用这个字典创建一个 HParams 类的实例，并将模型目录作为属性保存在该实例中。最后返回这个 HParams 实例。
def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams

# get_hparams_from_file 函数与 get_hparams_from_dir 函数类似，只是它读取的是指定的配置文件路径，而不是模型目录中的 "config.json" 文件。
def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams

# check_git_hash 函数用于检查当前的 Git 仓库的哈希值是否与保存在模型目录中的 "githash" 文件中的哈希值一致。如果不一致，则会记录一条警告信息。
def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)

# get_logger 函数用于创建一个 logger 对象，并将其配置为将日志信息写入模型目录中的 "train.log" 文件。
def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger

# HParams 类是一个用于存储配置参数的类。
# 它的构造函数可以接受一系列关键字参数，并将这些参数存储在实例的属性中。
# 这个类还实现了一些 Python 内置函数的魔术方法，使其可以像字典一样使用。
class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
