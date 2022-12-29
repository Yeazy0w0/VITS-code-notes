import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

# 这是单speaker版本，具体分析往下参考多speaker版本
class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


"""Multi speaker version""" # 这是多speaker版本
class TextAudioSpeakerLoader(torch.utils.data.Dataset):  #这个class继承自叫torch.utils.data.Dataset的class，其实就是一个dataset的类
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams): #init中定义了一些音频的属性、文本要做的过滤以及数据文件的位置
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text) #这个文件其实是一个字符串描述的，用一个文本文件，每一行写了音频的路径、说话人的id以及文本的字符等等，每一行实际反映了一个样本。通过load函数加载filelists中的audiopaths_sid_text文件（竖线分割，左边是音频的路径，中间是说话人的id，右边是文本）进行解析。
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self): #filter方法对一些太短的文本做一些过滤。
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text): #这个函数接受audiopath_sid_text这个列表作为输入进行解析，将数据从原始格式转换成计算机可以读懂的格式。
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text) #对text离散化，把text的字符变成一个个的索引
        spec, wav = self.get_audio(audiopath) #对音频进行load，把它读取进来，得到采样点、提取频谱、线性谱。
        sid = self.get_sid(sid) #将speaker的id转化成整型。
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index): #dataset被dataloader调用的时候会走getitem这个函数，getitem通过调用get_audio_text_speaker_pair来实现的。
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):  #返回整个数据集有多少个，返回self.audiopaths_sid_text的长度（filelist文件的行数）
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate(): #主要是init和call两个方法
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False): # init现在传的是return_ids这个参数
        self.return_ids = return_ids

    def __call__(self, batch): #call这个函数接收的是batch，也就是一个mini batch
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort( #第一步是做一个排序
            torch.LongTensor([x[1].size(1) for x in batch]), # 对每个batch里面的x[1]即第一个元素（频谱）的帧数（即size(1)）放到列表中变成一个LongTensor这样一个张量，对这样一个张量进行降序排序
            dim=0, descending=True) 
        # 得到一个ids_sorted_decreasing，表示batch里面按照频谱的长度进行降序排序过后的索引

        # 找文本的最大长度和频谱的最大长度以及音频的最大长度
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        # x[3]是speaker_id，不需要去填充，每个样本就是一个标量，一个mini batch里面speaker_id大小是一样的。

        # 这里定义三个初始量，是每个mini batch里面文本、频谱以及音频各自的长度是多少（真实的长度，是pad之前的长度）
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        
        #构建一个新的空的量sid，即speaker_id，个数也是len(batch)个。
        sid = torch.LongTensor(len(batch)) 

        # 进行一个填充，下面三个量是初始化的张量，表示填充后的文本、填充后的频谱和填充后的波形
        text_padded = torch.LongTensor(len(batch), max_text_len) # 文本都是以索引的形式存储的，它的类型是long，大小是batch的size乘以max_text_len
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len) # 频谱是浮点型的，所以是FloatTensor，大小是batch的size乘以batch[0][1].size(0)（即特征维度）再乘以max_spec_len（即时间维度）
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len) # 大小是batch的size乘以1（因为wav的每一个采样点是标量，所以是1）再乘以max_wav_len（即时间维度）
        #用0初始化一下
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        # 往3个张量里填东西
        for i in range(len(ids_sorted_decreasing)): # 对刚刚降序好的索引进行一个遍历，i是对索引进行遍历的一个索引
            row = batch[ids_sorted_decreasing[i]] # 取到当前这个batch的第i个样本，得到每个row
            # 对文本进行填充
            text = row[0]
            text_padded[i, :text.size(0)] = text # 把text的有效位赋值给前面的text.size(0)这么多列，后面的都变成了0，不需要管它。
            text_lengths[i] = text.size(0) # text_lengths表示文本的真实长度，对它进行赋值，在pad的时候还是要记录一下它填充前的真实长度
            # 同理频谱也是这样填充
            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec # 把有效的spec填充进去
            spec_lengths[i] = spec.size(1) # 把spec填充前的长度记录下来
            # wav同理
            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav # 把有效的wav填充进去
            wav_lengths[i] = wav.size(1) # 把wav填充前的长度记录下来

            sid[i] = row[3] #是个标量，什么也不用处理

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing 
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid
        # 返回7个量：填充后的文本、填充前的文本的真实长度、填充后的一个batch的频谱、填充前这个batch内每个频谱的真实长度、填充后的音频、填充前音频的真实长度以及说话人的id
        # collate函数是对一个mini batch进行操作，会在dataload的时候去调用


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler): #继承自torch.utils.data.distributed.DistributedSampler。
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    # boundaries按照边界[32,…,1000]划分桶，num_replicas是卡的个数，下面都是一些参数
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # 定义成员变量
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets() # 创建一个桶，返回每个桶和每个桶里的样本总数
        self.total_size = sum(self.num_samples_per_bucket) #把每一个桶里的元素的个数加起来
        self.num_samples = self.total_size // self.num_replicas #总的样本数除以GPU卡数，得到num_samples，即每个卡在每个epoch所能见到的样本个数。
  
    def _create_buckets(self): 
        buckets = [[] for _ in range(len(self.boundaries) - 1)] # 定义一个二维的列表buckets，大列表中有几个小列表定义的是boundaries的个数 - 1个小列表（也就是说有这么多个桶）
        for i in range(len(self.lengths)): # 这里对self.lengths进行len就是计算一个总的数据量。i对总的数据进行一个遍历，得到每个数据的长度，再将这个长度划分到它该属于哪个桶。
            length = self.lengths[i] # 对self.lengths进行遍历，这里的lengths是dataset中频谱的长度。
            idx_bucket = self._bisect(length) # bisect使用二分法进行划分
            if idx_bucket != -1:
                buckets[idx_bucket].append(i) #index不等于-1就把样本的索引放到第index桶里面去。
                # 这样就把所有的数据按照它自己的长度划分到了对应的桶里面去了。
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0: 
                buckets.pop(i)
                self.boundaries.pop(i+1)
                # 对空桶进行检查。找哪个桶是空的，如果是空桶，就把对应的边界给踢掉。
  
        num_samples_per_bucket = []
        for i in range(len(buckets)): # 对桶进行遍历
            len_bucket = len(buckets[i]) # 得到每一个桶有多少样本
            total_batch_size = self.num_replicas * self.batch_size # 算一个总的batch_size，因为可能会有多卡，比如num_replicas如果是2的话，就会有两倍batch_size，默认是1的话，total_batch_size = batch_size。
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size # 保证每个桶里的样本数量能够被total_batch_size所整除，如果不能整除，这个rem余数不等于0，就会在后面尽量补上。
            num_samples_per_bucket.append(len_bucket + rem) # len_bucket + rem补上，使得每个桶里的元素数目刚好是total_batch_size的整数倍。
        return buckets, num_samples_per_bucket
        # 找到每一个桶到底该有多少个样本
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator() # 定义了一个生成器，规定以什么样的随机顺序来去取样本
      g.manual_seed(self.epoch) # 手动设置种子，把self.epoch设置成Generator的种子。这里的epoch是从父类DistributedSampler继承过来的，父类中会有一个setepoch来去设置每次训练的时候大概是第几个周期。
      #保证训练的时候，每个周期样本的随机性是不一样，并且这个不一样是确定性的不一样（种子是确定的，将epoch的值作为它的种子）。
  
      indices = [] # 声明了一个indices空列表，如果我们指定了shuffle的人数的话，就会对桶进行遍历。
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist()) # 每个桶储存了数据的索引，并且每个桶都有长度，于是用len(bucket)代表它的长度，randperm将长度随机组合，随机性由generator所构成。
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
        # indices储存了每个桶的数据被打乱后的索引。
  
      batches = [] # 定义了一个空列表batches，作为iter方法的生成器
      for i in range(len(self.buckets)): # 对桶进行遍历，i是桶的索引
          bucket = self.buckets[i] # 取出第i个bucket
          len_bucket = len(bucket) # 去找到这个bucket它里面有多少样本
          ids_bucket = indices[i] # 把bucket里面所存储的样本的索引找出来，放到ids_bucket，这就是所有的样本的索引。
          num_samples_bucket = self.num_samples_per_bucket[i] #num_samples_bucket把桶里面有多少个样本也找出来。
          # num_samples_bucket不等于len_bucket，因为num_samples_bucket已经做过余数补偿了，大于等于len_bucket
          # 如果如果桶的元素数刚好是total_batch_size的整数倍的话，那么num_samples_bucket等于len_bucket，否则num_samples_bucket大于len_bucket
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket #计算差多少个余数
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
          # 知道桶里面缺多少个样本，于是就把样本索引ids_bucket扩充一下，加一些新的bucket，得到完整的ids_bucket
          #ids_bucket等于num_samples_bucket说明已经做了补偿，它刚好能被total_batch_size整除了。
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
          # 现在是支持多GPU训练的，于是每个GPU隔num_replicas个取一次桶里面属于这个GPU的样本索引。rank代表第几个GPU，num_replicas代表GPU的总数。如果只有一个GPU，那就是一个一个取。
  
          # batching 组batch
          for j in range(len(ids_bucket) // self.batch_size): #j代表第几个batch，根据bucket里的样本数目除以batch_size，得到mini batch的个数，每个mini batch都会得到一个batch。
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]] #batch是从bucket里面取的，取的索引是从ids_bucket里面拿的，每次拿j*self.batch_size到(j+1)*self.batch_size
              batches.append(batch) # 这样获得一个batch的样本，将它的样本的索引放到batches这个大的数组里面
              # 这样就完成了batches的组合，batches每个列表都表示都表示mini batche的样本的索引。
              # 刚好这里做了桶排序，所以每个mini batche里面样本的长度都是非常接近的。
  
      if self.shuffle: # 对batch进行shuffle
          batch_ids = torch.randperm(len(batches), generator=g).tolist() # 刚刚组了很多个mini batche的列表，接下来还可以对这个列表进行shuffle，这里的shuffle仍然是确定性的。
          batches = [batches[i] for i in batch_ids] # shuffle完之后，再重新把batch_ids组合成新的batches
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples # 新的batches乘以batch_size一定等于num_samples，num_samples是每个卡能见到的样本数目（在每个周期里面）
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
