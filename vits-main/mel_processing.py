import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

# 对数压缩是一种常用的音频处理技术，用于减小音频信号的动态范围。它通过对信号取对数来实现压缩。
# 这个函数接受三个参数：
# x：要进行压缩的信号。
# C：压缩因子。它用于缩小信号的动态范围。
# clip_val：信号的截断阈值。信号中小于此阈值的值会被截断为该阈值，以防在取对数时出现错误。
# 这个函数会返回对数压缩后的信号。
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 对数解压缩是将对数压缩后的信号还原为原来的信号的过程。
# 这个函数接受两个参数：
# x：要进行解压缩的信号。
# C：压缩因子。在压缩时使用的因子。
# 这个函数会返回解压缩后的信号。
def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

# 归一化是指将信号的幅度调整为满足一定范围的过程。这通常用于将信号的幅度限制在 0 到 1 之间，以便于模型的训练和评估。
# 这个函数接受一个参数：
# magnitudes：信号的幅度。
# 这个函数会使用 dynamic_range_compression_torch 函数进行对数压缩，并返回压缩后的信号。
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# 这是一个将归一化的信号的幅度进行还原的函数。
# 这个函数接受一个参数：
# magnitudes：归一化的信号的幅度。
# 这个函数会使用 dynamic_range_decompression_torch 函数进行对数解压缩，并返回解压缩后的信号。
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

# 这是一个用于计算频谱图的函数。频谱图是音频信号的频率范围内的能量分布图。
# 这个函数接受六个参数：
# y：要计算频谱图的信号。
# n_fft：FFT 窗口大小。它是频谱图的分辨率。
# sampling_rate：信号的采样率。
# hop_size：FFT 窗口移动的步长。
# win_size：分析窗的大小。
# center：是否对信号进行居中。
# 这个函数使用 PyTorch 中的 STFT 函数来计算信号的频谱图。这个函数返回频谱图。
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

# 这是一个将频谱图转换为梅尔谱图的函数。梅尔谱图是音频信号的频率范围内的能量分布图，但是它使用人类耳朵听觉特性为基础进行编码。
# 这个函数接受六个参数：
# spec：要转换的频谱图。
# n_fft：FFT 窗口大小。它是频谱图的分辨率。
# num_mels：梅尔谱图的梅尔带数。它是梅尔谱图的分辨率。
# sampling_rate：信号的采样率。
# fmin：梅尔谱图的最小频率。
# fmax：梅尔谱图的最大频率。
# 这个函数使用 Librosa 库中的函数计算频谱图的梅尔带权矩阵。然后使用这个矩阵。
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec

# 这是一个计算音频信号的梅尔谱图的函数。它使用了前面提到的函数 spectrogram_torch() 和 spec_to_mel_torch()。
# 这个函数接受九个参数：
# y：要计算梅尔谱图的信号。
# n_fft：FFT 窗口大小。
# num_mels：梅尔谱图的梅尔带数。
# sampling_rate：信号的采样率。
# hop_size：FFT 窗口的移动步长。
# win_size：FFT 窗口大小。
# fmin：梅尔谱图的最小频率。
# fmax：梅尔谱图的最大频率。
# center：是否将信号居中。
# 该函数首先调用 spectrogram_torch() 函数计算频谱图，然后调用 spec_to_mel_torch()
def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec
