import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0

#主函数
def main():
  """Assume Single Node Multi GPUs Training Only""" # 这里是多卡的部分
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)


  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data) #构造一个dataset，从名叫TextAudioSpeakerLoader的class（在data_utils中）实例化而来，这个class是读取文本、音频和speaker相关的，并且返回一个dataset。
  train_sampler = DistributedBucketSampler( #分布式的sampler，在data_utils中实现，TTS的任务数据长度变化可能很大(有些音频1秒，有些10秒)，通过一个桶排序对数据进行排序，这样每一个batch分到的样本长度变化范围没那么大，有效的数目尽可能地接近，提升训练效率。
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000], # 这个长度指的是频谱的单位数
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
      # bucket sampler 根据桶排序组batch，得到每个batch的样本的索引。

  collate_fn = TextAudioSpeakerCollate() # 主要是init和call两个方法
  # collate函数是对一个mini batch进行操作，会在dataload的时候去调用

  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler) #把train_dataset、collate_fn和train_sampler传进去
      # 组mini batch的时候是sampler来组的，同时每组完一个mini batch之后都会经过collate_fn进行后处理（pad，把一个mini batch里的文本、频谱和音频各自pad成相应的长度），这样得到train_loader
  if rank == 0: # 如果是在主机eval的话，要去做验证，做验证集的时候，在一个GPA上跑，包括记日志、log等，在主GPU上做
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  # rank不等于0的时候，只需要管训练

  # 定义生成器，SynthesizerTrn表示文本到音频的一整个模型
  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers, # 与单speaker的区别（2/2）
      **hps.model).cuda(rank) #net_g被传到了cuda上

  # 定义了一个多周期的判别器，是一个混合的判别器，同样会被传到cuda上。    
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

  # 因为这是一个GAN的训练任务，所以定义了两套优化器
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  # 由于这里是一个分布式的训练，所以用一个DDP把优化器包裹起来，包裹之后“net_g.model"才是模型。
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  # 如果在训练的时候，已经有现成的模型的话，那这里就会load一下，继续训练。这里写得非常自动化，不需要去指定load第几个pytorch文件，它会自动寻找最近的是哪一个模型文件进行读取。
  # 这种方法有优点也有缺点，优点是什么都不需要干，坏处是它默认了只读最后一个。
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0
  # 这里定义的global_step是训练步数，是等于epoch乘以每个epoch里面batch_size的个数。

  # 这里定义了两个学习率的指数衰减的方案，分别是生成器的衰减方案和判别器的衰减方案，衰减方案是一样的。
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  # 因为用到了混合精度训练（AMP，自动的混合训练），训练的时候会用fp16去训练，这样可以在效率和性能上取得一个平衡。
  # 这里实例化了一个GradScaler的API，这个API是torch.cuda.amp里面的，用fp16训练的API，得到一个scaler
  scaler = GradScaler(enabled=hps.train.fp16_run)

  # 对epoch进行循环，每个epoch在里面做train_and_evaluate，如果是在主GPU上，logger, writer还有验证集传上去，如果不是主GPU，则只要负责训练就好了。
  # 完成以后，还要对刚刚定义的学习率的方案进行step，更新一下学习率。
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


# 每个周期里面都会运行train_and_evaluate函数，这个函数里面还有一层for循环，为了对data_loader进行遍历
def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  # 这里会把train_loader.batch_sampler设置epoch，控制桶排序的随机性。
  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  # net_g和net_d变成train的模式，这种模式会记录梯度，在抓包等的时候祈祷相应的作用。
  net_g.train()
  net_d.train()
  # 对train_loader进行枚举，每一个train_loader都会返回7个东西，分别是：文本、文本长度、频谱、频谱长度、音频、音频长度以及speaker的id。
  # 然后分别将这7个量分别拷贝到，cuda（GPU）上面。
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True) # 与单speaker的区别（2/2）

    # autocast中的enabled等于true，意思是会用fp16的精度去做训练，去做它的前向运算和算梯度。
    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)
      # 将x等7个量送到生成器里面，生成器运行一遍，得到预测的波形和长度。
      # 这个训练是采样式的训练，不是将所有的频谱送入解码器中得到波形，而是从里面采样一小段频谱来生成波形，这样训练所耗的内存会变小一些。
      # ids_slice就是采样后频谱的id

      # 把频谱（线性谱）转成梅尔谱，因为后面的一个重构loss需要梅尔谱，这里将线性谱转化成梅尔谱，作为重构loss的标签。
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)

      # 这里slice_segments会取一部分梅尔谱，y_mel是真实的梅尔谱。
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      # y_hat_mel是预测的梅尔谱。预测的梅尔谱只能从预测的音频波形中得到，从波形得到梅尔谱就需要调用mel_spectrogram_torch函数去计算梅尔频谱，得到y_hat_mel
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      # 接下来要拿到真实的波形，因为判别器做判别的时候，需要以波形作为输入。刚刚生成的y_hat，不是原来的一整段音频，是从采样后的频谱里生成的，只是一小段音频。要得到y_hat的标签，也要从真实的y里面去取一小段音频。
      # 这里梅尔谱的ids_slice * hps.data.hop_length就是说：一个梅尔谱可能对应的256个波形点，所以hop_length就是256。这样就可以取到采样后的梅尔谱所对应的真实的音频y
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # 这里将y（真实的波形）和预测的波形都送入到判别器中，就会得到真实的判别器的输出和生成的判别器的输出。
      # 然后会在autocast(enabled=False)里面计算它的loss，这部分是不走fp16的。
      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g) #将判别器的真实的输出和预测的输出送入discriminator_loss中，分别得到判别器的总的损失loss_disc
        loss_disc_all = loss_disc
    # 更新判别器
    optim_d.zero_grad() # 对判别器的梯度置零
    scaler.scale(loss_disc_all).backward() # 把判别器的损失loss_disc_all送入scaler.scale()中进行backward()，这样可以计算判别器的每个参数的梯度
    scaler.unscale_(optim_d) # 对判别器的模型用scaler.unscale_还原一下，再截取梯度
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None) # clip_grad_value对梯度的值进行截断
    scaler.step(optim_d) # 对判别器的参数进行更新
  # 以上是混合精度训练的代码，判别器的更新

  # 生成器部分
    with autocast(enabled=hps.train.fp16_run): # 依旧在用fp16的精度去做训练
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat) # 算出一个对抗的loss（从判别器过来的），将真实的波形和预测的波形送入到discriminator中，得到中间特征的输出。
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float()) # 时长loss
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel # 梅尔谱重构loss，对真实的梅尔谱和预测的梅尔谱做一个l1_loss。其中系数c_mel等于45（见论文）。
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl # 由文本的先验编码器所预测的复杂分布和频谱经过的后验编码器所得到的高斯分布两个之间计算kl散度，系数c_kl等于1（论文）。
        loss_fm = feature_loss(fmap_r, fmap_g) # 将真实波形和预测波形同时送入到判别器之中，去看判别器的中层值特征是否相近。
        loss_gen, losses_gen = generator_loss(y_d_hat_g) # 对抗loss
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl # 得到生成器的总loss
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()
    #以上6行是对生成器进行更新

    # 在主GPU上对loss的值进行打印，保存相应的模型，并且写入到日志文件之中，对global_step进行更新。如果global_step等于eval_interval的整数倍，就会做一个evaluate验证
    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
