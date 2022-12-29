import torch 
from torch.nn import functional as F

import commons

# feature_loss：计算两个给定的特征图的差的平均值的绝对值的和，最后将结果乘 2。这个函数可能用于计算生成的图像与真实图像之间的差异，从而判断生成的图像的质量。
def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 

# discriminator_loss：计算判别器对真实图像和生成图像的输出的损失。对于真实图像，判别器应该输出 1，因此损失是（1 - 判别器输出）^ 2。对于生成图像，判别器应该输出 0，因此损失是 判别器输出 ^ 2。总损失是两种损失的和。
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses

# generator_loss：计算生成器对判别器的输出的损失。生成器希望判别器输出 1，因此损失是（1 - 判别器输出）^ 2。
def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


# VITS中kl散度计算公式代码
# kl_loss：计算两个给定的分布之间的 Kullback-Leibler 散度。Kullback-Leibler 散度是衡量两个分布之间差异的度量，常用于衡量两个概率分布之间的差异。
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

