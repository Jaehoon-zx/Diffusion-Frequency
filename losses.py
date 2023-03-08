# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""
import torch
import torch.fft
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
import precomp
import scipy
import torch_dct as dct

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]

    # batch = block_dct(batch, block_size=32)
    # batch = dct.dct_2d(batch, norm='ortho')

    # batch = dct.dct_2d(batch, norm='ortho') # (B, C, 24, 24)
    # batch = torch.fft.fft2(batch, norm='ortho')
    # batch = torch.fft.fftshift(batch)

    perturbed_data = noise + batch

    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]

    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  def loss_fn_freq_og(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    
    fft_batch = torch.fft.fftshift(torch.fft.fftn(batch, dim = (-1, -2), norm='ortho'))
    fft_batch = torch.cat((fft_batch.real, fft_batch.imag), dim = 1)
    fft_noise = torch.randn_like(fft_batch) * sigmas[:, None, None, None]

    perturbed_data = fft_noise + fft_batch
    score = model_fn(perturbed_data, labels)
    target = -fft_noise / (sigmas ** 2)[:, None, None, None]
    
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  def loss_fn_freq(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]

    S = batch.shape[-1]
    fft_batch = torch.fft.fftshift(torch.fft.fftn(batch, dim = (-1, -2), norm='ortho'))
    fft_batch = torch.cat((fft_batch.real, fft_batch.imag), dim = 1) # (B, 6, 28, 28)
    # fft_batch /= (precomp.mnist_fft_mean_24_rgb + 1e-10).to(batch.device)
    # fft_batch /= 10
    # fft_batch = torch.linalg.pinv(fft_batch)

    fft_noise = torch.randn_like(fft_batch) * sigmas[:, None, None, None]
    sp_noise = torch.randn_like(batch) * sigmas[:, None, None, None]

    perturbed_data_fft = fft_noise + fft_batch
    perturbed_data_sp = sp_noise + batch
    perturbed_data = torch.cat((perturbed_data_sp, perturbed_data_fft), dim =1)

    noise = torch.cat((sp_noise, fft_noise), dim =1)
    score = model_fn(perturbed_data, labels)
    sp_score, fft_score_real, fft_score_imag = torch.chunk(score, 3, dim = 1)
    fft_score = torch.cat((fft_score_real, fft_score_imag), dim =1)

    sp_target = - sp_noise / (sigmas ** 2)[:, None, None, None]
    fft_target = - fft_noise / (sigmas ** 2)[:, None, None, None]
    
    losses = torch.square(torch.cat((0*(sp_score - sp_target), (fft_score - fft_target)), dim = 1))
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)

    # batch *= (precomp.mnist_fft_mean_24_rgb[0,0:3] + 1e-10).to(batch.device)

    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  def loss_fn_freq(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    
    fft_batch = torch.fft.fftshift(torch.fft.fftn(batch, dim = (-1, -2), norm='ortho'))
    fft_batch = torch.cat((fft_batch.real, fft_batch.imag), dim = 1) # (B, 2, 28, 28)
    fft_noise = torch.randn_like(fft_batch)

    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * fft_batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * fft_noise

    noise = torch.randn_like(batch)
    perturbed_data_spatial = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                              sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    perturbed_data = torch.cat((perturbed_data_spatial ,perturbed_data), dim = 1) # stack real, imag parts
    
    score = model_fn(perturbed_data, labels)
    sp_score, fft_score_real, fft_score_imag = torch.chunk(score, 3, dim = 1)
    fft_score = torch.cat((fft_score_real, fft_score_imag), dim =1)
    
    losses = torch.square(torch.cat((sp_score - noise, (fft_score - fft_noise)), dim = 1))

    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn

def block_multiplier(batch, block_size = 4, k = 1):
    result = torch.ones_like(batch)
    N, M = batch.shape[-2], batch.shape[-1] # (B, C, 24, 24)
    for i in range(N):
        for j in range(M):
            ref_x = i - i % block_size
            ref_y = j - j % block_size
            result[:,:,i,j] = np.exp(k* ((i-ref_x)**2 + (j-ref_y)**2)**(1/10))
    return result

def block_dct(tensor, block_size = 4):
    B, C, N, M = tensor.shape # (B, C, 24, 24)
    xs = list(torch.chunk(tensor, N//block_size, -2))
    for i in range(len(xs)):
        xys = list(torch.chunk(xs[i], M//block_size, -1))
        for j in range(len(xys)):
            xys[j] = dct.dct_2d(xys[j], norm='ortho')
        xs[i] = torch.cat(xys, -1)
    block_dct_tensor = torch.cat(xs, -2)

    return block_dct_tensor

def block_idct(tensor, block_size = 4):
    B, C, N, M = tensor.shape # (B, C, 24, 24)
    xs = list(torch.chunk(tensor, N//block_size, -2))
    for i in range(len(xs)):
        xys = list(torch.chunk(xs[i], M//block_size, -1))
        for j in range(len(xys)):
            xys[j] = dct.idct_2d(xys[j], norm='ortho')
        xs[i] = torch.cat(xys, -1)
    block_dct_tensor = torch.cat(xs, -2)

    return block_dct_tensor



def log_tanh(tensor, lamb = 1):

    return torch.log(torch.tanh(tensor/ lamb) + 1)

def log_elu(tensor):

    return torch.log( tensor + 1)

def scaler(batch, s = 1):
  result = torch.ones_like(batch) # (B, 6, 32, 32)
  N, M = batch.shape[2], batch.shape[3]
  for i in range(N):
    for j in range(M):
      if N//2 -s <= i <= N//2 + s and M//2 - s <= j <= M//2 +s:
        result[:,:,i,j] = 0.1
  return result

def multiplier(batch, k = 0.1):
  result = torch.ones_like(batch) # (B, 6, 32, 32)
  N, M = batch.shape[2], batch.shape[3]
  for i in range(N):
    for j in range(M):
      result[:,:,i,j] = np.exp(k* np.sqrt((i-N//2)**2 + (j-M//2)**2))
  return result

def freq_mask(batch, l1=0.1, l2=0.01, r1=0.4, r2=0.7):
    result = torch.ones_like(batch) # (B, 6, 32, 32)
    H, W = batch.shape[2], batch.shape[3]
    R = H * np.sqrt(2)

    for i in range(H):
        for j in range(W):
            if np.sqrt(i**2 + j**2) > r2 * R:
                result[:,:,i,j] = l2
            elif np.sqrt(i**2 + j**2) > r1 * R:
                result[:,:,i,j] = l1
                
    return result


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']

    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
