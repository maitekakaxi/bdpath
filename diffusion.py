import itertools
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict
import torch.nn as nn
import dataloader
import metrics
import models
import noise_schedule
import utils
import numpy as np
import itertools
from dataloader import PathTimeTokenizer,PathTimeDataset
def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))

@dataclass
class Loss:
  loss: torch.FloatTensor           # 总损失
  path_loss: torch.FloatTensor      # 路径损失（可单独 log）
  time_loss: torch.FloatTensor      # 时间损失（可单独 log）
  nlls: torch.FloatTensor           # 每个 token 的路径 loss（例如 NLL）
  token_mask: torch.FloatTensor     # attention mask


# class Loss:
#   loss: torch.FloatTensor
#   nlls: torch.FloatTensor
#   token_mask: torch.FloatTensor

        
        


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = None
    self.max_path_id = config.model.max_path_id
    self.vocab_size = self.config.data.vocab_size
    self.time_vocab_size = config.model.time_vocab_size
    self.time_embedding = nn.Embedding(self.time_vocab_size, config.model.hidden_size)    #
    self.path_head = nn.Linear(config.model.hidden_size, self.vocab_size)
    self.time_head = nn.Linear(config.model.hidden_size, config.model.time_vocab_size)
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    self.mdlm_loss_scale = self.config.algo.mdlm_loss_scale
    self.Time_tokenizer = PathTimeTokenizer
    self.time_weight = 1.0
    trajs_csv = "/root/workspace/MTNet/data/demo/trajs_demo.csv"
    times_csv = "/root/workspace/MTNet/data/demo/tstamps_demo.csv"
    self.dataset = PathTimeDataset(
        trajs_csv, times_csv, tokenizer,
        block_size=config.block_size
    )

    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = 0
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.parameterization == 'ar':
      self.block_size = 1
    if self.config.algo.backbone == 'dit' or self.config.algo.backbone =='path_time':
      self.backbone = models.dit.DIT(
                                self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
      #  egenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.DummyMetrics(config)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {
      'dit', 'ar', 'hf_dit','path_time'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
      
    if self.parameterization in {'sedd'}:
      assert self.time_conditioning
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    if self.parameterization == 'ar':
      return None
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma


  def forward(self, input_ids,input_time, sigma, sample_mode=False, store_kv=False):
      sigma = self._process_sigma(sigma)

      with torch.amp.autocast('cuda', dtype=torch.float32):
        out_dict = self.backbone(
            indices=input_ids,
            input_time=input_time,
            sigma=sigma,
            sample_mode=sample_mode,
            store_kv=store_kv
        )

      if self.cross_attn:
          input_ids = input_ids[:, :self.config.model.length]  # restore xt
          
      logits_path = self._subs_parameterization(logits=out_dict["path_logits"], xt=input_ids)
      time_pred = out_dict["time_pred"]
      return {
          "path_logits": logits_path,
          "time_pred": time_pred
      }
    
    
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  # def training_step(self, batch, batch_idx):
  #   del batch_idx
  #   losses = self._loss(batch['input_ids'],
  #                       batch['attention_mask'])
  #   self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
  #   self.log(name='trainer/loss',
  #            value=losses.loss.item(),
  #            on_step=True,
  #            on_epoch=False,
  #            sync_dist=True)
  #   return losses.loss
  def training_step(self, batch, batch_idx):
    losses = self._loss(batch['path_ids'], batch['time_values'], batch['attn_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log('trainer/loss', losses.loss.item(), on_step=True, on_epoch=False, sync_dist=True)
    return losses.loss




  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    if hasattr(self.metrics.valid_nlls, "compute"):
        computed_metrics = self.metrics.valid_nlls.compute()
        
        # 明确 log 出 val/nll，供 ModelCheckpoint 使用！
        if "val/nll" in computed_metrics:
            self.log("val/nll", computed_metrics["val/nll"], on_step=False, on_epoch=True, sync_dist=True)

        # 其他指标也顺便 log 一下
        for k, v in computed_metrics.items():
            if k != "val/nll":  # 避免重复 log
                self.log(name=k, value=v, on_step=False, on_epoch=True, sync_dist=True)

    if self.ema:
        self.ema.restore(self._get_parameters())

    if self.var_min and not self.trainer.sanity_checking:
        self.log("sampling_eps_min", self.sampling_eps_min,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log("sampling_eps_max", self.sampling_eps_max,
                 on_epoch=True, on_step=False, sync_dist=True)

#     for k, v in self.metrics.valid_nlls.items():
#       self.log(name=k,  value=v.compute(), on_step=False,
#               on_epoch=True, sync_dist=True)
#     if self.ema:
#       self.ema.restore(self._get_parameters())
#     if self.var_min and not self.trainer.sanity_checking:
# #      self._clipped_schedule_search()
#       self.log('sampling_eps_min',
#                self.sampling_eps_min,
#                on_epoch=True,
#                on_step=False,
#                sync_dist=True)
#       self.log('sampling_eps_max',
#                self.sampling_eps_max,
#                on_epoch=True,
#                on_step=False,
#                sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    

    losses = self._loss(batch['path_ids'],
                          batch['time_values'],
                          batch['attn_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]
  
  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    assert self.training
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size

  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)
    #xt = xt.reshape(xt.shape[0], -1)
    return xt
  # def _sample_prior(self, *batch_dims):
  #   return self.mask_index * torch.ones(
  #     * batch_dims, dtype=torch.int64, device=self.device)
  def _sample_prior(self, n_samples, seqlen):
      x_path = torch.full((n_samples, seqlen), self.mask_index,
                          device=self.device, dtype=torch.long)
      x_time = torch.randn((n_samples, seqlen), device=self.device)
      return x_path, x_time

  # def _sample_prior(self, n_samples, seqlen):
  #     dev = self.device
  #     x_path = torch.full((n_samples, seqlen),
  #                         self.mask_index, device=dev, dtype=torch.long)
  #     #x_path[:, 0] = self.Time_tokenizer.bos_token_id

  #     x_time = torch.randint(0, 256,
  #                           (n_samples, seqlen),
  #                           device=dev) + 8263   # ← 加偏移
  #     #x_time[:, 0] = self.Time_tokenizer.time_offset          # BOS for time
  #     return x_path, x_time




  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  # @torch.no_grad()
  # def _ddpm_caching_update(self, x, t, dt, p_x0=None):
  #   _, move_chance_t = self.noise(t)
  #   _, move_chance_s = self.noise(t - dt)
  #   sigma_t = self._sigma_from_p(move_chance_t)
  #   move_chance_t = move_chance_t[:, None]
  #   move_chance_s = move_chance_s[:, None]
  #   mask_prob = move_chance_s / move_chance_t

  #   if p_x0 is None:
  #     p_x0 = self.forward(x,
  #                       sigma_t,
  #                       sample_mode=True).to(torch.float64)
  #     p_x0 = p_x0.exp()
  #     p_x0 = self._nucleus_sample(p_x0)

  #   if self.config.sampling.first_hitting:
  #     x_block = _sample_categorical(p_x0[:, -self.block_size:])
  #     # randomly and uniformly select an index in the block (among masked tokens)
  #     num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
  #     ind = torch.randint(0, num_masked, (x_block.shape[0],))
  #     ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
  #     mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
  #     x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
  #   else:
  #     q_xs = p_x0 * (1 - mask_prob)
  #     q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
  #     x_block = _sample_categorical(q_xs[:, -self.block_size:])  
  #   copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
  #   x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
  #   x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

  #   # compute kv cache if all tokens in a block are sampled
  #   if self.config.sampling.kv_cache and self.mask_index not in x_block:
  #     _ = self.forward(x_new, sigma_t, sample_mode=True, store_kv=True)

  #   if not torch.allclose(x_new, x):
  #     tokens_sampled = (x_new - x) != 0
  #     return None, x_new
  #   else:
  #     return p_x0, x_new
  @torch.no_grad()
  def _ddpm_caching_update(self, x_path, x_time, t, dt, p_x0=None):
      """
      x_path: [B, L] masked path token
      x_time: [B, L] noisy continuous time
      t: [B] current timestep scalar
      dt: step size scalar
      p_x0: optional cached logits (path)
      """
      # 1. 获取当前 timestep 对应的移动概率（mask prob）
      _, move_chance_t = self.noise(t)
      _, move_chance_s = self.noise(t - dt)
      sigma_t = self._sigma_from_p(move_chance_t)  # [B]
      move_chance_t = move_chance_t[:, None]
      move_chance_s = move_chance_s[:, None]
      mask_prob = move_chance_s / move_chance_t   # [B, 1]

      # 2. 如果路径 logits 没缓存，就 forward 一次获取 logits + time_pred
      if p_x0 is None:
          out = self.forward(
              input_ids=x_path,
              input_time=x_time,
              sigma=sigma_t,
              sample_mode=True
          )
          p_x0 = out['path_logits'].to(torch.float64)  # [B, L, V]
          p_x0 = p_x0.exp()                            # 转为概率
          p_x0 = self._nucleus_sample(p_x0)            # 可能做 top-p 剪枝

      # 3. 路径采样：first hitting 或普通 categorical
      if self.config.sampling.first_hitting:
          x_block = _sample_categorical(p_x0[:, -self.block_size:])
          num_masked = (x_path[:, -self.block_size:] == self.mask_index).sum(-1)
          ind = torch.randint(0, num_masked, (x_block.shape[0],))
          ind = (x_path[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
          mask = (torch.arange(self.block_size, device=x_path.device) == ind[:, None]).to(x_block.dtype)
          x_block = x_block * mask + x_path[:, -self.block_size:] * (1 - mask)
      else:
          q_xs = p_x0 * (1 - mask_prob)
          q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
          x_block = _sample_categorical(q_xs[:, -self.block_size:])

      # 4. 合并非 mask token
      copy_flag = (x_path[:, -self.block_size:] != self.mask_index).to(x_block.dtype)
      x_block = copy_flag * x_path[:, -self.block_size:] + (1 - copy_flag) * x_block
      x_new_path = torch.cat((x_path[:, :-self.block_size], x_block), dim=-1)

      # 5. 时间采样：直接取网络回归结果（不做 sampling）
      out = self.forward(
          input_ids=x_new_path,
          input_time=x_time,
          sigma=sigma_t,
          sample_mode=True,
          store_kv=True if self.config.sampling.kv_cache else False
      )
      t_pred = out['time_pred']                     # [B, L]
      t_block = t_pred[:, -self.block_size:]        # 当前 block 的时间值
      x_new_time = torch.cat((x_time[:, :-self.block_size], t_block), dim=-1)
      if not torch.allclose(x_new_path, x_path):
          return None, x_new_path, x_new_time
      else:
          return p_x0, x_new_path, x_new_time








  @torch.no_grad()
  def _ar_sampler(self, bsz, context_len=1024):
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache()

    with torch.amp.autocast('cuda', dtype=torch.float32):
      # precompute token buffer
      num_pred_tokens = self.num_tokens - 1
      x = torch.zeros(
        (bsz, num_pred_tokens + 1),
        dtype=torch.long,
        device=self.device)
      x[:, 0] = self.tokenizer.bos_token_id
      stop = False
      for i in tqdm(range(num_pred_tokens)):
        # need to sample a gumbel for each token
        # to save memory in variable-length sampling
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, self.vocab_size))
                .to(self.device))
        next_logits = self.forward(
          x[:, :i + 1][:, -context_len:],
          None,
          store_kv=self.config.sampling.kv_cache)[:, -1:].to(torch.float64)
    
        next_logits = next_logits.exp()
        next_logits = self._nucleus_sample(next_logits).log()
        y = (next_logits[:, -1] + noise).argmax(-1)
        # check if we need to resample (or stop sampling for variable-length sampling)
        if (i+1) > 256:
          stop, x_out = self._check_stop_conds(x[:, :i+1])
          if stop:
            x = x_out
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None
        elif stop:
          break
        x[:, i + 1] = y
      return x
  

  # @torch.no_grad()
  # def _sample(
  #   self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
  #   """Generate samples from the model."""
  #   if seqlen is None:
  #     seqlen = self.config.model.length
  #   if batch_size_per_gpu is None:
  #     batch_size_per_gpu = self.config.loader.eval_batch_size
  #   samples = []
  #   if self.sampler == 'semi_ar':
  #     for _ in range(self.config.sampling.num_sample_batches):
  #       sample_i, num_tries = None, 0
  #       while sample_i is None:
  #         num_tries += 1
  #         sample_i, nfes = self._semi_ar_sampler(
  #           n_samples=batch_size_per_gpu,
  #           num_strides=(seqlen // self.block_size), 
  #           num_steps=num_steps,
  #           seqlen=seqlen)
  #         if num_tries > 10:
  #           raise ValueError('Sampling failed.')
  #       samples.append(sample_i)
  #       self.metrics.nfes.update(nfes)
  #       self.metrics.gen_nfes.append(nfes)
  #   samples = torch.cat(samples, dim=0) 
  #   return self.tokenizer.batch_decode(samples)


  @torch.no_grad()
  def _sample(
      self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None
  ):
      """Generate path + time samples from the model."""
      if seqlen is None:
          seqlen = self.config.model.length
      if batch_size_per_gpu is None:
          batch_size_per_gpu = self.config.loader.eval_batch_size
      #if num_steps is None:
          #num_steps = self.config.sampling.num_steps
      all_paths, all_times = [], []
      time_mean, time_std = self.dataset.get_time_stats()
      if self.sampler == 'semi_ar':
          for _ in range(self.config.sampling.num_sample_batches):
              sample_path, sample_time, num_tries = None, None, 0
              while sample_path is None:
                  num_tries += 1
                  sample_path, sample_time, nfes = self._semi_ar_sampler(
                      n_samples=batch_size_per_gpu,
                      num_strides=(seqlen // self.block_size),
                      num_steps=num_steps,
                      seqlen=seqlen,
                  )
                  if num_tries > 10:
                      raise ValueError("Sampling failed.")
              self.metrics.nfes.update(nfes)
              self.metrics.gen_nfes.append(nfes)
              sample_time = decode_log_time_samples(sample_time, time_mean, time_std)
              #sample_time = sample_time.tolist()
              all_paths.append(sample_path)
              all_times.append(sample_time)
      all_paths = torch.cat(all_paths, dim=0).cpu()
      all_times = torch.cat(all_times, dim=0).cpu()
      decoded_paths = []
      decoded_times = []
      for path, t in zip(all_paths.tolist(), all_times.tolist()):
          if 0 in path:
              cutoff = path.index(0)
              path = path[:cutoff]
              t = t[:cutoff]
          roads = [r for r in path]
          times = [x * time_std + time_mean for x in t]
          decoded_paths.append(roads)
          decoded_times.append(times)
      return decoded_paths, decoded_times

  # @torch.no_grad()
  # def _sample(
  #     self,
  #     seqlen: int | None = None,
  #     num_steps: int | None = None,
  #     eps: float = 1e-5,
  #     batch_size_per_gpu: int | None = None,
  # ):
  #     """
  #     生成 (path_ids, time_ids)   形状均为 [B, T]
  #     - path_ids ∈ [PATH_OFFSET, PATH_OFFSET+PATH_VOCAB)
  #     - time_ids ∈ [TIME_OFFSET, TIME_OFFSET+TIME_VOCAB)
  #     """
  #     # ----------- 0. 通用参数 -------------
  #     seqlen = seqlen or self.config.model.length
  #     num_steps = num_steps or self.config.sampling.num_steps
  #     batch_size_per_gpu = batch_size_per_gpu or self.config.loader.eval_batch_size

  #     B, T   = batch_size_per_gpu, seqlen
  #     device = self.device

  #     # ----------- 1. 预留采样结果 -------------
  #     paths_all, times_all = [], []

  #     # ----------- 2. 三种采样分支 -------------
  #     if self.parameterization == "ar":                       # (1) Auto‑Regressive
  #         for _ in range(self.config.sampling.num_sample_batches):
  #             path_ids, time_ids, tries = None, None, 0
  #             while path_ids is None:
  #                 tries += 1
  #                 path_ids, time_ids = self._ar_sampler(B, T)       # <-- 返两份
  #                 if tries > 10:
  #                     raise RuntimeError("AR sampling failed.")
  #             paths_all.append(path_ids)
  #             times_all.append(time_ids)
  #             self.metrics.gen_nfes.append(T)                       # 保持原有统计

  #     elif self.sampler == "semi_ar":                          # (2) Semi‑AR (block-wise)
  #         for _ in range(self.config.sampling.num_sample_batches):
  #             path_ids, time_ids, tries = None, None, 0
  #             while path_ids is None:
  #                 tries += 1
  #                 path_ids, nfes = self._semi_ar_sampler(
  #                     n_samples=B,
  #                     num_strides=T // self.block_size,
  #                     num_steps=num_steps,
  #                     seqlen=T,
  #                 )
  #                 if tries > 10:
  #                     raise RuntimeError("semi‑AR sampling failed.")
  #             paths_all.append(path_ids)
  #             #times_all.append(time_ids)
  #             #self.metrics.nfes.update(nfes)
  #             #self.metrics.gen_nfes.append(nfes)

  #     else:                                                    # (3) Analytic / DDPM‑like
  #         nfes = num_steps
  #         for _ in range(self.config.sampling.num_sample_batches):
  #             path_ids, time_ids, tries = None, None, 0
  #             while path_ids is None:
  #                 tries += 1
  #                 path_ids, time_ids = self._analytic_sampler(
  #                     n_samples=B,
  #                     num_steps=num_steps,
  #                     seqlen=T,
  #                     eps=eps,
  #                 )
  #                 if tries > 10:
  #                     raise RuntimeError("Analytic sampling failed.")
  #             paths_all.append(path_ids)
  #             #times_all.append(time_ids)
  #             #self.metrics.nfes.update(nfes)
  #             #self.metrics.gen_nfes.append(nfes)

  #     # ----------- 3. 拼接结果 -------------
  #     paths_all = torch.cat(paths_all,  dim=0)      # [B_total, T]

  #     return paths_all
  
  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)


  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    paths,times = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self.metrics.record_generative_perplexity(
      paths,
      self.config.model.length,
      self.config.loader.eval_batch_size,
      self.device)
    return paths,times



  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma).to(torch.float64)
    if self.config.sampling.nucleus_p == 1.0:
      return model_output.exp()
    model_output = model_output - model_output.logsumexp(-1, keepdim=True)
    model_output = self._nucleus_sample(model_output.exp())
    return model_output

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_p(self.noise(t)[1])
    sigma_s = self._sigma_from_p(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self.get_score(x, sigma_t)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)


  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_p(self.noise(t)[1])
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples


  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
      offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
      offset_b = offset_b.view(batch_dims[0], num_blocks)
      _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0_path, time_values, t=None, sampling_eps_min=None, sampling_eps_max=None): #(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if t is None:
      t = self._sample_t(x0_path.shape,
                         x0_path.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))
    dsigma = - loss_scale * torch.expm1(sigma) 

    if self.mdlm_loss_scale:
      sigma, dsigma = self.noise.total_noise(t), self.noise.rate_noise(t)
      p = 1 - torch.exp(-sigma)
      loss_scale = - (dsigma / torch.expm1(sigma))
    noise = torch.randn_like(time_values)
    alpha_bar = torch.exp(-sigma).view(-1, 1).to(time_values.device)
    time_xt = alpha_bar.sqrt() * time_values + (1 - alpha_bar).sqrt() * noise
    xt = self.q_xt(x0_path,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
    if self.ignore_bos:
      xt[:, 0] = x0_path[:, 0]
    x_input = torch.cat((xt, x0_path), dim=-1)
    time_input = torch.cat((time_xt,time_values),dim=-1)
    model_output = self.forward(
        input_ids=x_input,
        sigma=sigma,
        input_time=time_input
    )
    utils.print_nans(model_output["path_logits"], 'path_logits')
    utils.print_nans(model_output["time_pred"], 'time_pred')
    time_mean, time_std = self.dataset.get_time_stats()
    path_logits = model_output["path_logits"]
    time_pred = model_output["time_pred"]  
    log_time_pred = time_pred * time_std + time_mean
    log_time_true = time_values * time_std + time_mean
    time_pred = torch.exp(log_time_pred)
    time_true = torch.exp(log_time_true)
    if self.parameterization == 'sedd':
      return dsigma * self._score_entropy(
        path_logits, sigma, xt, x0_path)
    time_loss = F.smooth_l1_loss(time_pred, time_true)  
    log_p_theta = torch.gather(
      input=path_logits,
      dim=-1,
      index=x0_path[:, :, None]).squeeze(-1)
    path_loss = loss_scale * log_p_theta 
    return path_loss,time_loss

  def _loss(self, input_ids, time_values, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       input_ids, attention_mask)
    path_loss, time_loss = self._forward_pass_diffusion(
      input_tokens,
      time_values,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max,)

    path_loss_masked = path_loss * attention_mask
    time_loss_masked = time_loss * attention_mask

    # normalized loss
    path_nll = path_loss_masked.sum() / attention_mask.sum()
    time_nll = time_loss_masked.sum() / attention_mask.sum()

    # time_weight is a float (configurable)
    total_loss = path_nll + self.time_weight * time_nll

    return Loss(
        loss=total_loss,
        path_loss=path_nll,
        time_loss=time_nll,
        nlls=path_loss_masked,
        token_mask=attention_mask
    )








  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _score_entropy(self, log_score, sigma, xt, x0):
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  @torch.no_grad()
  def _analytic_sampler(self,
                        n_samples: int,
                        num_steps: int,
                        seqlen: int,
                        eps: float = 1e-5):
      # -------- ① 取初值 --------
      x_path, x_time = self._sample_prior(n_samples, seqlen)

      # -------- ② 构造时间步表 --------
      sigmas = torch.linspace(1.0, eps, num_steps + 1, device=self.device)

      # -------- ③ 主循环：Greedy 更新两路 --------
      for i in tqdm(range(num_steps), desc="step"):
          sigma = sigmas[i].expand(n_samples, 1)          # [B,1]

          logits_path, logits_time = self.forward(
              input_ids=x_path,
              time_ids=x_time,
              sigma=sigma
          )

          x_path = torch.argmax(logits_path,  dim=-1)
          x_time = torch.argmax(logits_time, dim=-1) + 8263

      # -------- ④ 最后一次 denoise --------
      sigma = sigmas[-1].expand(n_samples, 1)
      logits_path, logits_time = self.forward(x_path, x_time, sigma)
      x_path = torch.argmax(logits_path,  dim=-1)
      x_time = torch.argmax(logits_time, dim=-1) + 8263

      # -------- ⑤ 安全检查 --------
      assert x_time.min() >= 8263
      assert x_time.max() <  8263 + 256

      return x_path, x_time         

  @torch.no_grad()
  def _semi_ar_sampler(
      self, n_samples, num_steps, num_strides, seqlen, context_size=1024
  ):
      if seqlen is None:
          seqlen = self.config.model.length
      sampling_steps = 0

      mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
      if mdlm_semi_ar:
          num_strides = self.config.model.length // 512 - 1

      ones = torch.ones((n_samples, 1), dtype=self.dtype, device=self.device)
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      dt = 1 / num_steps

      # Reset cached KVs
      if self.config.sampling.kv_cache:
          self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

      x_accum_path, x_accum_time = None, None

      for stride_num in tqdm(range(num_strides)):
          # 1. Sample a new block (masked path + noisy time)
          x_path, x_time = self._sample_prior(n_samples, self.block_size)

          if stride_num == 0:
              x_accum_path = x_path
              x_accum_time = x_time
          else:
              x_accum_path = torch.cat((x_accum_path, x_path), dim=1)
              x_accum_time = torch.cat((x_accum_time, x_time), dim=1)

          # 2. Select the forward index range
          if mdlm_semi_ar and stride_num > 0:
              fwd_idx = torch.arange(512 * stride_num, 512 * stride_num + self.block_size, device=self.device)
          else:
              end_idx = (stride_num + 1) * self.block_size
              start_idx = max(end_idx - context_size, 0)
              fwd_idx = torch.arange(start_idx, end_idx, device=self.device)

          # 3. Start DDPM denoising for current block
          p_x0_cache = None
          t = 1.0

          for i in range(num_steps):
              if self.mask_index not in x_accum_path[:, fwd_idx]:
                  break
              if self.config.sampling.first_hitting:
                  u = np.random.rand()
                  num_masked = (x_accum_path[:, fwd_idx] == self.mask_index).sum(-1).item()
                  t *= u ** (1 / num_masked)
              else:
                  t = timesteps[i]

              p_x0_cache, x_next_path, x_next_time = self._ddpm_caching_update(
                  x_path=x_accum_path[:, fwd_idx],
                  x_time=x_accum_time[:, fwd_idx],
                  t=t * ones,
                  dt=dt,
                  p_x0=p_x0_cache
              )
              if p_x0_cache is None:
                  sampling_steps += 1
              x_accum_path[:, fwd_idx] = x_next_path
              x_accum_time[:, fwd_idx] = x_next_time
          if x_accum_path.shape[1] > 256:
              stop, x_accum_path = self._check_stop_conds(x_accum_path)
              if (stop and not self.config.sampling.var_length) or (stop and x_accum_path.shape[-1] == 1):
                  return None, None, sampling_steps
              elif stop:
                  break

      return x_accum_path, x_accum_time, sampling_steps

  
  
  # def _semi_ar_sampler(
  #   self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
  #   if seqlen is None:
  #     seqlen = self.config.model.length
  #   sampling_steps = 0
          
  #   mdlm_semi_ar = self.config.algo.name == 'mdlm' and self.config.model.length > self.block_size
  #   if mdlm_semi_ar:
  #     # sliding window of length 512 for mdlm semi-ar decoding
  #     num_strides = self.config.model.length // 512
  #     num_strides -= 1

  #   ones = torch.ones((n_samples,1), dtype=self.dtype,
  #                     device=self.device)
    
  #   # reset kvs
  #   if self.config.sampling.kv_cache:
  #     self.backbone.reset_kv_cache()

  #   for stride_num in tqdm(range(num_strides)):
  #     # sample next block
  #     if stride_num == 0:
  #       x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
  #       #x_accum[:, 0] = self.tokenizer.bos_token_id
  #     else:
  #       if mdlm_semi_ar:
  #         x = self._sample_prior(n_samples, 512).to(self.device)
  #       else:
  #         x = self._sample_prior(n_samples, self.block_size).to(self.device)
  #       x_accum = torch.cat((x_accum, x), dim=1)

  #     # compute logits in a sliding window (context passed to model can't exceed context_size)
  #     end_idx = (stride_num + 1) * self.block_size
  #     start_idx = max(end_idx - context_size, 0)
  #     fwd_idx = torch.arange(start_idx, end_idx)
  #     if mdlm_semi_ar and stride_num > 0: # MDLM
  #       fwd_idx = torch.arange(512*(stride_num), (512*(stride_num))+self.block_size)

  #     dt = 1 / num_steps
  #     p_x0_cache = None
  #     timesteps = torch.linspace(1, 0, num_steps, device=self.device)
  #     t = 1
  #     for i in range(num_steps):
  #       if self.mask_index not in x_accum:
  #         break

  #       # faster (equivalent) sampler from zheng et al (2025)
  #       if self.config.sampling.first_hitting:
  #         u = np.random.rand()
  #         num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
  #         t *= u**(1 / num_masked)
              
  #       elif not self.config.sampling.first_hitting:
  #         t = timesteps[i]

  #       p_x0_cache, x_next = self._ddpm_caching_update(
  #           x=x_accum[:, fwd_idx],
  #           t=t * ones,
  #           dt=dt,
  #           p_x0=p_x0_cache,)
  #       if p_x0_cache is None:
  #         sampling_steps += 1
       
  #       x_accum[:, fwd_idx] = x_next

  #     # check if we need to resample (or stop sampling for variable-length sampling)
  #     if x_accum.shape[1] > 256:
  #       stop, x_accum = self._check_stop_conds(x_accum)
  #       if (stop and not self.config.sampling.var_length) \
  #         or (stop and x.shape[-1] == 1):
  #         return None, None
  #       elif stop:
  #         break
  #   return x_accum, sampling_steps
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
  
def decode_log_time_samples(log_time_tensor, time_mean, time_std):
    """
    log_time_tensor: [B, L] tensor from model
    time_mean, time_std: mean and std in log domain
    """
    log_time = log_time_tensor * time_std + time_mean
    time = torch.exp(log_time)
    time = torch.clamp(time, min=0.0, max=180.0)  # optional cap at 1 hour
    return time