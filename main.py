import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers
import pickle
import dataloader
import diffusion
import utils
from dataloader import PathTimeTokenizer
import pandas as pd 
# from path_time_diffusion import get_path_time_dataloaders
# tokenizer = PathTimeTokenizer(max_path_id=8263, num_time_bins=256)
# train_loader, val_loader = get_path_time_dataloaders(config, tokenizer)
#os.environ['CUDA_VISIBLE_DEVICES="2"]
import os
os.environ["WANDB_MODE"] = "disabled"

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=None,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for name, dl in [('train', train_ds), ('valid', valid_ds)]:
    print(f'{name} sample:')
    batch = next(iter(dl))
    print(batch['path_ids'][0][-k:])
    print(batch['time_values'][0][-k:])





def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  all_samples,all_times = model.restore_model_and_sample(
    num_steps=config.algo.T)
  print('Text samples:', all_samples,all_times)
  
  csv_path = config.sampling.logdir
  csv_time = config.sampling.logdir_t
  #if config.sampling.var_length:
    #print(all_samples)
    #save_dict['samples'] = ['' for _ in range(len(all_samples))]

  #all_samples_cpu = all_samples.cpu()  

  pd.DataFrame(all_samples).to_csv(csv_path, index=False, header=False)
  pd.DataFrame(all_times).to_csv(csv_time, index=False, header=False)

  ##utils.update_and_save_csv(save_dict, csv_path)
  return all_samples,all_times

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)



def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Eval.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=None)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  seed = config.seed
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    #strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  L.seed_everything(seed)
  config.seed = seed
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  trainer.validate(model, valid_ds)
  
def _train(config, logger, tokenizer):
  logger.info('Starting Training')
  tokenizer = PathTimeTokenizer(max_path_id=8263, num_time_bins=256, max_time=3600)
  train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(config,tokenizer = None)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=[hydra.utils.instantiate(cb) for cb in config.get('callbacks', {}).values()],
    logger=L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb) if config.get('wandb') else None
  )

  ckpt_path = config.checkpointing.resume_ckpt_path if config.checkpointing.resume_from_ckpt else None
  trainer.fit(model, train_ds, valid_ds, ckpt_path=None)
  
  
  
@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = None
  if config.mode == 'sample_eval':
    config.wandb = None
    samples,times = generate_samples(config, logger,tokenizer)
  elif config.mode == 'ppl_eval':
    config.wandb = None
    _ppl_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()