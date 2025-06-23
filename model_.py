
from models.path_time_backbone import BlockDiffusionModel
import torch
class Diffusion(L.LightningModule):
    def __init__(self, config):
        ...
        # -------- create backbone --------
        self.backbone = BlockDiffusionModel(
            vocab_size      = config.model.vocab_size,      # 8264 (road+PAD)
            time_vocab_size = config.model.time_vocab_size, # 256
            n_embd          = config.model.dim,
            n_layer         = config.model.n_layer,
            block_size      = config.model.block_size,      # 16/32/64
            block_causal    = True,
        )
    def forward(self, batch, sample_mode=False, store_kv=False):
        input_ids = batch['input_ids']      # [B, L']
        time_ids  = batch['time_ids']       # [B, L']
        sigma     = self._process_sigma(batch.get("sigma", None))

        logits_path, logits_time = self.backbone(
            indices   = input_ids,
            time_ids  = time_ids,
            sigma     = sigma,
            sample_mode = sample_mode,
            store_kv    = store_kv,
        )

        # SUBS 仍然调用
        logits_path = self._subs_parameterization(logits_path, xt=input_ids)
        logits_time = self._subs_parameterization(logits_time, xt=time_ids)

        return logits_path, logits_time
    def training_step(self, batch, batch_idx):
        logits_path, logits_time = self(batch)

        loss_path = F.cross_entropy(
            logits_path.reshape(-1, logits_path.size(-1)),
            batch['input_ids'].reshape(-1),
            ignore_index=0
        )
        loss_time = F.cross_entropy(
            logits_time.reshape(-1, logits_time.size(-1)),
            batch['time_ids'].reshape(-1),
            ignore_index=0
        )

        loss = loss_path + loss_time
        self.log_dict({'loss': loss, 'lp': loss_path, 'lt': loss_time})
        return loss


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