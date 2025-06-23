# bd/models/__init__.py
#from .path_time_backbone import BlockDiffusionModel
import torch, torch.nn as nn
from .dit import Rotary, LayerNorm
from path_time_diffusion import PathTimeTokenizer
class BlockDiffusionModel(nn.Module):
    """Lightweight encoder that returns *two* logits tensors.
    This class is **NOT** the full BD3‑LM denoiser; it is a drop‑in
    placeholder so we can sanity‑check shapes before wiring into the
    original diffusion wrapper. It deliberately has **no sigma / KV /
    block‑causal masking** yet.
    """
    def __init__(self, vocab_total:int,config,vocab_time:int,vocab_path:int,d_model:int=192, n_heads:int=8, n_layers:int=6):
        super().__init__()
        #self.tok = tokenizer
        # self.vocab_total  = tokenizer.time_offset + tokenizer.num_time_bins      # full range used by the embedding
        # self.vocab_path   = tokenizer.time_offset         # == #special + #roads
        # self.vocab_time   = tokenizer.num_time_bins       # fixed #log‑bins

        # ---------- layers ----------
        self.embed = nn.Embedding(vocab_total, d_model)
        enc_layer  = nn.TransformerEncoderLayer(d_model, n_heads,
                                                4*d_model, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, n_layers)
        self.path_head = nn.Linear(d_model, vocab_path)
        self.time_head = nn.Linear(d_model, vocab_time)
        self.n = config.model.length
    def forward(self, config,tokens_path:torch.Tensor, tokens_time:torch.Tensor,sample_mode=False, store_kv=False,sigma=None):
        """tokens_path/tokens_time: [B, L] long
        returns logits_path, logits_time same shape + vocab dim"""
        x = torch.cat([tokens_path, tokens_time], dim=1)      # [B, 2L]
        h = self.encoder(self.embed(x))                       # [B, 2L, d]
        B, L = tokens_path.shape
        logits_path = self.path_head(h[:, :L])             # [B, L, vocab_path]
        logits_time = self.time_head(h[:, L:])      
        logits_path = logits_path[:, :self.n]
        logits_time = logits_time[:, :self.n]# [B, L, vocab_time]
        return logits_path, logits_time
