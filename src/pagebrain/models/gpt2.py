from typing import List
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from pagebrain.modules import PagedGPT2Block
from pagebrain.sequence import SeqId


class PagedGPT2LMHeadModel(nn.Module):
  def __init__(self, base_model: GPT2LMHeadModel, cache_manager):
    super().__init__()

    self.config = base_model.config

    self.wte = base_model.transformer.wte
    self.wpe = base_model.transformer.wpe
    self.drop = base_model.transformer.drop
    self.ln_f = base_model.transformer.ln_f
    self.lm_head = base_model.lm_head

    d_model = base_model.config.hidden_size
    num_heads = base_model.transformer.h[0].attn.num_heads
    d_head = d_model // num_heads

    self.h = nn.ModuleList()
    for layer_idx, block in enumerate(base_model.transformer.h):
      paged_block = PagedGPT2Block(
        base_block = block,
        layer_idx = layer_idx,
        cache_manager = cache_manager,
        num_heads = num_heads,
        d_head = d_head,
      )
      self.h.append(paged_block)

  def forward(
    self,
    input_ids: torch.Tensor,
    seq_ids: List[SeqId],
    input_pos: torch.Tensor,
    cache_pos: torch.Tensor,
  ):
    batch_size, q_len = input_ids.shape
    device = input_ids.device

    position_ids = torch.zeros([batch_size, q_len], device=device, dtype=torch.long)
    for sample_idx, (start, length) in enumerate(input_pos.tolist()):
      position_ids[sample_idx, :length] = torch.arange(start, start + length, device=device)

    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds
    hidden_states = self.drop(hidden_states)

    for block in self.h:
      hidden_states = block(
        hidden_states,
        seq_ids=seq_ids,
        input_pos=input_pos,
        cache_pos=cache_pos,
      )

    hidden_states = self.ln_f(hidden_states)
    logits = self.lm_head(hidden_states)
    return logits
