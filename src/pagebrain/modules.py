import logging
from typing import List
import torch
from torch import nn

from pagebrain.cache import CacheManager
from pagebrain.sequence import SeqId


logger = logging.getLogger(__name__)


class GPT2PagedAttention(nn.Module):
  def __init__(
    self,
    base_attn: nn.Module,
    layer_idx: int,
    cache_manager: CacheManager,
    num_heads: int,
    d_head: int,
  ):
    super().__init__()

    self.base_attn = base_attn
    self.layer_idx = layer_idx
    self.cache_manager = cache_manager

    self.num_heads = num_heads
    self.d_head = d_head

  @torch.no_grad()
  def forward(
    self,
    hidden_states: torch.Tensor,
    seq_ids: List[SeqId],
    input_pos: torch.Tensor,
    cache_pos: torch.Tensor,
  ):
    assert hidden_states.size(0) == len(seq_ids), \
    f'The number of samples in hidden_states(={hidden_states.size(0)}) '
    f'and seq_ids(={len(seq_ids)}) does not match'

    batch_size, q_len, d_model = hidden_states.shape
    num_heads = self.num_heads
    d_head = self.d_head
    assert d_head == d_model // num_heads, \
    f'The last dimension of hidden_states(={d_model}) does not match '
    f'self.num_heads(={num_heads}) self.d_head(={d_head})'

    device = hidden_states.device
    dtype = hidden_states.dtype

    qkv = self.base_attn.c_attn(hidden_states)     # [B, Tq, 3C]
    q, k_curr, v_curr = qkv.split(d_model, dim=-1) # [B, Tq, C] x3

    q      = q.view(batch_size, q_len, num_heads, d_head).transpose(1, 2).contiguous()      # [B, H, Tq, D]
    k_curr = k_curr.view(batch_size, q_len, num_heads, d_head).transpose(1, 2).contiguous() # [B, H, Tq, D]
    v_curr = v_curr.view(batch_size, q_len, num_heads, d_head).transpose(1, 2).contiguous() # [B, H, Tq, D]
    assert q.shape == torch.Size([batch_size, num_heads, q_len, d_head])
    assert k_curr.shape == torch.Size([batch_size, num_heads, q_len, d_head])
    assert v_curr.shape == torch.Size([batch_size, num_heads, q_len, d_head])

    m   = torch.full( [batch_size, num_heads, q_len,      1], -float('inf'), device=device, dtype=dtype) # [B, H, Tq, 1]
    l   = torch.zeros([batch_size, num_heads, q_len,      1], device=device, dtype=dtype)                # [B, H, Tq, 1]
    acc = torch.zeros([batch_size, num_heads, q_len, d_head], device=device, dtype=dtype)                # [B, H, Tq, D]

    for k_p, v_p, page_pos in self.cache_manager.iter_page(seq_ids, self.layer_idx, cache_pos):
      # !! Causal + attention mask not applied yet — must be implemented !!
      s_p = q @ k_p.transpose(-1, -2)

      m_p = s_p.max(dim=-1, keepdim=True).values
      m_new = torch.maximum(m, m_p)
      alpha = torch.exp(m - m_new)
      m = m_new

      e_p = torch.exp(s_p - m)
      acc = alpha * acc + e_p @ v_p
      l = alpha * l + e_p.sum(dim=-1, keepdim=True)

    self.cache_manager.update(seq_ids, self.layer_idx, k_curr, v_curr, input_pos)

    s_curr = q @ k_curr.transpose(-1, -2)

    m_p = s_curr.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m, m_p)
    alpha = torch.exp(m - m_new)
    m = m_new

    e_p = torch.exp(s_curr - m)
    acc = alpha * acc + e_p @ v_curr
    l = alpha * l + e_p.sum(dim=-1, keepdim=True)

    context = acc / torch.clamp(l, min=1e-9)
    logger.debug(context.shape)
    context = context.transpose(1, 2).contiguous().view(batch_size, q_len, d_model)

    out = self.base_attn.c_proj(context)
    out = self.base_attn.resid_dropout(out)

    return out
