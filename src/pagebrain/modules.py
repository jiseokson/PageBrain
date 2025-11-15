import logging
from typing import List
import torch
from torch import nn

from pagebrain.cache import CacheManager
from pagebrain.sequence import SeqId


logger = logging.getLogger(__name__)


def make_attn_mask(
  batch_size: int,
  k_len: int,
  pos: torch.Tensor,
  device, dtype
) -> torch.Tensor:
  attn_mask = torch.zeros([batch_size, k_len], device=device, dtype=torch.int)
  for sample_idx, (start, length) in enumerate(pos.tolist()):
    attn_mask[sample_idx, start : start + length] = 1
  attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min
  attn_mask = attn_mask[:, None, None, :]
  return attn_mask


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
    self.page_size = cache_manager.page_size

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
    scale = 1.0 / d_head**0.5
    assert d_head == d_model // num_heads, \
    f'The last dimension of hidden_states(={d_model}) does not match '
    f'self.num_heads(={num_heads}) self.d_head(={d_head})'

    device = hidden_states.device
    dtype = hidden_states.dtype

    qkv = self.base_attn.c_attn(hidden_states)     # [B, Tq, 3C]
    q, k_curr, v_curr = qkv.split(d_model, dim=-1) # [B, Tq, C] x3

    q      = q.view(batch_size, q_len, num_heads, d_head).transpose(1, 2).contiguous()
    k_curr = k_curr.view(batch_size, q_len, num_heads, d_head).transpose(1, 2).contiguous()
    v_curr = v_curr.view(batch_size, q_len, num_heads, d_head).transpose(1, 2).contiguous()
    assert q.shape == torch.Size([batch_size, num_heads, q_len, d_head])
    assert k_curr.shape == torch.Size([batch_size, num_heads, q_len, d_head])
    assert v_curr.shape == torch.Size([batch_size, num_heads, q_len, d_head])

    m   = torch.full( [batch_size, num_heads, q_len,      1], -float('inf'), device=device, dtype=dtype)
    l   = torch.zeros([batch_size, num_heads, q_len,      1], device=device, dtype=dtype)
    acc = torch.zeros([batch_size, num_heads, q_len, d_head], device=device, dtype=dtype)

    for k_p, v_p, page_pos in self.cache_manager.iter_page(seq_ids, self.layer_idx, cache_pos):
      attn_mask = make_attn_mask(batch_size, self.page_size, page_pos, device, dtype)
      s_p = q @ k_p.transpose(-1, -2) * scale + attn_mask

      m_p = s_p.max(dim=-1, keepdim=True).values
      m_new = torch.maximum(m, m_p)
      alpha = torch.exp(m - m_new)
      m = m_new

      e_p = torch.exp(s_p - m)
      acc = alpha * acc + e_p @ v_p
      l = alpha * l + e_p.sum(dim=-1, keepdim=True)

    self.cache_manager.update(seq_ids, self.layer_idx, k_curr, v_curr, input_pos)

    aligend_input_pos = input_pos.clone()
    aligend_input_pos[:, 0] = 0
    attn_mask = make_attn_mask(batch_size, q_len, aligend_input_pos, device, dtype)
    causal_mask = torch.triu(torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1)

    s_curr = q @ k_curr.transpose(-1, -2) * scale + attn_mask
    s_curr = s_curr.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m_p = s_curr.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m, m_p)
    alpha = torch.exp(m - m_new)
    m = m_new

    e_p = torch.exp(s_curr - m)
    acc = alpha * acc + e_p @ v_curr
    l = alpha * l + e_p.sum(dim=-1, keepdim=True)

    context = acc / torch.clamp(l, min=1e-9)
    context = context.transpose(1, 2).contiguous().view(batch_size, q_len, d_model)

    out = self.base_attn.c_proj(context)
    out = self.base_attn.resid_dropout(out)

    return out
