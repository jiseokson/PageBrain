import logging
import torch
from torch.nn import functional as F
from pagebrain.config import PageBrainConfig
from pagebrain.models.gpt2 import PagedGPT2LMHeadModel
from pagebrain.sequence import SequenceGroup


logger = logging.getLogger('uvicorn')


def sample_top_p(logits: torch.Tensor, temperature: torch.Tensor, p: torch.Tensor):
  batch_size, vocab_size = logits.shape

  temperature = temperature.unsqueeze(1)
  logits = logits / temperature.clamp(min=1e-5)

  sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

  sorted_probs = F.softmax(sorted_logits, dim=-1)
  cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

  p = p.unsqueeze(1) # [B, 1]
  cutoff = cumulative_probs > p

  cutoff[:, 1:] = cutoff[:, :-1].clone()
  cutoff[:, 0] = False

  sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))

  sorted_probs = F.softmax(sorted_logits, dim=-1)

  probs = torch.zeros_like(sorted_probs).scatter_(-1, sorted_indices, sorted_probs)

  next_token = torch.multinomial(probs, num_samples=1) # [B, 1]
  return next_token.squeeze(1) # [B]


def sample_top_k(logits: torch.Tensor, temperature: torch.Tensor, k: torch.Tensor):
  batch_size, vocab_size = logits.shape

  temperature = temperature.unsqueeze(1)
  logits = logits / temperature.clamp(min=1e-5)

  sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

  k = k.clamp(min=1, max=vocab_size).long()

  kth_values = sorted_logits.gather(1, (k - 1).unsqueeze(1))  # [B, 1]

  mask = logits < kth_values  # [B, V]
  logits = logits.masked_fill(mask, float("-inf"))

  probs = F.softmax(logits, dim=-1)

  next_token = torch.multinomial(probs, num_samples=1) # [B, 1]

  return next_token.squeeze(1) # [B]


class Executor:
  def __init__(self, base_model, cache_manager, config: PageBrainConfig):
    self.base_model = base_model
    self.cache_manager = cache_manager

    self.config = config
    self.model_name = config.model_name
    self.device = config.device

    # !! Need logic to select the appropriate implementation based on model_name !!
    self.model = PagedGPT2LMHeadModel(base_model, cache_manager).to(self.device)
    self.model.eval()

  def step(self, seq_group: SequenceGroup) -> torch.Tensor:
    with torch.no_grad():
      logits = self.model(
        input_ids=seq_group.input_ids,
        seq_ids=seq_group.seq_ids,
        input_pos=seq_group.input_pos,
        cache_pos=seq_group.cache_pos,
      )

    batch_size = len(seq_group.seqs)
    next_logits = logits[torch.arange(batch_size), seq_group.input_pos[:, 1]-1]
    
    greedy_sample_ids = []

    top_p_sample_ids = []
    top_p_temperatures = []
    top_ps = []

    top_k_sample_ids = []
    top_k_temperatures = []
    top_ks = []

    for sample_idx, seq in enumerate(seq_group.seqs):
      if seq.method == 'greedy':
        greedy_sample_ids.append(sample_idx)
      elif seq.method == 'top_p':
        top_p_sample_ids.append(sample_idx)
        top_p_temperatures.append(seq.temperature)
        top_ps.append(seq.top_p)
      elif seq.method == 'top_k':
        top_k_sample_ids.append(sample_idx)
        top_k_temperatures.append(seq.temperature)
        top_ks.append(seq.top_k)

    if len(greedy_sample_ids) > 0:
      greedy_sample_ids = torch.tensor(greedy_sample_ids, device=self.device, dtype=torch.long)
      greedy_next_token_ids = next_logits[greedy_sample_ids].argmax(dim=-1)
    
    if len(top_p_sample_ids) > 0:
      top_p_sample_ids = torch.tensor(top_p_sample_ids, device=self.device, dtype=torch.long)
      top_p_temperatures = torch.tensor(top_p_temperatures, device=self.device)
      top_ps = torch.tensor(top_ps, device=self.device)
      top_p_next_token_ids = sample_top_p(next_logits[top_p_sample_ids], temperature=top_p_temperatures, p=top_ps)

    if len(top_k_sample_ids):
      top_k_sample_ids = torch.tensor(top_k_sample_ids, device=self.device, dtype=torch.long)
      top_k_temperatures = torch.tensor(top_k_temperatures, device=self.device)
      top_ks = torch.tensor(top_ks, device=self.device)
      top_k_next_token_ids = sample_top_k(next_logits[top_k_sample_ids], temperature=top_k_temperatures, k=top_ks)

    next_token_ids = torch.zeros([batch_size], device=self.device, dtype=torch.long)
    if len(greedy_sample_ids) > 0:
      next_token_ids[greedy_sample_ids] = greedy_next_token_ids
    if len(top_p_sample_ids) > 0:
      next_token_ids[top_p_sample_ids] = top_p_next_token_ids
    if len(top_k_sample_ids):
      next_token_ids[top_k_sample_ids] = top_k_next_token_ids

    return next_token_ids
