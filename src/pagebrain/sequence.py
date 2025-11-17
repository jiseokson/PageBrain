import asyncio
from typing import List, Literal, Optional
import uuid

import torch


SeqId = str


class Sequence:
  def __init__(self,
    prompt: str,
    max_new_tokens: int,
    method: Literal['greedy', 'top_p', 'top_k'],
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[float],
    event: asyncio.Event,
  ):
    self.id: SeqId = str(uuid.uuid4().hex)

    self.prompt: str = prompt
    self.max_new_tokens: int = max_new_tokens
    self.method: Literal['greedy', 'top_p', 'top_k'] = method
    self.temperature: float = temperature
    self.top_p: Optional[float] = top_p
    self.top_k: Optional[float] = top_k

    self.event: asyncio.Event = event
    # If SequenceGroup.update() finishes processing all tokens in the token buffer,
    # insert a new token into the buffer for the next generation step and set new_token=True.
    self.new_token = False
    # Set when the Engine determines that gen_tokens has reached max_new_tokens
    self.done = False

    self.token_buffer: List[int] = []
    self.cache_start: int = 0
    self.cache_len: int = 0
    self.input_start: int = 0
    self.input_len: int = 0
    # Decoding of generated token is handled by the Engine
    self.gen_tokens: List[str] = []


class SequenceGroup:
  def __init__(self, seqs: List[Sequence], device):
    self.seqs = seqs
    self.device = device

    self.input_ids: torch.Tensor
    self.seq_ids: List[SeqId] = [seq.id for seq in seqs]
    self.batch_size = len(seqs)

    self.cache_pos: torch.Tensor
    self.input_pos: torch.Tensor

    cache_starts = [seq.cache_start for seq in seqs]
    cache_lens = [seq.cache_len for seq in seqs]
    input_starts = [seq.input_start for seq in seqs]
    input_lens = [seq.input_len for seq in seqs]

    self.cache_pos = torch.tensor(
      [cache_starts, cache_lens], device=device, dtype=torch.long
    ).transpose(0, 1).contiguous()
    self.input_pos = torch.tensor(
      [input_starts, input_lens], device=device, dtype=torch.long
    ).transpose(0, 1).contiguous()

    batch_size = len(seqs)
    input_width = max(input_lens)
    self.input_ids = torch.zeros([batch_size, input_width], device=device, dtype=torch.long)
    for sample_idx, seq in enumerate(seqs):
      self.input_ids[sample_idx, :seq.input_len] = torch.tensor(seq.token_buffer[:seq.input_len])

  def update(self, next_token_ids: torch.Tensor, next_tokens: List[str]):
    for seq, next_token_id, next_token in zip(self.seqs, next_token_ids.tolist(), next_tokens):
      if seq.input_len == len(seq.token_buffer):
        seq.token_buffer.append(next_token_id)
        seq.gen_tokens.append(next_token)
        seq.new_token = True
      seq.token_buffer = seq.token_buffer[seq.input_len:]

    cache_pos = torch.stack(
      [self.cache_pos[:, 0], self.input_pos[:, 0] + self.input_pos[:, 1]], dim=-1
    )
    input_pos = torch.stack(
      [self.cache_pos[:, 1], torch.ones([self.batch_size], device=self.device, dtype=torch.long)], dim=-1
    )

    for seq, (cache_start, cache_len) in zip(self.seqs, cache_pos.tolist()):
      seq.cache_start = cache_start
      seq.cache_len = cache_len

    for seq, (input_start, input_len) in zip(self.seqs, input_pos.tolist()):
      seq.input_start = input_start
      seq.input_len = input_len
