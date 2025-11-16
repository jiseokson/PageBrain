import asyncio
from typing import Literal, Optional
import uuid


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
    self.new_token = False
    self.done = False

    # self.cache_pos
    # self.input_pos
    # self.token_ids
    # self.num_gen_tokens
    self.token_text: str
