import asyncio
import json
from typing import Literal, Optional

from pagebrain.sequence import Sequence


class EngineRequest:
  def __init__(
    self,
    prompt: str,
    max_new_tokens: int,
    method: Literal['greedy', 'top_p', 'top_k'],
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[float],
  ):

    self.prompt: str = prompt
    self.max_new_tokens: int = max_new_tokens
    self.method: Literal['greedy', 'top_p', 'top_k'] = method
    self.temperature: float = temperature
    self.top_p: Optional[float] = top_p
    self.top_k: Optional[float] = top_k

    if method == 'greedy':
      self.top_p = None
      self.top_k = None
    elif method == 'top_p':
      self.top_k = None
    elif method == 'top_k':
      self.top_p = None

  def __repr__(self) -> str:
    return (
      f'EngineRequest(prompt={repr(self.prompt)}, '
      f'max_new_tokens={self.max_new_tokens}, '
      f'method="{self.method}", '
      f'temperature={self.temperature}, '
      f'top_p={self.top_p}, '
      f'top_k={self.top_k})'
    )


class Engine:
  def __init__(
    self,
    model_name: str,
    device,
    kv_dtype,
  ):
    self.model_name = model_name
    self.device = device
    self.kv_dtype = kv_dtype

    self.seq_queue = asyncio.Queue()

    # self.scheduler
    # self.block_manager
    # self.cache_manager
    # self.executor

  def start(self):
    asyncio.create_task(self.engine_loop())

  async def engine_loop(self):
    while True:
      await self.step()
      await asyncio.sleep(0.01)

  async def step(self):
    # Gather newly arrived requests and add them to the scheduler
    seqs = []
    while not self.seq_queue.empty():
      seq = await self.seq_queue.get()
      seqs.append(seq)

    # The scheduler issues the sequence group to generate for this step
    # Adjust the current input position, cache position, and related states
    pass

    # Generate the sequence group issued through the executor
    pass

    # After the generation step, deliver tokens for each request and set their events
    for seq in seqs:
      seq.token_text = seq.prompt
      seq.new_token = True
      seq.done = True
      seq.event.set()

  async def add_request(self, request: EngineRequest):
    event = asyncio.Event()
    seq = Sequence(
      request.prompt,
      request.max_new_tokens,
      request.method,
      request.temperature,
      request.top_p,
      request.top_k,
      event,
    )
    await self.seq_queue.put(seq)

    start_event = {
      'event': 'start',
      'request_id': seq.id,
    }
    yield f'data: {json.dumps(start_event, ensure_ascii=False)}\n\n'

    while True:
      await event.wait()
      event.clear()

      if seq.new_token:
        token_event = {
          'event': 'token',
          'request_id': seq.id,
          'text': seq.token_text,
        }
        seq.new_token = False
        yield f'data: {json.dumps(token_event, ensure_ascii=False)}\n\n'

      if seq.done:
        break

    end_event = {
      'event': 'end',
      'request_id': seq.id,
    }
    yield f'data: {json.dumps(end_event, ensure_ascii=False)}\n\n'
    yield 'data: [DONE]\n\n'
