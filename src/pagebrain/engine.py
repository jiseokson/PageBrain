from argparse import Namespace
import asyncio
import json
import logging
from typing import List, Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagebrain.block import BlockManager
from pagebrain.cache import CacheManager
from pagebrain.config import PageBrainConfig
from pagebrain.executor import Executor
from pagebrain.schedule import Scheduler
from pagebrain.sequence import Sequence, SequenceGroup


logger = logging.getLogger('uvicorn')


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
    self.top_p: Optional[float] = 0.9 if top_p is None else top_p
    self.top_k: Optional[float] = 50 if top_k is None else top_k

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
  def __init__(self, config: PageBrainConfig):
    self.config = config

    self.seq_queue = asyncio.Queue()
    self.MAX_FETCH_REQ = config.MAX_FETCH_REQ

    self.base_model = None
    self.tokenizer = None

    self.scheduler: Scheduler = None
    self.block_manager: BlockManager = None
    self.cache_manager: CacheManager = None
    self.executor: Executor = None

  def _init(self):
    self.base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.config.device)
    self.config.base_model_config = self.base_model.config

    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # !! Need logic to determine BlockManager constructor arguments from current model and hardware settings !!
    self.config.num_blocks = 200
    self.config.num_layers = 12
    self.config.num_heads = 12
    self.config.d_head = 64

    self.block_manager = BlockManager(
      num_blocks=self.config.num_blocks,
      num_layers=self.config.num_layers,
      num_heads=self.config.num_heads,
      d_head=self.config.d_head,
      page_size=self.config.page_size,
      dtype=self.config.kv_dtype,
      device=self.config.device,
    )
    self.cache_manager = CacheManager(self.block_manager)

    self.scheduler = Scheduler(self.config.device)
    self.executor = Executor(
      self.config.model_name,
      self.base_model,
      self.cache_manager,
      self.config.device
    )

  def start(self):
    self._init()
    asyncio.create_task(self.engine_loop())

  async def engine_loop(self):
    while True:
      await self.step()
      await asyncio.sleep(0.01)

  async def step(self):
    # Gather newly arrived requests and add them to the scheduler
    seqs = []
    while not self.seq_queue.empty() and len(seqs) < self.MAX_FETCH_REQ:
      seq = await self.seq_queue.get()
      seqs.append(seq)

    if len(seqs) > 0:
      self._init_batch_sequence_before_sched(seqs)
      self.scheduler.add(seqs)

    # The scheduler issues the sequence group to generate for this step
    # Adjust the current input position, cache position, and related states
    seq_group: SequenceGroup = self.scheduler.schedule()
    if seq_group is None:
      return

    # Generate the sequence group issued through the executor
    next_token_ids: torch.Tensor = self.executor.step(seq_group)

    # Update each state of Sequence in SequenceGroup & Scheduler
    next_tokens = self.tokenizer.batch_decode(next_token_ids.tolist())
    # Sequence group update and scheduler update must be applied in this order
    # because the scheduler determines completion based on the number of newly generated tokens.
    seq_group.update(next_token_ids, next_tokens)
    done_seqs = self.scheduler.update(seq_group)
    if done_seqs is not None:
      seq_ids = [seq.id for seq in done_seqs]
      self.cache_manager.free(seq_ids)

    # After the generation step, deliver tokens for each request and set their events
    # This also needs to be handed over to the executor
    for seq in seq_group.seqs:
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

      while seq.reply_idx < len(seq.gen_tokens):
        token_event = {
          'event': 'token',
          'request_id': seq.id,
          'text': seq.gen_tokens[-1],
        }
        seq.reply_idx += 1
        yield f'data: {json.dumps(token_event, ensure_ascii=False)}\n\n'

      if seq.done:
        break

    end_event = {
      'event': 'end',
      'request_id': seq.id,
    }
    yield f'data: {json.dumps(end_event, ensure_ascii=False)}\n\n'
    yield 'data: [DONE]\n\n'

  def _init_batch_sequence_before_sched(self, seqs: List[Sequence]):
    # Before adding to the scheduler, pre-encode the tokens to be processed next
    prompts = [seq.prompt for seq in seqs]
    inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.config.device)

    input_ids = inputs['input_ids']
    input_lens = inputs['attention_mask'].sum(dim=-1).tolist()

    batch_size = len(seqs)
    for sample_idx in range(batch_size):
      input_len = input_lens[sample_idx]
      seqs[sample_idx].token_buffer.extend(input_ids[sample_idx, :input_len].tolist())
