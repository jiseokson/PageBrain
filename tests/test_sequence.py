import asyncio
import logging
import random

import torch
from transformers import AutoTokenizer
from pagebrain.engine import Engine
from pagebrain.sequence import Sequence, SequenceGroup
from utils import make_engine_reqs, set_random_seed


logger = logging.getLogger(__name__)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'openai-community/gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


prompts = [
  'Once upon a time, there was a',
  'In the future, AI will',
  'The meaning of life is',
  'FastAPI is a great framework for',
  'Transformers models are powerful for',
  'It was a sunny day when',
  'Quantum computing will change',
  'The secret to happiness is',
  'Long ago in a galaxy far',
  'Python is the best language for',
]


def test_sequence_group(use_seed):
  if use_seed:
    set_random_seed(42)

  engine_reqs = make_engine_reqs(prompts)

  seqs = []
  for engine_req in engine_reqs:
    seq = Sequence(
      engine_req.prompt,
      engine_req.max_new_tokens,
      engine_req.method,
      engine_req.temperature,
      engine_req.top_p,
      engine_req.top_k,
      asyncio.Event(),
    )
    seqs.append(seq)

  engine = Engine(model_name=model_name, device=device)
  # Fill the token_buffer using the function below
  # Tests for this function are written in test_engine.py
  engine._init_batch_sequence_before_sched(seqs)

  # Simulate a situation where the scheduler determines the prefill forward length
  cache_starts = []
  cache_lens = []
  input_starts = []
  input_lens = []
  for seq in seqs:
    tokens = len(seq.token_buffer)
    seq.cache_start = 0
    seq.cache_len = 0
    seq.input_start = 0
    seq.input_len = random.randint(1, tokens)

    cache_starts.append(seq.cache_start)
    cache_lens.append(seq.cache_len)
    input_starts.append(seq.input_start)
    input_lens.append(seq.input_len)

  seq_group = SequenceGroup(seqs, device=device)
  assert (seq_group.cache_pos[:, 0] == torch.tensor(cache_starts, device=device)).all().item()
  assert (seq_group.cache_pos[:, 1] == torch.tensor(cache_lens, device=device)).all().item()
  assert (seq_group.input_pos[:, 0] == torch.tensor(input_starts, device=device)).all().item()
  assert (seq_group.input_pos[:, 1] == torch.tensor(input_lens, device=device)).all().item()

  # Test whether input_ids are filled up to the region determined by the prefill forward
  for seq, input_id, (start, length) in \
    zip(seqs, seq_group.input_ids, seq_group.input_pos.tolist()):
    assert (input_id[:length].squeeze() == torch.tensor(seq.token_buffer[:length], device=device)).all().item()


def test_sequence_group_full_update(use_seed):
  if use_seed:
    set_random_seed(42)

  engine_reqs = make_engine_reqs(prompts)

  seqs = []
  for engine_req in engine_reqs:
    seq = Sequence(
      engine_req.prompt,
      engine_req.max_new_tokens,
      engine_req.method,
      engine_req.temperature,
      engine_req.top_p,
      engine_req.top_k,
      asyncio.Event(),
    )
    seqs.append(seq)

  engine = Engine(model_name=model_name, device=device)
  # Fill the token_buffer using the function below
  # Tests for this function are written in test_engine.py
  engine._init_batch_sequence_before_sched(seqs)

  # Simulate a situation where the scheduler determines the prefill forward length
  cache_starts = []
  cache_lens = []
  input_starts = []
  input_lens = []
  for seq in seqs:
    tokens = len(seq.token_buffer)
    seq.cache_start = 0
    seq.cache_len = 0
    seq.input_start = 0
    # Forward is determined for the full token length of every sample,
    # so all new tokens must be added to the buffer
    seq.input_len = tokens

    cache_starts.append(seq.cache_start)
    cache_lens.append(seq.cache_len)
    input_starts.append(seq.input_start)
    input_lens.append(seq.input_len)

  seq_group = SequenceGroup(seqs, device=device)

  batch_size = len(prompts)
  next_token_ids = torch.randint(0, 100, [batch_size], device=device)
  next_tokens = tokenizer.batch_decode(next_token_ids.tolist())
  seq_group.update(next_token_ids, next_tokens)

  for seq, next_token_id in zip(seqs, next_token_ids.tolist()):
    assert len(seq.token_buffer) == 1
    assert seq.token_buffer[-1] == next_token_id


def test_sequence_group_few_update(use_seed):
  if use_seed:
    set_random_seed(42)

  engine_reqs = make_engine_reqs(prompts)

  seqs = []
  for engine_req in engine_reqs:
    seq = Sequence(
      engine_req.prompt,
      engine_req.max_new_tokens,
      engine_req.method,
      engine_req.temperature,
      engine_req.top_p,
      engine_req.top_k,
      asyncio.Event(),
    )
    seqs.append(seq)

  engine = Engine(model_name=model_name, device=device)
  # Fill the token_buffer using the function below
  # Tests for this function are written in test_engine.py
  engine._init_batch_sequence_before_sched(seqs)

  # Simulate a situation where the scheduler determines the prefill forward length
  cache_starts = []
  cache_lens = []
  input_starts = []
  input_lens = []
  buffer_lens = []
  for seq in seqs:
    tokens = len(seq.token_buffer)
    seq.cache_start = 0
    seq.cache_len = 0
    seq.input_start = 0
    # A situation where all samples are determined to forward fewer tokens 
    # than their full length. Therefore, the token buffer should shrink only 
    # by the number of processed tokens, and no new tokens should be added.
    seq.input_len = random.randint(1, tokens-1)

    cache_starts.append(seq.cache_start)
    cache_lens.append(seq.cache_len)
    input_starts.append(seq.input_start)
    input_lens.append(seq.input_len)
    buffer_lens.append(tokens)

  seq_group = SequenceGroup(seqs, device=device)

  batch_size = len(prompts)
  next_token_ids = torch.randint(0, 100, [batch_size], device=device)
  next_tokens = tokenizer.batch_decode(next_token_ids.tolist())
  seq_group.update(next_token_ids, next_tokens)

  for seq, buffer_len, input_len in zip(seqs, buffer_lens, input_lens):
    assert len(seq.token_buffer) == buffer_len - input_len
