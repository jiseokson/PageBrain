import argparse
import asyncio
import logging
import torch
from transformers import AutoTokenizer
from pagebrain.config import PageBrainConfig
from pagebrain.endpoints.args import get_args_parser
from pagebrain.engine import Engine
from pagebrain.sequence import Sequence
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


def test_make_engine_request(use_seed):
  if use_seed:
    set_random_seed(42)

  engine_reqs = make_engine_reqs(prompts)
  for engine_req in engine_reqs:
    assert 0.1 <= engine_req.temperature <= 2.0
    if engine_req.top_p is not None:
      assert 0.5 <= engine_req.top_p <= 1.0
    if engine_req.top_k is not None:
      assert 20 <= engine_req.top_k <= 200


def test__init_batch_sequence_before_sched(use_seed):
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

  inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  input_lens = inputs['attention_mask'].sum(dim=-1)

  parser: argparse.ArgumentParser = get_args_parser()
  config = PageBrainConfig(parser.parse_args())
  engine = Engine(config)
  engine._init()
  engine._init_batch_sequence_before_sched(seqs)

  for seq, input_len in zip(seqs, input_lens.tolist()):
    logger.debug(seq.prompt)
    logger.debug(seq.token_buffer)
    assert type(seq.token_buffer) == type([])
    assert len(seq.token_buffer) == input_len
