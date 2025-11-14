import logging
import random
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagebrain.block import BlockManager
from pagebrain.cache import CacheManager
from utils import make_dummy_input_cache_pos, make_dummy_keys_values, set_random_seed

logger = logging.getLogger(__name__)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'openai-community/gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

num_heads = model.config.n_head
num_layers = model.config.n_layer
d_head = model.config.n_embd // num_heads
kv_dtype = torch.float32


def test_prefill_update(use_seed):
  if use_seed:
    set_random_seed(42)
  
  num_blocks = 1000
  batch_size = 10
  max_len = 100
  page_size = 32

  block_manager = BlockManager(num_blocks, num_layers, num_heads, d_head, page_size, device, dtype=kv_dtype)
  cache_manager = CacheManager(block_manager)

  layer_idx = 0
  seq_ids = [str(uuid.uuid4().hex) for _ in range(batch_size)]
  _, cache_pos = make_dummy_input_cache_pos(batch_size, max_len, device, dtype=torch.int)
  kv_seq_len = torch.max(cache_pos[:, 1]).item()
  keys, values = make_dummy_keys_values(batch_size, num_heads, kv_seq_len, d_head, device, dtype=kv_dtype)

  cache_manager.update(seq_ids, layer_idx, keys, values, cache_pos)

  read_keys, read_values = [], []
  for k_p, v_p, page_pos in cache_manager.iter_page(seq_ids, layer_idx, cache_pos):
    # k_p, v_p: [B, H, P, D]
    read_keys.append(k_p)
    read_values.append(v_p)
  read_keys = torch.cat(read_keys, dim=2) # [B, H, n*P, D]
  read_values = torch.cat(read_values, dim=2) # [B, H, n*P, D]

  cache_len = cache_pos[:, 1].tolist()

  for orig_key, read_key, clen in zip(keys, read_keys, cache_len):
    if clen == 0: continue
    assert orig_key[:, :clen, :].shape == read_key[:, :clen, :].shape
    assert torch.isclose(orig_key[:, :clen, :], read_key[:, :clen, :]).all().item()

  for orig_value, read_value, clen in zip(values, read_values, cache_len):
    if clen == 0: continue
    assert orig_value[:, :clen, :].shape == read_value[:, :clen, :].shape
    assert torch.isclose(orig_value[:, :clen, :], read_value[:, :clen, :]).all().item()


def test_step_update(use_seed):
  if use_seed:
    set_random_seed(42)
  
  num_blocks = 1000
  batch_size = 10
  max_len = 100
  page_size = 32

  block_manager = BlockManager(num_blocks, num_layers, num_heads, d_head, page_size, device, dtype=kv_dtype)
  cache_manager = CacheManager(block_manager)

  layer_idx = 0
  seq_ids = [str(uuid.uuid4().hex) for _ in range(batch_size)]
  input_pos, _ = make_dummy_input_cache_pos(batch_size, max_len, device, dtype=torch.int)
  kv_seq_len = torch.max(input_pos[:, 1]).item()
  keys, values = make_dummy_keys_values(batch_size, num_heads, kv_seq_len, d_head, device, dtype=kv_dtype)

  cache_manager.update(seq_ids, layer_idx, keys, values, input_pos)

  read_keys = [[] for _ in range(batch_size)]
  read_values = [[] for _ in range(batch_size)]
  for k_p, v_p, page_pos in cache_manager.iter_page(seq_ids, layer_idx, input_pos):
    # k_p, v_p: [B, H, P, D]
    # page_pos: [B, 2]
    for sample_idx, (start, length) in enumerate(page_pos.tolist()):
      if length == 0: continue
      read_keys[sample_idx].append(k_p[sample_idx, :, start : start+length, :])
      read_values[sample_idx].append(v_p[sample_idx, :, start : start+length, :])

  read_keys = [torch.cat(read_key_list, dim=1) for read_key_list in read_keys]
  read_values = [torch.cat(read_value_list, dim=1) for read_value_list in read_values]

  cache_len = input_pos[:, 1].tolist()

  for orig_key, read_key, clen in zip(keys, read_keys, cache_len):
    if clen == 0: continue
    assert orig_key[:, :clen, :].shape == read_key[:, :clen, :].shape
    assert torch.isclose(orig_key[:, :clen, :], read_key[:, :clen, :]).all().item()

  for orig_value, read_value, clen in zip(values, read_values, cache_len):
    if clen == 0: continue
    assert orig_value[:, :clen, :].shape == read_value[:, :clen, :].shape
    assert torch.isclose(orig_value[:, :clen, :], read_value[:, :clen, :]).all().item()
