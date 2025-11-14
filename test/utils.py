import random

import torch


def set_random_seed(random_seed):
  random_seed = 42
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

def make_dummy_input_cache_pos(batch_size, max_len, device, dtype):
  input_pos = torch.zeros(batch_size, 2, device=device, dtype=dtype)
  cache_pos = torch.zeros(batch_size, 2, device=device, dtype=dtype)

  # 50 : 40 : 10 = [0, len] : [start, 1] : [start, len]
  for sample_idx in range(batch_size):
    if random.randint(0, 1):
      # start = 0
      input_pos[sample_idx, 0] = 0
      input_pos[sample_idx, 1] = random.randint(1, max_len)
    else:
      # start != 0
      if random.randint(0, 4):
        # len == 1
        input_pos[sample_idx, 0] = random.randint(0, max_len-1)
        input_pos[sample_idx, 1] = 1
      else:
        # len > 1
        a, b = random.randint(0, max_len-1), random.randint(0, max_len-1)
        a, b = min(a, b), max(a, b)
        start = a
        length = b - a + 1
        input_pos[sample_idx, 0] = start
        input_pos[sample_idx, 1] = length

  cache_pos[:, 1] = input_pos[:, 0]

  return input_pos, cache_pos


def make_dummy_keys_values(batch_size, num_heads, seq_len, d_head, device, dtype):
  keys = torch.rand([batch_size, num_heads, seq_len, d_head], device=device, dtype=dtype)
  values = torch.rand([batch_size, num_heads, seq_len, d_head], device=device, dtype=dtype)
  return keys, values
