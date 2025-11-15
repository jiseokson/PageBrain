import logging
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pagebrain.block import BlockManager
from pagebrain.cache import CacheManager
from pagebrain.modules import GPT2PagedAttention
from utils import set_random_seed


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
d_model = model.config.n_embd
d_head = model.config.n_embd // num_heads
kv_dtype = torch.float32

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


def test_GPT2PagedAttention_prefill(use_seed):
  if use_seed:
    set_random_seed(42)

  # Determine prefill_len as the valid token length shared by all batch samples
  inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  prefill_len = torch.min(inputs['attention_mask'].sum(-1)).item()
  inputs = {k: v[:, :prefill_len] for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs, use_cache=True, output_hidden_states=True)

  # Prepare the inputs required by each layer’s attention module
  # GPT2Attention receives layer-normalized hidden states as input
  layer_attn_input_states = []
  # Also, prepare the outputs for each layer as well
  layer_attn_output_states = []
  with torch.no_grad():
    for block, block_input_states in zip(model.transformer.h, outputs.hidden_states[:-1]):
      attn_input_states = block.ln_1(block_input_states)
      attn_output_states = block.attn(attn_input_states)[0]

      layer_attn_input_states.append(attn_input_states)
      layer_attn_output_states.append(attn_output_states)

  num_blocks = 1000
  page_size = 32

  block_manager = BlockManager(num_blocks, num_layers, num_heads, d_head, page_size, device, dtype=kv_dtype)
  cache_manager = CacheManager(block_manager)

  # Create a PagedAttention module for each layer
  layer_paged_attn = [
    GPT2PagedAttention(
      model.transformer.h[layer_idx].attn,
      layer_idx,
      cache_manager,
      num_heads,
      d_head,
    ).eval()
    for layer_idx in range(num_layers)
  ]

  # For each layer, forward the PagedAttention module using new input to obtain output
  batch_size = len(prompts)
  layer_paged_attn_output_states = []
  seq_ids = [str(uuid.uuid4().hex) for _ in range(batch_size)]
  for paged_attn, attn_input_states in zip(layer_paged_attn, layer_attn_input_states):
    cache_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.int)

    input_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.int)
    input_pos[:, 0] = 0
    input_pos[:, 1] = prefill_len

    with torch.no_grad():
      paged_attn_output_states = paged_attn(attn_input_states, seq_ids, input_pos, cache_pos) # [B, T, C]
    layer_paged_attn_output_states.append(paged_attn_output_states)

  # Compare outputs of the baseline attention and PagedAttention
  for attn_output_states, paged_attn_output_states in \
    zip(layer_attn_output_states, layer_paged_attn_output_states):
    assert torch.isclose(attn_output_states, paged_attn_output_states, rtol=1e-4, atol=1e-4).all().item()


def test_GPT2PagedAttention_step_after_prefill(use_seed):
  if use_seed:
    set_random_seed(42)

  # Determine prefill_len as the valid token length shared by all batch samples
  inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  prefill_len = torch.min(inputs['attention_mask'].sum(-1)).item()
  inputs = {k: v[:, :prefill_len] for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs, use_cache=True, output_hidden_states=True)

  # Prepare the inputs required by each layer’s attention module
  # GPT2Attention receives layer-normalized hidden states as input
  layer_attn_input_states = []
  with torch.no_grad():
    for block, block_input_states in zip(model.transformer.h, outputs.hidden_states[:-1]):
      attn_input_states = block.ln_1(block_input_states)
      layer_attn_input_states.append(attn_input_states)

  # After generating a random input corresponding to the next token,
  # run a forward pass together with the previous inputs to obtain the attention-module outputs for each layer.
  batch_size = len(prompts)
  layer_next_states = []
  layer_attn_output_states = []
  with torch.no_grad():
    for block, attn_input_states in zip(model.transformer.h, layer_attn_input_states):
      next_states = torch.rand([batch_size, 1, d_model], device=device, dtype=kv_dtype) # [B, 1, C]
      attn_input_states = torch.cat([attn_input_states, next_states], dim=1)            # [B, T+1, C]
      # Slice only the output corresponding to the last token
      attn_output_states = block.attn(attn_input_states)[0][:, -1:] # [B, 1, C]

      layer_next_states.append(next_states)
      layer_attn_output_states.append(attn_output_states)

  # PagedAttention accesses the KV cache through CacheManager, so create this object
  num_blocks = 1000
  page_size = 32

  block_manager = BlockManager(num_blocks, num_layers, num_heads, d_head, page_size, device, dtype=kv_dtype)
  cache_manager = CacheManager(block_manager)

  # Write the KV cache obtained from the HF model’s prefill forward
  seq_ids = [str(uuid.uuid4().hex) for _ in range(batch_size)]
  for layer_idx, layer_cache in enumerate(outputs.past_key_values.layers):
    input_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.int)
    input_pos[:, 1] = prefill_len
    cache_manager.update(seq_ids, layer_idx, layer_cache.keys, layer_cache.values, input_pos)

  # Create a PagedAttention module for each layer
  layer_paged_attn = [
    GPT2PagedAttention(
      model.transformer.h[layer_idx].attn,
      layer_idx,
      cache_manager,
      num_heads,
      d_head,
    ).eval()
    for layer_idx in range(num_layers)
  ]

  # For each layer, forward the PagedAttention module using the cache and new input to obtain output
  layer_paged_attn_output_states = []
  for paged_attn, next_states in zip(layer_paged_attn, layer_next_states):
    cache_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.int)
    cache_pos[:, 0] = 0
    cache_pos[:, 1] = prefill_len

    input_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.int)
    input_pos[:, 0] = prefill_len
    input_pos[:, 1] = 1

    with torch.no_grad():
      paged_attn_output_states = paged_attn(next_states, seq_ids, input_pos, cache_pos) # [B, 1, C]
    layer_paged_attn_output_states.append(paged_attn_output_states)

  # Compare outputs of the baseline attention and PagedAttention
  for attn_output_states, paged_attn_output_states in \
    zip(layer_attn_output_states, layer_paged_attn_output_states):
    assert torch.isclose(attn_output_states, paged_attn_output_states, rtol=1e-4, atol=1e-4).all().item()
