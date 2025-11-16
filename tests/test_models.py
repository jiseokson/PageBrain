import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagebrain.block import BlockManager
from pagebrain.cache import CacheManager
from pagebrain.models.gpt2 import PagedGPT2LMHeadModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'openai-community/gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.to(device)
base_model.eval()

num_heads = base_model.config.n_head
num_layers = base_model.config.n_layer
d_model = base_model.config.n_embd
d_head = base_model.config.n_embd // num_heads
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


def test_PagedGPT2LMHeadModel():
  batch_size = len(prompts)
  inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
  seq_ids = [str(uuid.uuid4().hex) for _ in range(batch_size)]

  cache_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.long)

  input_lens = inputs['attention_mask'].sum(dim=-1)
  input_pos = torch.zeros([batch_size, 2], device=device, dtype=torch.long)
  input_pos[:, 1] = input_lens
  assert input_pos.shape == torch.Size([batch_size, 2])
  

  num_blocks = 1000
  page_size = 32

  block_manager = BlockManager(num_blocks, num_layers, num_heads, d_head, page_size, device, dtype=kv_dtype)
  cache_manager = CacheManager(block_manager)

  model = PagedGPT2LMHeadModel(base_model, cache_manager)
  model.eval()

  logits = model(
    input_ids=inputs['input_ids'],
    seq_ids=seq_ids,
    input_pos=input_pos,
    cache_pos=cache_pos,
  )
  
  assert logits.shape == torch.Size([batch_size, torch.max(input_pos[:, 1]).item(), base_model.config.vocab_size])
  
