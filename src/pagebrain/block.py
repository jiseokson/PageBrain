from typing import List

import torch

from pagebrain.config import PageBrainConfig


# Separate block index space for each KV, and layer_idx
class BlockManager:
  def __init__(self, config: PageBrainConfig):
    self.config = config
    self.num_blocks = config.num_blocks
    self.num_layers = config.num_layers
    self.num_heads = config.num_heads
    self.d_head = config.d_head
    self.page_size = config.page_size

    self.device = config.device
    self.dtype = config.dtype

    num_blocks, num_layers, num_heads, page_size, d_head = self.num_blocks, self.num_layers, self.num_heads, self.page_size, self.d_head
    device, dtype  = self.device, self.dtype

    self.gpu_k_pool = [
      torch.empty([num_blocks, num_heads, page_size, d_head], device=device, dtype=dtype)
      for _ in range(num_layers)
    ]
    self.gpu_v_pool = [
      torch.empty([num_blocks, num_heads, page_size, d_head], device=device, dtype=dtype)
      for _ in range(num_layers)
    ]

    self.free_list = [list(range(num_blocks)) for _ in range(num_layers)]

  def alloc(self, layer_idx: int, num_pages: int):
    # !! Need to implement safeguard logic for when no pages are available for allocation !!
    pages = [self.free_list[layer_idx].pop() for _ in range(num_pages)]
    return pages

  def free(self, layer_idx: int, pages: List[int]):
    self.free_list[layer_idx].extend(pages)

  def append_token(self):
    pass

  def build_index(self):
    pass
