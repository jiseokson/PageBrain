from typing import List

import torch


# Separate block index space for each K, V, and layer_idx
class BlockManager:
  def __init__(self, num_blocks, num_layers, num_heads, d_head, page_size, device, dtype):
    self.num_blocks = num_blocks
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.d_head = d_head
    self.page_size = page_size

    self.device = device
    self.dtype = dtype

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
