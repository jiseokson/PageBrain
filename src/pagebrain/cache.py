from collections import defaultdict
import logging
from typing import List, Tuple

import torch

from pagebrain.block import BlockManager


logger = logging.getLogger(__name__)
SeqId = str

# !! Temporary workaround: implemented by directly accessing BlockManager.pool for recording !!
class CacheManager:
  def __init__(self, block_manager: BlockManager):
    self.block_manager: BlockManager = block_manager
    self.block_table = defaultdict(list)
    self.page_size = block_manager.page_size

  def update(self, seq_ids: List[SeqId], layer_idx: int, k_curr: torch.Tensor, v_curr: torch.Tensor, input_pos: torch.Tensor):
    # Identify and allocate missing pages
    total_tokens = input_pos[:, 0] + input_pos[:, 1]                    # [B]
    total_pages = (total_tokens + self.page_size - 1) // self.page_size # [B]
    logger.debug(f'input_pos:    {input_pos.tolist()}')
    logger.debug(f'total_tokens: {total_tokens.tolist()}')
    logger.debug(f'total_pages:  {total_pages.tolist()}')
    for seq_id, total_page in zip(seq_ids, total_pages):
      allocated_blocks = len(self.block_table[(seq_id, layer_idx)])
      if allocated_blocks < total_page:
        num_pages = total_page - allocated_blocks
        pages = self.block_manager.alloc(layer_idx, num_pages)
        self.block_table[(seq_id, layer_idx)].extend(pages)

    # Compute target page indices and start offsets for writing
    start_page_ids = (input_pos[:, 0] // self.page_size).tolist()    # [B]
    start_page_offsets = (input_pos[:, 0] % self.page_size).tolist() # [B]
    logger.debug(f'start_page_ids:     {start_page_ids}')
    logger.debug(f'start_page_offsets: {start_page_offsets}')

    k_pool = self.block_manager.gpu_k_pool[layer_idx] # [num_blocks, num_heads, page_size, d_head]
    v_pool = self.block_manager.gpu_v_pool[layer_idx] # [num_blocks, num_heads, page_size, d_head]
    for sample_idx, (seq_id, start_page_idx, start_page_offset, new_token) in enumerate(zip(seq_ids, start_page_ids, start_page_offsets, input_pos[:, 1].tolist())):
      block_ids = self.block_table[(seq_id, layer_idx)]

      # The first page may already contain data; write into remaining space
      first_block_id = block_ids[start_page_idx]
      token_idx = min(self.page_size - start_page_offset, new_token)
      logger.debug(f'sample_idx: {sample_idx} - wrote kv_curr[{sample_idx}, :, :{token_idx}, :] into BlockManager pool')
      k_pool[first_block_id, :, start_page_offset : min(self.page_size, start_page_offset + new_token), :] = k_curr[sample_idx, :, :token_idx, :]
      v_pool[first_block_id, :, start_page_offset : min(self.page_size, start_page_offset + new_token), :] = v_curr[sample_idx, :, :token_idx, :]

      # Fill subsequent pages repeatedly by page_size
      for block_id in block_ids[start_page_idx+1:]:
        assert token_idx < new_token
        logger.debug(f'sample_idx: {sample_idx} - wrote kv_curr[{sample_idx}, :, {token_idx}:{min(token_idx + self.page_size, new_token)}, :] into BlockManager pool')
        k_pool[block_id, :, 0 : min(self.page_size, new_token - token_idx), :] = k_curr[sample_idx, :, token_idx : min(token_idx + self.page_size, new_token), :]
        v_pool[block_id, :, 0 : min(self.page_size, new_token - token_idx), :] = v_curr[sample_idx, :, token_idx : min(token_idx + self.page_size, new_token), :]
        token_idx += self.page_size
      
  def iter_pages(self, seq_ids: List[SeqId], layer_idx: int, cache_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # Reads KV cache from the BlockManager pool and reconstructs it into tensors.
    # Causes many copies and is inefficient; written for testing purposes.
    
    start_page_ids = cache_pos[:, 0] // self.page_size    # [B]
    start_page_offsets = cache_pos[:, 0] % self.page_size # [B]
    
    for seq_id in seq_ids:
      block_ids = self.block_table[(seq_id, layer_idx)]

  def iter_page_index(self, seq_ids: List[SeqId], layer_idx: int, cache_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    pass

  def free(self, seq_ids: List[SeqId], layer_idx: int):
    for seq_id in seq_ids:
      block_ids = self.block_table[(seq_id, layer_idx)]
      self.block_manager.free(layer_idx, block_ids)
