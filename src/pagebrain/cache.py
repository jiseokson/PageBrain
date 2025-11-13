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

  def update(
    self,
    seq_ids: List[SeqId],
    layer_idx: int,
    k_curr: torch.Tensor,
    v_curr: torch.Tensor,
    input_pos: torch.Tensor
  ) -> None:
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
    first_page_ids = (input_pos[:, 0] // self.page_size).tolist()    # [B]
    first_page_offsets = (input_pos[:, 0] % self.page_size).tolist() # [B]

    logger.debug(f'first_page_ids:     {first_page_ids}')
    logger.debug(f'first_page_offsets: {first_page_offsets}')

    k_pool = self.block_manager.gpu_k_pool[layer_idx] # [num_blocks, num_heads, page_size, d_head]
    v_pool = self.block_manager.gpu_v_pool[layer_idx] # same as above
    input_tokens = input_pos[:, 1].tolist()
    for sample_idx, (seq_id, first_page_idx, first_page_offset, input_token) \
      in enumerate(zip(seq_ids, first_page_ids, first_page_offsets, input_tokens)):
      block_ids = self.block_table[(seq_id, layer_idx)]

      # The first page may already contain data; write into remaining space
      first_block_idx = block_ids[first_page_idx]
      src_idx = min(self.page_size - first_page_offset, input_token)
      first_page_end = min(self.page_size, first_page_offset + input_token)
      k_pool[first_block_idx, :, first_page_offset : first_page_end, :] = k_curr[sample_idx, :, :src_idx, :]
      v_pool[first_block_idx, :, first_page_offset : first_page_end, :] = v_curr[sample_idx, :, :src_idx, :]

      logger.debug(
        f'wrote kv_curr[{sample_idx}, :, :{src_idx}, :] '
        f'into BlockManager pool[{first_block_idx}, :, {first_page_offset}:{first_page_end}, :]'
      )

      # Fill subsequent pages repeatedly by page_size
      for block_idx in block_ids[first_page_idx+1:]:
        assert src_idx < input_token

        dst_end = min(self.page_size, input_token - src_idx)
        src_end = min(src_idx + self.page_size, input_token)
        k_pool[block_idx, :, :dst_end, :] = k_curr[sample_idx, :, src_idx : src_end, :]
        v_pool[block_idx, :, :dst_end, :] = v_curr[sample_idx, :, src_idx : src_end, :]
        logger.debug(
          f'wrote kv_curr[{sample_idx}, :, {src_idx}:{src_end}, :] '
          f'into BlockManager pool[{block_idx}, :, 0:{dst_end}, :]'
        )

        src_idx += self.page_size
      
  def iter_page(
    self,
    seq_ids: List[SeqId],
    layer_idx: int,
    cache_pos: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # Reads KV cache from the BlockManager pool and reconstructs it into tensors.
    # Causes many copies and is inefficient; written for testing purposes.
    
    start_page_ids = cache_pos[:, 0] // self.page_size    # [B]
    start_page_offsets = cache_pos[:, 0] % self.page_size # [B]
    
    for seq_id in seq_ids:
      block_ids = self.block_table[(seq_id, layer_idx)]

  def iter_page_index(
    self,
    seq_ids: List[SeqId],
    layer_idx: int,
    cache_pos: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    pass

  def free(self, seq_ids: List[SeqId], layer_idx: int):
    for seq_id in seq_ids:
      block_ids = self.block_table[(seq_id, layer_idx)]
      self.block_manager.free(layer_idx, block_ids)
