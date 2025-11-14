from collections import defaultdict
import logging
from typing import Generator, List, Tuple

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
      if input_token == 0: continue

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
  ) -> Generator[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reads KV cache from the BlockManager pool and reconstructs it into tensors.
    # Causes many copies and is inefficient; written for testing purposes.

    logger.debug(
      f'entered iter_page(\n'
      f'  seq_ids={[seq_id[:5] for seq_id in seq_ids]},\n'
      f'  layer_idx={layer_idx}, cache_pos={cache_pos.tolist()})'
    )

    # Compute index range of target pages to read
    first_page_ids = cache_pos[:, 0] // self.page_size                        # [B]
    last_page_ids = (cache_pos[:, 0] + cache_pos[:, 1] - 1) // self.page_size # [B]
    # last_page_ids[last_page_ids == -1] = 0
    min_page_idx = torch.min(first_page_ids).item()
    max_page_idx = torch.max(last_page_ids).item()

    logger.debug(f'first_page_ids: {first_page_ids}')
    logger.debug(f'last_page_ids:  {last_page_ids}')
    logger.debug(f'target page index range: [{min_page_idx}, {max_page_idx}]')

    # Block index list of batch samples
    batch_block_ids = []
    for seq_id in seq_ids:
      block_ids = self.block_table[(seq_id, layer_idx)]
      batch_block_ids.append(block_ids)

    logger.debug(f'batch_block_ids: {batch_block_ids}')

    # Before reading, fetch cache spec information
    k_pool = self.block_manager.gpu_k_pool[layer_idx] # [num_blocks, num_heads, page_size, d_head]
    v_pool = self.block_manager.gpu_v_pool[layer_idx] # same as above
    num_heads = self.block_manager.num_heads
    # Set to BlockManager’s page_size when CacheManager is created, so both values remain identical.
    page_size = self.page_size
    d_head = self.block_manager.d_head
    device = self.block_manager.device
    kv_dtype = self.block_manager.dtype

    for page_idx in range(min_page_idx, max_page_idx + 1):
      read_mask = (first_page_ids <= page_idx) & (page_idx <= last_page_ids) # [B]
      read_block_ids = []
      for read_flag, block_ids in zip(read_mask.tolist(), batch_block_ids):
        read_block_ids.append(block_ids[page_idx] if read_flag else None)

      logger.debug(f'page_idx={page_idx} - read_mask={read_mask.tolist()}')
      logger.debug(f'page_idx={page_idx} - read_block_ids={read_block_ids}')

      # List of [H, P, D] shaped tensors
      k_pages = [
        k_pool[block_idx, :, :, :] if block_idx is not None
        else torch.empty([num_heads, page_size, d_head], device=device, dtype=kv_dtype)
        for block_idx in read_block_ids
      ]
      v_pages = [
        v_pool[block_idx, :, :, :] if block_idx is not None
        else torch.empty([num_heads, page_size, d_head], device=device, dtype=kv_dtype)
        for block_idx in read_block_ids
      ]

      k_p = torch.stack(k_pages) # [B, H, P, D]
      v_p = torch.stack(v_pages) # [B, H, P, D]

      logger.debug(f'KV page shape: {k_p.shape}')

      page_mask = None # [B, 2]

      yield k_p, v_p, page_mask

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
