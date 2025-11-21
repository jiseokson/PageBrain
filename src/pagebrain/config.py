from argparse import Namespace

import torch


class PageBrainConfig:
  def __init__(self, args: Namespace):
    # Engine
    self.MAX_FETCH_REQ = 64
    # Scheduler
    self.MAX_SEQ = 256
    self.MAX_PREFILL_LEN = 128

    self.host = args.host
    self.port = args.port

    self.model_name = args.model
    self.device = args.device
    self.dtype = torch.float32

    # Set during the Engine._init() call
    self.base_model_config = None

    # Set during the Engine._init() call
    self.num_blocks: int = None
    self.num_layers: int = None
    self.num_heads: int = None
    self.d_head: int = None
    self.page_size: int = 32
    self.kv_dtype: torch.dtype = torch.float32
