import heapq
import itertools
import logging
from typing import List, Optional

from pagebrain.sequence import Sequence, SequenceGroup


logger = logging.getLogger('uvicorn')


class Scheduler:
  def __init__(self, device):
    self.device = device

    self.seq_pool = []
    heapq.heapify(self.seq_pool)

    # !! Temporary workaround to avoid comparison operations !!
    # !! between Seq objects when remain_token is identical  !!
    self._counter = itertools.count()  

    self.MAX_SEQ = 128
    self.MAX_PREFILL_LEN = 128

  def add(self, seqs: List[Sequence]):
    # Currently uses the simplest strategy: prioritize sequences with fewer remaining tokens
    # !! New scheduling policies can be implemented here !!
    for seq in seqs:
      remain_tokens = seq.max_new_tokens - len(seq.gen_tokens)
      _cnt = next(self._counter)
      heapq.heappush(self.seq_pool, (remain_tokens, _cnt, seq))

  def schedule(self) -> SequenceGroup:
    seqs = []
    for _ in range(self.MAX_SEQ):
      try:
        _, _, seq = heapq.heappop(self.seq_pool)
        seqs.append(seq)
      except IndexError:
        break

      if seq.input_len == 0:
        seq.input_len = min(len(seq.token_buffer), self.MAX_PREFILL_LEN)

    if len(seqs) == 0:
      return None
    
    logger.info(f'Scheduler.schedule(): len(seqs)={len(seqs)}')

    return SequenceGroup(seqs, device=self.device)

  def update(self, seq_group: SequenceGroup) -> Optional[List[Sequence]]:
    # Compare the number of tokens generated so far with max_new_tokens
    # Decide whether to include the sequence back into the scheduler (continue generating)
    # or exclude it (stop generation). For sequences determined to stop, set done=True.
    done_seqs = []
    gen_seqs = []
    for seq in seq_group.seqs:
      if len(seq.gen_tokens) == seq.max_new_tokens:
        seq.done = True
        done_seqs.append(seq)
      else:
        gen_seqs.append(seq)

    self.add(gen_seqs)
    return done_seqs
