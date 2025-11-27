# PageBrain
PageBrain is a lightweight LLM serving system that implements PagedAttention and page-level KV cache management from scratch — enabling efficient memory usage, minimized fragmentation, and scalable multi-request inference on a single GPU.

This project was developed as a graduation capstone and continues to evolve into a personal research/engineering project. It aims to provide an educational reference implementation of modern LLM serving techniques, inspired by systems like vLLM but written to be readable, hackable, and extensible.

## ✨ Key Features
- 🚀 **Page-based KV Cache Memory System**\
  Fully custom logical–physical KV layout with block tables, mapping, and allocation.

- 🧩 **PagedAttention Module (Custom Implementation)**\
  No full-context reconstruction. Attends only to necessary blocks, accumulating scores across pages.

- 🧠 **HuggingFace Model Integration**\
  Works with GPT-2 (others planned). Drop-in replacement for standard attention.

- 🔄 **Prefill + Decode Step Unified Forward Loop**\
  Ensures identical results to vanilla HF models under matching conditions.

- 📦 **Multi-Request Scheduler**\
  Request queue → Sequence → SequenceGroup → Forward execution pipeline.

## 🏗 System Architecture

<div align="center">
    <img src="https://github.com/user-attachments/assets/c4c88e77-7bef-4947-8863-affadc851285" alt="System Architecture" width="60%">
    <p>Fig.1 - System architecture of PageBrain</p>
</div>

### Core Components
- **Engine** – Central loop that polls requests, schedules sequences, and triggers model forward passes.

- **Scheduler** – Converts user requests into sequences, groups them, and decides execution order.

- **SequenceGroup** – Batched group of active sequences for a single forward call.

- **Executor** – Runs the model with the provided SequenceGroup + CacheManager.

- **CacheManager / BlockManager** – Handles KV cache allocation, updates, lookup, and physical memory layouts.

- **PagedAttention module** – Computes attention using logical block indices instead of full KV tensors.

## 🔍 PagedAttention
<div align="center">
    <img src="https://github.com/user-attachments/assets/c45f9ba9-ea4f-4e44-90ce-e07fe84d24b8" alt="Attention" width="50%">
    <p>Fig.2 - Illustraion of attention</p>
</div>
Modern decoder-only Transformers rely on a repeated attention loop during text generation.
At every generation step, the model:

1. receives a new input sequence,
2. computes new q, k, v vectors for the final token,
3. attends to all previous key/value vectors,
4. and produces the next output token.

During this process, one important observation is that previous K/V vectors never change.
Thus, these intermediate results are typically cached and reused at every step to avoid recomputation.

### The Problem: Repeated Computation & Memory Fragmentation
Although KV caching is conceptually simple, implementing it efficiently on GPUs is not.
- Each layer stores massive K/V tensors.
- PyTorch allocates them contiguously on GPU memory.
- As generation progresses, the cache grows token-by-token, and eventually GPU memory becomes filled with non-contiguous free spaces.

Even if the GPU has enough total memory, it can still fail to allocate a continuous block large enough →
classic memory fragmentation problem.

### OS-Inspired Solution: Page-based Cache Management
PageBrain adopts the same idea used in operating systems:

> Introduce a virtual memory abstraction for KV cache
> → logical token indices are mapped to physical GPU blocks.

- GPU memory is divided into fixed-size blocks (“pages”).
- Each logical K/V position (layer × token index) maps to a page entry.
- A BlockTable manages all logical → physical mappings.
- New K/V entries no longer require contiguous GPU space.

This allows PageBrain to overcome fragmentation entirely while keeping allocation fast.

### The Core: Page-by-Page Attention Accumulation
<div align="center">
    <img src="https://github.com/user-attachments/assets/7cda546e-7401-4c07-9f4b-22cc3f995676" alt="PagedAttention" width="70%">
    <p>Fig.3 - Illustraion of PagedAttention</p>
</div>
With the page-based KV layout established, PageBrain implements a custom PagedAttention operator:

- K/V vectors are stored in fixed-size pages.
- During attention computation, the system iterates over pages, not the full dense tensor.
- Only the blocks relevant for the current sequence are read.
- Attention scores are accumulated incrementally as the pages are scanned.

This means:
- No reconstruction of the full KV tensor
- No need for contiguous memory
- Consistent attention scores thanks to correct masking + block indexing
- Efficient scaling for long sequences

The math shown in Fig.3 describes exactly this process:
accumulating `a_ij * v_j` page-by-page until the full context has been consumed.

## 🚀 Installation
> Python 3.10+ recommended
``` bash
git clone https://github.com/yourname/PageBrain.git
cd PageBrain
pip install -r requirements.txt
pip install -e .
```

## ▶️ Basic Usage
PageBrain can be used in two ways:
- (CLI Server) Launch the full PageBrain inference server with a single command.
- (Python API) Load the model and execute generation directly inside your own Python code.

### 1. Run the PageBrain Server (CLI)
PageBrain also provides a command-line interface for running a full LLM serving backend.
This mode launches:
- the HTTP API server
- the request scheduler
- KV cache manager
- PagedAttention-enabled model

... so you can send requests via REST API or curl.
  
```bash
pagebrain --model openai-community/gpt2
```

### 2. Programmatic Usage (Python API)
You can load a HuggingFace model, wrap it with PageBrain’s paged KV cache system, and run inference directly in Python.
This approach is ideal for research, experimentation, or embedding PageBrain into a larger application.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from pagebrain.models.gpt2 import PagedGPT2LMHeadModel
from pagebrain.block import BlockManager
from pagebrain.cache import CacheManager

# Load tokenizer & base model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
base = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Initialize cache manager for paged KV memory
block_manager = BlockManager(config)
cache_manager = CacheManager(block_manager, config)

# Wrap the model with PageBrain's PagedAttention-enabled module
model = PagedGPT2LMHeadModel(base, cache_manager).eval()

# Example inference
inputs = tokenizer("Hello, PageBrain!", return_tensors="pt")
outputs = model(inputs['input_ids'], input_pos=..., cache_pos=...)
```
