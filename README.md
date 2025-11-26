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
    <img src="https://github.com/user-attachments/assets/7cda546e-7401-4c07-9f4b-22cc3f995676" alt="PagedAttention" width="70%">
    <p>Fig.2 - Illustraion of PagedAttention</p>
</div>

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
from pagebrain.cache import CacheManager

# Load tokenizer & base model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
base = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Initialize cache manager for paged KV memory
block_manager = BlockManager(config)
cache = CacheManager(block_manager, config)

# Wrap the model with PageBrain's PagedAttention-enabled module
model = PagedGPT2LMHeadModel(base, cache).eval()

# Example inference
inputs = tokenizer("Hello, PageBrain!", return_tensors="pt")
outputs = model.(inputs['input_ids'], input_pos=..., cache_pos=...)
```
