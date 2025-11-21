import asyncio
import shutil
import json
import time
import random
import statistics
from typing import Dict, List, Tuple, Optional

import aiohttp
import pandas as pd


URL = 'http://localhost:8000/generate'
RESULTS_CSV = 'benchmark_results.csv'

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
] * 10

num_prompt_tokens = [8, 6, 6, 7, 6, 6, 5, 5, 6, 6] * 10



ESC = '\x1b['
BASE_ROW = 5
TERM_WIDTH, TERM_HEIGHT = shutil.get_terminal_size(fallback=(120, 40))


def build_payload(prompt: str) -> Dict:
  max_new_tokens = random.randint(20, 50)

  method = random.choice(['greedy', 'top_p', 'top_k'])
  temperature = round(random.uniform(0.7, 1.3), 2)

  payload: Dict = {
      'prompt': prompt,
      'max_new_tokens': max_new_tokens,
      'method': method,
      'temperature': temperature,
  }

  if method == 'top_p':
    payload['top_p'] = round(random.uniform(0.8, 0.98), 2)
  elif method == 'top_k':
    payload['top_k'] = random.choice([16, 32, 40, 64, 80])

  return payload


def clear_screen():
  print(ESC + '2J' + ESC + 'H', end='')


def move_cursor(row: int, col: int = 0):
  print(f'{ESC}{row};{col}H', end='')


def clear_line():
  print(ESC + '2K', end='')


async def update_ui_line(
  idx: int,
  text: str,
  payload: Dict,
  lock: asyncio.Lock,
):
  row = BASE_ROW + idx
  header = f'[{idx}] {payload["method"]:>6} T={payload["temperature"]:.2f}'

  if payload['method'] == 'top_p':
    header += f' p={payload.get("top_p"):.2f}'
  elif payload['method'] == 'top_k':
    header += f' k={payload.get("top_k")}'

  prompt_preview = payload['prompt'][:30].replace('\n', ' ')
  header += f' | \'{prompt_preview}...\' -> '

  line = header + text.replace('\n', ' ')

  max_width = TERM_WIDTH - 1
  if len(line) > max_width:
    line = line[: max_width - 3] + '...'

  async with lock:
    move_cursor(row, 0)
    clear_line()
    print(line, end='', flush=True)


async def fetch_stream(
  session: aiohttp.ClientSession,
  payload: Dict,
  idx: int,
  ui_lock: asyncio.Lock,
) -> Tuple[List[str], Dict, Optional[float]]:
  tokens: List[str] = []
  start = time.perf_counter()
  generated_text = ''

  try:
    async with session.post(URL, json=payload) as resp:
      async for raw in resp.content:
        decoded = raw.decode().strip()
        if not decoded:
          continue

        if not decoded.startswith('data:'):
          continue

        data_str = decoded[len('data:'):].strip()

        if data_str == '[DONE]':
          break

        try:
          obj = json.loads(data_str)
        except json.JSONDecodeError:
          continue

        event = obj.get('event')
        if event == 'token':
          text = obj.get('text', '')
          tokens.append(text)
          generated_text += text
          await update_ui_line(idx, generated_text, payload, ui_lock)
        elif event == 'start':
          pass
        elif event == 'end':
          pass

  except Exception as e:
    async with ui_lock:
      row = BASE_ROW + idx
      move_cursor(row, 0)
      clear_line()
      print(f'[{idx}] ❌ Request failed: {e}', end='', flush=True)
    return [], payload, None

  end = time.perf_counter()
  latency = end - start

  await update_ui_line(idx, generated_text + '  (DONE)', payload, ui_lock)

  return tokens, payload, latency


async def main():
    clear_screen()
    print('=== LLM Streaming Benchmark ===')
    print(f'Target URL: {URL}')
    print(f'Total Requests: {len(prompts)}')
    print()
    print('Streaming responses:')

    payloads = [build_payload(p) for p in prompts]

    for i in range(len(prompts)):
      print()

    ui_lock = asyncio.Lock()

    start_time = time.perf_counter()

    connector = aiohttp.TCPConnector(
      limit=0
    )
    async with aiohttp.ClientSession(connector=connector) as session:
      tasks = [
        asyncio.create_task(fetch_stream(session, payload, idx, ui_lock))
        for idx, payload in enumerate(payloads)
      ]
      results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    summary_row = BASE_ROW + len(prompts) + 2
    move_cursor(summary_row, 0)
    print()

    responses, used_payloads, latencies = zip(*results)

    total_generated_tokens = 0
    print('\n=== Per-request summary ===')
    for idx, (tokens, payload, n_prompt_tokens, latency) in enumerate(
      zip(responses, used_payloads, num_prompt_tokens, latencies)
    ):
      received_tokens = len(tokens)
      total_generated_tokens += received_tokens

      print(f'\n--- Request {idx} ---')
      print(f'Prompt          : {payload["prompt"]!r}')
      print(f'Method          : {payload["method"]}')
      print(f'Temperature     : {payload["temperature"]}')
      if payload['method'] == 'top_p':
        print(f'top_p           : {payload.get("top_p")}')
      elif payload['method'] == 'top_k':
        print(f'top_k           : {payload.get("top_k")}')
      print(f'max_new_tokens  : {payload["max_new_tokens"]}')
      if latency is not None:
        print(f'Latency         : {latency:.3f}s')
      else:
        print('Latency         : FAILED')

      print(f'Prompt tokens   : {n_prompt_tokens}')
      print(f'Generated tokens: {received_tokens}')
      print('Generated text  : ', ''.join(tokens))

    valid_latencies = [l for l in latencies if l is not None]
    if valid_latencies:
      avg_latency = statistics.mean(valid_latencies)
      if len(valid_latencies) > 1:
        p95_latency = statistics.quantiles(valid_latencies, n=100)[94]
      else:
        p95_latency = avg_latency
    else:
      avg_latency = float('nan')
      p95_latency = float('nan')

    throughput_reqs = len(prompts) / elapsed if elapsed > 0 else float('nan')
    throughput_tokens = total_generated_tokens / elapsed if elapsed > 0 else float('nan')

    print('\n' + '=' * 60)
    print(f'Total Requests           : {len(prompts)}')
    print(f'Total Elapsed Time       : {elapsed:.3f}s')
    print(f'Average Latency          : {avg_latency:.3f}s')
    print(f'95th Percentile Latency  : {p95_latency:.3f}s')
    print(f'Throughput (req/s)       : {throughput_reqs:.3f}')
    print(f'Throughput (tokens/s)    : {throughput_tokens:.3f}')
    print('=' * 60)

    df = pd.DataFrame(
      {
        'throughput_req_per_s': [throughput_reqs],
        'throughput_tok_per_s': [throughput_tokens],
        'avg_latency': [avg_latency],
        'p95_latency': [p95_latency],
        'total_elapsed': [elapsed],
        'num_requests': [len(prompts)],
        'total_generated_tokens': [total_generated_tokens],
      }
    )
    df.to_csv(RESULTS_CSV, index=False)
    print(f'\nSaved results to: {RESULTS_CSV}')


if __name__ == '__main__':
    asyncio.run(main())
