import argparse
from contextlib import asynccontextmanager
import json
import logging
from typing import Literal, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from pagebrain.endpoints.logo import logo_ascii_art
from pagebrain.engine import Engine, EngineRequest


logger = logging.getLogger('uvicorn')


class GenerationRequest(BaseModel):
  prompt: str
  max_new_tokens: int = Field(1, ge=1)
  method: Literal['greedy', 'top_p', 'top_k'] = 'top_p'
  temperature: float = Field(1.0, gt=0.0)
  top_p: Optional[float] = None
  top_k: Optional[float] = None


app = FastAPI()
engine: Engine

@app.post("/generate")
async def generate(request: GenerationRequest):
  engine_request = EngineRequest(
    prompt=request.prompt,
    max_new_tokens=request.max_new_tokens,
    method=request.method,
    temperature=request.temperature,
    top_p = 0.9 if request.top_p is None else request.top_p,
    top_k = 50  if request.top_k is None else request.top_k,
  )
  logger.info(f'A new generation request has arrived: {engine_request}')

  iterator = engine.add_request(engine_request)

  return StreamingResponse(iterator)


def main():
  parser = argparse.ArgumentParser(
    description=logo_ascii_art,
    formatter_class=argparse.RawDescriptionHelpFormatter
  )

  parser.add_argument(
    '--host', type=str, default='0.0.0.0',
    help='Host address',
  )
  parser.add_argument(
    '--port', type=int, default=8000,
    help='Port number',
  )

  parser.add_argument(
    '--model', type=str, default='openai-community/gpt2',
    help='Name of the language model to load (HuggingFace model ID)'
  )
  parser.add_argument(
    '--device', type=str, default='cuda',
    help='Device to run the model on'
  )
  parser.add_argument(
    '--dtype', type=str, default='torch.float32',
    help='Data type used for storing KV cache'
  )

  args = parser.parse_args()

  global engine
  engine = Engine(
    model_name=args.model,
    device=args.device,
    kv_dtype=args.dtype,
  )

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    engine.start()
    yield

  app.router.lifespan_context = lifespan

  print(logo_ascii_art)
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
  main()
