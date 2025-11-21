from contextlib import asynccontextmanager
import logging
from typing import Literal, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from pagebrain.config import PageBrainConfig
from pagebrain.endpoints.args import get_args_parser
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

  iterator = engine.add_request(engine_request)

  return StreamingResponse(iterator)


def main():
  parser = get_args_parser()
  args = parser.parse_args()
  pagebrain_config = PageBrainConfig(args)

  global engine
  engine = Engine(config=pagebrain_config)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    engine.start()
    yield

  app.router.lifespan_context = lifespan

  print(logo_ascii_art)
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
  main()
