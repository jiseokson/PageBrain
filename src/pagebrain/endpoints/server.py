import argparse
from contextlib import asynccontextmanager
import json
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


class GenerationRequest(BaseModel):
  prompt: str
  max_new_tokens: int
  temperature: float = 1.0
  top_p: float = None
  top_k: float = None


@asynccontextmanager
async def lifespan(app):
  yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: GenerationRequest):
  req_id = str(uuid.uuid4().hex)

  async def iter_line(req_id):
    start_event = {
      'event': 'start',
      'request_id': req_id,
    }
    yield 'data: ' + json.dumps(start_event, ensure_ascii=False) + '\n\n'

    token_event = {
      'event': 'token',
      'request_id': req_id,
      'text': "Welcome to PageBrain!",
    }
    yield 'data: ' + json.dumps(token_event, ensure_ascii=False) + '\n\n'

    end_event = {
      'event': 'end',
      'request_id': req_id,
    }
    yield 'data: ' + json.dumps(end_event, ensure_ascii=False) + '\n\n'

  return StreamingResponse(iter_line(req_id))


def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

  args = parser.parse_args()
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
  main()
