import argparse

from pagebrain.endpoints.logo import logo_ascii_art


def get_args_parser() -> argparse.ArgumentParser:
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
    '--device', type=str, default='cuda:0',
    help='Device to run the model on'
  )

  return parser
