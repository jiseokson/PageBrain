import pytest


def pytest_addoption(parser):
  parser.addoption(
    '--use-seed', action='store_true', default=False,
    help='Enable fixed random seed for reproducibility'
  )

@pytest.fixture
def use_seed(request):
  return request.config.getoption('--use-seed')
