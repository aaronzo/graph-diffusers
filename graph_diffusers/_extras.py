from importlib.util import find_spec

TORCH = bool(find_spec("torch"))
TORCH_SPARSE = bool(find_spec("torch_sparse"))
