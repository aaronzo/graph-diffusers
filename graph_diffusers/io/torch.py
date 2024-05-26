import graphblas as gb
from typing import TYPE_CHECKING, Optional, Hashable
import functools as ft
import torch
import warnings
from graph_diffusers import _extras

if TYPE_CHECKING:
    from graph_diffusers._typing import _GraphblasModule as gb
    from numpy.typing import DTypeLike

if _extras.TORCH_SPARSE:
    from torch_sparse import SparseTensor as _SparseTensor
else:

    class _SparseTensor: ...


def torch_to_graphblas(
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    weighted: bool = False,
    dtype: "Optional[DTypeLike]" = None,
) -> gb.Matrix:
    if isinstance(edge_index, _SparseTensor):
        return torch_sparse_tensor_to_graphblas(edge_index, weighted=weighted, dtype=dtype)
    if edge_index.is_sparse_csr:
        return torch_sparse_csr_to_graphblas(edge_index, weighted=weighted, dtype=dtype)
    return torch_edge_index_to_graphblas(edge_index, num_nodes=num_nodes, dtype=dtype)


def torch_sparse_csr_to_graphblas(
    adj_t: torch.Tensor, *, weighted: bool = False, dtype: "Optional[DTypeLike]" = None
) -> gb.Matrix:
    if not isinstance(dtype, Hashable):
        warnings.warn(
            f"Unhashable dtype {dtype} passed when converting from torch to graphblas." "The result will not be cached."
        )
        return _torch_edge_index_to_graphblas.__wrapped__(adj_t, weighted=weighted, dtype=dtype)
    return _torch_sparse_csr_to_graphblas(adj_t, weighted=weighted, dtype=dtype)


def torch_edge_index_to_graphblas(
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    dtype: "Optional[DTypeLike]" = None,
) -> gb.Matrix:
    if not isinstance(dtype, Hashable):
        warnings.warn(
            f"Unhashable dtype {dtype} passed when converting from torch to graphblas. The result will not be cached."
        )
        return _torch_edge_index_to_graphblas.__wrapped__(edge_index, num_nodes=num_nodes, dtype=dtype)
    return _torch_edge_index_to_graphblas(edge_index, num_nodes=num_nodes, dtype=dtype)


if _extras.TORCH_SPARSE:
    import torch_sparse

    def torch_sparse_tensor_to_graphblas(
        adj_t: torch_sparse.SparseTensor, *, weighted: bool = False, dtype: "Optional[DTypeLike]" = None
    ) -> gb.Matrix:
        return torch_sparse_csr_to_graphblas(
            adj_t.to_torch_sparse_csr_tensor(),
            weighted=weighted,
            dtype=dtype,
        )


@ft.lru_cache(maxsize=1)
def _torch_sparse_csr_to_graphblas(
    adj_t: torch.Tensor,
    weighted: bool,
    dtype: "Optional[DTypeLike]",
) -> gb.Matrix:
    if not adj_t.is_sparse_csr:
        adj_t = adj_t.to_sparse_csr()
    return gb.Matrix.from_csr(
        indptr=adj_t.crow_indices().detach().cpu().numpy(),
        col_indices=adj_t.col_indices().detach().cpu().numpy(),
        values=1.0 if not weighted else adj_t.values().detach().cpu().numpy(),
        nrows=adj_t.shape[0],
        ncols=adj_t.shape[0],
        dtype=dtype,
    )


@ft.lru_cache(maxsize=1)
def _torch_edge_index_to_graphblas(
    edge_index: torch.Tensor,
    num_nodes: Optional[int],
    dtype: "Optional[DTypeLike]",
) -> gb.Matrix:
    return gb.Matrix.from_coo(*edge_index, dtype=dtype, nrows=num_nodes, ncols=num_nodes)
