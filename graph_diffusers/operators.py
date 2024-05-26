"""Graph Diffusion operators which can be pre-computed."""

import numpy as np
from typing import TYPE_CHECKING, Iterator, Callable, Union, Literal
import graphblas as gb
from graph_diffusers._utils import eval_expr, transpose_if, recover_triad_count
import functools as ft

if TYPE_CHECKING:
    from graph_diffusers._typing import _GraphblasModule as gb
    from numpy.typing import DTypeLike


@ft.lru_cache(maxsize=1)
def rw_norm(adj: gb.Matrix, *, copy: bool = True) -> gb.Matrix:
    return eval_expr(adj / adj.reduce_rowwise(), out=None if copy else adj)


@ft.lru_cache(maxsize=1)
def gcn_norm(adj: gb.Matrix, *, copy: bool = True) -> gb.Matrix:
    if copy:
        adj = adj.dup()
    eye = gb.Vector.from_scalar(1.0, adj.shape[0], dtype=adj.dtype).diag()
    adj(mask=gb.select.diag(adj).S) << adj + eye
    sqrt_deg = adj.reduce_rowwise().apply(gb.unary.sqrt)
    return eval_expr(sqrt_deg * adj * sqrt_deg, adj)


def norm(adj: gb.Matrix, method: Union[Literal["gcn"], Literal["rw"]], *, copy: bool = True) -> gb.Matrix:
    if method == "rw":
        return rw_norm(adj, copy=copy)
    return gcn_norm(adj, copy=copy)


def simple(adj: gb.Matrix) -> Callable[[np.ndarray], np.ndarray]:
    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x_hat = gb.Vector.from_dense(x, dtype=X.dtype)
            x_hat << adj @ x_hat
            X_hat.append(x_hat.to_dense(dtype=X.dtype, fill_value=0))
        return np.array(X_hat, dtype=X.dtype).T

    return diffuse_fn


def power(adj: gb.Matrix, k: int) -> Callable[[np.ndarray], np.ndarray]:
    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x_hat = gb.Vector.from_dense(x, dtype=X.dtype)
            for _ in range(k):
                x_hat << adj @ x_hat
            X_hat.append(x_hat.to_dense(dtype=X.dtype, fill_value=0))
        return np.array(X_hat, dtype=X.dtype).T

    return diffuse_fn


def appnp(adj: gb.Matrix, alpha: float = 0.15, iterations: int = 50) -> Callable[[np.ndarray], np.ndarray]:
    beta = 1 - alpha

    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x = gb.Vector.from_dense(x, dtype=X.dtype)
            x_hat = x.dup()
            for _ in range(iterations):
                x_hat << beta * (adj @ x_hat) + alpha * x

            X_hat.append(x_hat.to_dense(dtype=X.dtype))
        return np.array(X_hat).T

    return diffuse_fn


def triangle(adj: gb.Matrix, directed: bool = True, intermediate_int_type: "DTypeLike" = np.int32) -> gb.Matrix:
    A = gb.unary.one(gb.select.offdiag(adj)).new(dtype=intermediate_int_type)
    # could be better to transpose A
    A(mask=gb.select.offdiag(A).S, accum=gb.binary.plus) << transpose_if(directed, (-2 * A @ A))
    res = gb.select.valuegt(recover_triad_count(A)).new(dtype=adj.dtype)
    return res


def diffuse_powers(diffuse: Callable[[np.ndarray], np.ndarray], X: np.ndarray, k: int) -> Iterator[np.ndarray]:
    for _ in range(k):
        X = diffuse(X)
        yield X
