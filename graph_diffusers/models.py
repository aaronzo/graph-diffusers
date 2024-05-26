import numpy as np
from typing import Union, Literal, TYPE_CHECKING
import graphblas as gb
from itertools import chain
from abc import ABCMeta, abstractmethod
import graph_diffusers.operators as op

if TYPE_CHECKING:
    from graph_diffusers._typing import _GraphblasModule as gb


class Diffusion(metaclass=ABCMeta):
    @abstractmethod
    def num_features(self, in_features: int) -> int: ...

    @abstractmethod
    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray: ...


class SimpleGCNDiffusion(Diffusion):
    def __init__(self, k: int) -> None:
        self.k = k

    def num_features(self, in_features: int) -> int:
        return in_features

    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        adj_gcn = op.gcn_norm(adj)
        return op.power(adj_gcn, self.k)(X)


class SIGNDiffusion(Diffusion):
    def __init__(
        self,
        s: int,
        p: int,
        t: int,
        s_norm: Union[Literal["gcn"], Literal["rw"]] = "gcn",
        p_norm: Union[Literal["gcn"], Literal["rw"]] = "rw",
        t_norm: Union[Literal["gcn"], Literal["rw"]] = "rw",
    ) -> None:
        self.s = s
        self.p = p
        self.t = t
        self.s_norm = s_norm
        self.p_norm = p_norm
        self.t_norm = t_norm
        if self.r < 1:
            raise ValueError

    @property
    def r(self) -> int:
        return self.s + self.p + self.t

    def num_features(self, in_features: int) -> int:
        return self.r * in_features

    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        ops = []
        if s := self.s:
            simple_diffuser = op.simple(op.norm(adj, method=self.s_norm))
            ops.append(op.diffuse_powers(simple_diffuser, X, s))
        if p := self.p:
            ppr_diffuser = op.appnp(op.norm(adj, method=self.p_norm))
            ops.append(op.diffuse_powers(ppr_diffuser, X, p))
        if t := self.t:
            adj_triangle = op.triangle(adj, directed=False)
            triangle_diffuser = op.simple(op.norm(adj_triangle, method=self.t_norm, copy=False))
            ops.append(op.diffuse_powers(triangle_diffuser, X, t))

        return np.hstack(tuple(chain(*ops)), dtype=X.dtype)
