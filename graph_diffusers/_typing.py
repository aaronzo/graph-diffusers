from typing import Protocol
from types import ModuleType

import graphblas.binary
import graphblas.core.matrix
import graphblas.core.vector
import graphblas.io
import graphblas.select
import graphblas.unary


class _GraphblasModule(Protocol):
    def __new__(cls) -> "_GraphblasModule":
        raise NotImplementedError

    class Matrix(graphblas.core.matrix.Matrix, Protocol): ...

    class Vector(graphblas.core.vector.Vector, Protocol): ...

    unary: ModuleType = graphblas.unary
    binary: ModuleType = graphblas.binary
    select: ModuleType = graphblas.select
    io: ModuleType = graphblas.io
