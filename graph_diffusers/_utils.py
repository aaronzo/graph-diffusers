import graphblas as gb
from typing import TYPE_CHECKING, Optional, Union, Callable, Any

if TYPE_CHECKING:
    from graph_diffusers._typing import _GraphblasModule as gb
    from graphblas.core.matrix import MatrixExpression, TransposedMatrix


def eval_expr(expr: "MatrixExpression", out: Optional[gb.Matrix] = None, **kw: Any) -> gb.Matrix:
    if out is None:
        return expr.new(**kw)
    out(**kw) << expr
    return out


def transpose_if(
    condition: bool,
    matrix: "Union[gb.Matrix, MatrixExpression]",
) -> "Union[MatrixExpression, MatrixExpression, TransposedMatrix]":
    return matrix.T if condition else matrix


def _recover_triad_count(x: int) -> float:
    return (1 - x) / 2 if x % 2 else x


recover_triad_count: Callable[[gb.Matrix], "MatrixExpression"] = gb.unary.register_anonymous(_recover_triad_count)
