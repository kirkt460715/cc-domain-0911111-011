"""First-order GUM uncertainty propagation.

Sensitivity coefficients are obtained with forward-mode automatic
differentiation (dual numbers), so they are exact partial derivatives of the
supported arithmetic at the evaluation point rather than finite-difference
approximations.  The combined variance is

    u_c^2 = c^T U c

where the covariance matrix ``U`` is assembled from documented random
uncertainties, shared systematic effects and explicit correlation entries.
"""

import ast
from dataclasses import dataclass
import decimal
from decimal import Decimal, localcontext

from ..errors import InputError
from ..values import number, decimal_text
from . import dimensions as dims


@dataclass(frozen=True)
class Dual:
    value: Decimal
    partials: tuple


def _add(left, right):
    return Dual(
        left.value + right.value,
        tuple(a + b for a, b in zip(left.partials, right.partials)),
    )


def _sub(left, right):
    return Dual(
        left.value - right.value,
        tuple(a - b for a, b in zip(left.partials, right.partials)),
    )


def _mul(left, right):
    return Dual(
        left.value * right.value,
        tuple(
            a * right.value + b * left.value
            for a, b in zip(left.partials, right.partials)
        ),
    )


def _div(left, right):
    return Dual(
        left.value / right.value,
        tuple(
            a / right.value - b * left.value / (right.value * right.value)
            for a, b in zip(left.partials, right.partials)
        ),
    )


def _pow(base, exponent):
    power = int(exponent)
    if power == 0:
        return Dual(Decimal(1), tuple(Decimal(0) for _ in base.partials))
    value = base.value**power
    factor = Decimal(power) * base.value ** (power - 1)
    return Dual(value, tuple(factor * partial for partial in base.partials))


def evaluate_dual(tree, names):
    """Evaluate an AST node carrying one partial per named input."""

    def walk(node):
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.Constant):
            return Dual(
                Decimal(str(node.value)),
                tuple(Decimal(0) for _ in next(iter(names.values())).partials),
            )
        if isinstance(node, ast.UnaryOp):
            operand = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return Dual(-operand.value, tuple(-p for p in operand.partials))
            return operand
        left = walk(node.left)
        right = walk(node.right)
        if isinstance(node.op, ast.Add):
            return _add(left, right)
        if isinstance(node.op, ast.Sub):
            return _sub(left, right)
        if isinstance(node.op, ast.Mult):
            return _mul(left, right)
        if isinstance(node.op, ast.Div):
            return _div(left, right)
        if isinstance(node.op, ast.Pow):
            return _pow(left, _literal_exponent(node.right))
        raise InputError("unsupported expression operator")

    with localcontext() as ctx:
        ctx.prec = 40
        return walk(tree)


def _literal_exponent(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -int(node.operand.value)
    return int(node.value)


def validate_symmetric(matrix, size, *, field="covariance"):
    tolerance = Decimal("1e-30")
    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                raise InputError(
                    "covariance matrix is not symmetric",
                    field=field,
                    details={"row": i, "column": j},
                )


def validate_psd(matrix, labels, *, field="covariance"):
    """Reject matrices that are not positive semi-definite via LDL factorisation."""

    size = len(matrix)
    work = [row[:] for row in matrix]
    tolerance = Decimal("-1e-24")
    with localcontext() as ctx:
        ctx.prec = 40
        for j in range(size):
            diagonal = work[j][j]
            if diagonal < tolerance:
                raise InputError(
                    "covariance matrix is not positive semi-definite",
                    field=field,
                    details={"input": labels[j], "diagonal": decimal_text(diagonal)},
                )
            if diagonal <= 0:
                for k in range(j + 1, size):
                    if work[k][j] != 0:
                        raise InputError(
                            "covariance matrix is not positive semi-definite",
                            field=field,
                            details={"input": labels[j]},
                        )
                continue
            for k in range(j + 1, size):
                work[k][j] = work[j][k] / diagonal
            for k in range(j + 1, size):
                factor = work[k][j]
                for m in range(j + 1, k + 1):
                    work[k][m] -= factor * work[j][m]
        for i in range(size):
            if work[i][i] < tolerance:
                raise InputError(
                    "covariance matrix is not positive semi-definite",
                    field=field,
                    details={"input": labels[i]},
                )


def parse_matrix(raw, labels, unit_vectors, field="covariance"):
    size = len(labels)
    if not isinstance(raw, list) or len(raw) != size or any(
        not isinstance(row, list) or len(row) != size for row in raw
    ):
        raise InputError(
            "covariance matrix must be a square matrix matching the inputs",
            field=field,
        )
    matrix = []
    for i, row in enumerate(raw):
        parsed = []
        for j, cell in enumerate(row):
            value = number(cell, f"{field}[{i}][{j}]", nonnegative=False)
            if not value.is_finite():
                raise InputError("covariance entry must be finite", field=field)
            parsed.append(value)
        matrix.append(parsed)
    validate_symmetric(matrix, size, field=field)
    for i in range(size):
        if matrix[i][i] < 0:
            raise InputError(
                "covariance diagonal cannot be negative",
                field=field,
                details={"input": labels[i]},
            )
    # Covariance only exists between quantities of matching dimension; a
    # different unit is allowed after conversion, a different dimension is not.
    for i in range(size):
        for j in range(size):
            if matrix[i][j] != 0 and not dims.equal(
                unit_vectors[i], unit_vectors[j]
            ):
                raise InputError(
                    "covariance entries require matching dimensions",
                    field=field,
                    details={
                        "left": labels[i],
                        "right": labels[j],
                        "left_dimension": dims.describe(unit_vectors[i]),
                        "right_dimension": dims.describe(unit_vectors[j]),
                    },
                )
    validate_psd(matrix, labels, field=field)
    return matrix


def sensitivity(tree, variables, values):
    duals = {
        name: Dual(
            values[name],
            tuple(Decimal(1) if index == position else Decimal(0)
                  for index in range(len(variables))),
        )
        for position, name in enumerate(variables)
    }
    try:
        result = evaluate_dual(tree, duals)
    except (decimal.DivisionByZero, decimal.InvalidOperation) as exc:
        raise InputError(
            "expression cannot be evaluated at the documented input values",
            details={"reason": type(exc).__name__},
        ) from exc
    return result.value, dict(zip(variables, result.partials))


def combined_variance(coefficients, covariance):
    with localcontext() as ctx:
        ctx.prec = 40
        total = Decimal(0)
        size = len(coefficients)
        for i in range(size):
            for j in range(size):
                total += coefficients[i] * coefficients[j] * covariance[i][j]
        if total < 0 and total > Decimal("-1e-24"):
            total = Decimal(0)
        if total < 0:
            raise InputError("combined variance is negative")
        return total
