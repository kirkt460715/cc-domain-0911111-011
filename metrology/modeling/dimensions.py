"""Dimension vectors for measurement-model expressions.

A dimension is represented as a mapping of base dimension names to integer
exponents.  Dimensionless quantities use an empty mapping.  Registry units
carry a single ``dimension`` label; the vector of an input is therefore a
single-entry mapping, and derived dimensions produced by multiplication,
division and integer powers stay expressible with integer exponents.
"""

from ..errors import InputError

MAX_EXPONENT = 12


def normalize(vector):
    return {name: exponent for name, exponent in vector.items() if exponent != 0}


def combine(left, right, sign):
    result = dict(left)
    for name, exponent in right.items():
        result[name] = result.get(name, 0) + sign * exponent
    return normalize(result)


def product(left, right):
    return combine(left, right, 1)


def quotient(left, right):
    return combine(left, right, -1)


def power(vector, exponent):
    integer = int(exponent)
    if abs(integer) > MAX_EXPONENT:
        raise InputError(
            "integer power exponent outside supported range",
            details={"exponent": integer, "maximum": MAX_EXPONENT},
        )
    return normalize({name: value * integer for name, value in vector.items()})


def single(name):
    return normalize({name: 1})


def dimensionless(vector):
    return not vector


def require_additive(left, right):
    if normalize(left) != normalize(right):
        raise InputError(
            "dimensions must match inside sums and differences",
            details={"left": describe(left), "right": describe(right)},
        )


def describe(vector):
    vector = normalize(vector)
    if not vector:
        return "1"
    parts = []
    for name in sorted(vector):
        exponent = vector[name]
        parts.append(name if exponent == 1 else f"{name}^{exponent}")
    return "*".join(parts)


def equal(left, right):
    return normalize(left) == normalize(right)
