"""Fixed-seed Monte Carlo cross-check for nonlinear models.

The first-order GUM result can be unreliable for strongly nonlinear
expressions.  When requested, the documented covariance matrix is drawn from
a multivariate normal distribution using its LDL factorisation and a
deterministic :class:`random.Random` instance seeded by the request.  Every
draw is re-evaluated through the same expression; failures (division by zero,
non-finite results, negative bases under integer powers) are counted as
invalid samples rather than silently dropped.  The run records the seed,
distribution summary, invalid sample count and the difference from the GUM
result, and never calls the result a confidence level.
"""

import math
import random
from decimal import Decimal, localcontext

from ..errors import InputError
from ..values import integer, number
from .gum import Dual, evaluate_dual


class SamplingCheck:
    def __init__(self, model, tree, variables, working_values, covariance, seed):
        self.model = model
        self.tree = tree
        self.variables = variables
        self.values = working_values
        self.covariance = covariance
        self.seed = integer(seed, "seed", 0, 2**32 - 1)
        self._factor = self._ldl(covariance)

    @staticmethod
    def _ldl(matrix):
        size = len(matrix)
        work = [
            [float(cell) if cell != 0 else 0.0 for cell in row] for row in matrix
        ]
        diagonals = []
        for j in range(size):
            diagonal = work[j][j]
            if diagonal < -1e-18:
                raise InputError(
                    "sampling requires a positive semi-definite covariance matrix"
                )
            if diagonal <= 0.0:
                diagonals.append(0.0)
                for k in range(j + 1, size):
                    work[k][j] = 0.0
                continue
            diagonals.append(diagonal)
            for k in range(j + 1, size):
                work[k][j] = work[j][k] / diagonal
            for k in range(j + 1, size):
                factor = work[k][j]
                for m in range(j + 1, k + 1):
                    work[k][m] -= factor * work[j][m]
        return work, diagonals

    def _draw(self, rng):
        size = len(self.variables)
        z = [rng.gauss(0.0, 1.0) for _ in range(size)]
        y = [0.0] * size
        lower, diagonals = self._factor
        for i in range(size):
            if diagonals[i] <= 0.0:
                y[i] = 0.0
                continue
            total = z[i] * math.sqrt(diagonals[i])
            for j in range(i):
                total += lower[i][j] * y[j]
            y[i] = total
        return [
            self.values[name] + Decimal(repr(shift))
            for name, shift in zip(self.variables, y)
        ]

    def run(self, sample_count):
        integer(sample_count, "samples", 10, 1_000_000)
        rng = random.Random(self.seed)
        samples = []
        invalid = []
        zero_partials = tuple(Decimal(0) for _ in self.variables)
        for index in range(sample_count):
            drawn = self._draw(rng)
            names = {
                name: Dual(value, zero_partials)
                for name, value in zip(self.variables, drawn)
            }
            try:
                result = evaluate_dual(self.tree, names).value
                if not result.is_finite():
                    raise ValueError("non-finite sample result")
            except (ArithmeticError, ValueError, OverflowError) as exc:
                invalid.append({"index": index, "reason": type(exc).__name__})
                continue
            samples.append(float(result))
        if not samples:
            raise InputError("every sampled draw was invalid")
        samples.sort()
        return {
            "seed": self.seed,
            "requested_samples": sample_count,
            "valid_samples": len(samples),
            "invalid_samples": len(invalid),
            "distribution": {
                "mean": _decimal(mean(samples)),
                "standard_deviation": _decimal(standard_deviation(samples)),
                "median": _decimal(percentile(samples, 50)),
                "p02_5": _decimal(percentile(samples, 2.5)),
                "p97_5": _decimal(percentile(samples, 97.5)),
                "minimum": _decimal(samples[0]),
                "maximum": _decimal(samples[-1]),
            },
            "invalid": invalid[:20],
        }


def mean(values):
    return sum(values) / len(values)


def standard_deviation(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(
        sum((value - average) ** 2 for value in values) / (len(values) - 1)
    )


def percentile(sorted_values, percent):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (percent / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def _decimal(value):
    with localcontext() as ctx:
        ctx.prec = 40
        return format(Decimal(repr(value)), "f")


def compare(gum_value, gum_uncertainty, sampling_summary):
    """Document the discrepancy without asserting statistical coverage."""

    simulated = number(sampling_summary["distribution"]["standard_deviation"])
    reference = gum_uncertainty
    if reference == 0:
        relative = None if simulated == 0 else Decimal("Infinity")
    else:
        relative = abs(simulated - reference) / reference
    value_shift = abs(
        number(sampling_summary["distribution"]["mean"]) - gum_value
    )
    return {
        "gum_standard_uncertainty": _text(reference),
        "sampling_standard_deviation": sampling_summary["distribution"][
            "standard_deviation"
        ],
        "absolute_uncertainty_difference": _text(abs(simulated - reference)),
        "relative_uncertainty_difference": _text(relative)
        if relative is not None
        else None,
        "mean_shift": _text(value_shift),
        "review_threshold_exceeded": (
            relative is not None and relative > Decimal(".1")
        ),
    }


def _text(value):
    if value is None:
        return None
    with localcontext() as ctx:
        ctx.prec = 40
        return format(value, "f")
