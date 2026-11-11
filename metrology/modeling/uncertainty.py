"""Documented uncertainty budgets.

Every evaluation input keeps three independently documented uncertainty
sources:

* a random standard uncertainty (type-A style repeatability),
* links to shared systematic effects, which create covariance between every
  input affected by the same reference standard,
* an optional explicit covariance matrix.

All quantities are converted into deterministic working units (the unitary
unit of each dimension) before matrix assembly, so covariance entries always
carry consistent units.  Matrices are rejected when asymmetric, not positive
semi-definite or dimensionally incompatible.
"""

from dataclasses import dataclass
from decimal import Decimal

from ..errors import InputError
from ..values import decimal_text, identifier, number, require_object
from .gum import parse_matrix, validate_psd, validate_symmetric
from .models import unit_vector


@dataclass(frozen=True)
class SystematicEffect:
    id: str
    uncertainty: Decimal
    unit: str
    certificate: str
    label: str = ""

    @classmethod
    def parse(cls, data, registry):
        fields = {"id", "uncertainty", "unit", "certificate", "label"}
        require_object(data, fields, {"id", "uncertainty", "unit", "certificate"})
        registry.get(data["unit"])
        return cls(
            identifier(data["id"]),
            number(data["uncertainty"], "uncertainty", nonnegative=True),
            identifier(data["unit"], "unit"),
            identifier(data["certificate"], "certificate"),
            str(data.get("label", ""))[:512],
        )

    def as_dict(self):
        return {
            "id": self.id,
            "uncertainty": decimal_text(self.uncertainty),
            "unit": self.unit,
            "certificate": self.certificate,
            "label": self.label,
        }


@dataclass(frozen=True)
class EffectLink:
    effect: str
    coefficient: Decimal = Decimal(1)

    @classmethod
    def parse(cls, data):
        require_object(
            data, {"effect", "coefficient"}, {"effect"}
        )
        return cls(
            identifier(data["effect"], "effect"),
            number(data.get("coefficient", 1), "coefficient"),
        )

    def as_dict(self):
        return {
            "effect": self.effect,
            "coefficient": decimal_text(self.coefficient),
        }


@dataclass(frozen=True)
class InputBudget:
    name: str
    value: Decimal
    random_uncertainty: Decimal
    effects: tuple

    @classmethod
    def parse(cls, data):
        require_object(
            data,
            {"name", "value", "random_uncertainty", "effects"},
            {"name"},
        )
        effects = tuple(
            EffectLink.parse(item) for item in data.get("effects", [])
        )
        if len({link.effect for link in effects}) != len(effects):
            raise InputError(
                "an effect can be linked to an input only once",
                field="effects",
            )
        return cls(
            identifier(data["name"], "name"),
            number(data["value"], "value") if data.get("value") is not None else None,
            number(
                data.get("random_uncertainty", 0),
                "random_uncertainty",
                nonnegative=True,
            ),
            effects,
        )

    def as_dict(self):
        return {
            "name": self.name,
            "value": decimal_text(self.value),
            "random_uncertainty": decimal_text(self.random_uncertainty),
            "effects": [link.as_dict() for link in self.effects],
        }


class UncertaintyBudget:
    """Assembles the full covariance matrix for one model evaluation."""

    def __init__(self, model, budgets, effects, explicit_matrix, registry):
        self.model = model
        self.registry = registry
        self.inputs = model.inputs
        names = [item.name for item in self.inputs]
        if len(budgets) != len(names) or {item.name for item in budgets} != set(names):
            raise InputError(
                "an uncertainty budget entry is required for every model input"
            )
        self.budgets = {item.name: item for item in budgets}
        self.effects = {item.id: item for item in effects}
        if len(self.effects) != len(effects):
            raise InputError("systematic effect ids must be unique")
        for budget in budgets:
            for link in budget.effects:
                if link.effect not in self.effects:
                    raise InputError(
                        "unknown systematic effect",
                        field="effects",
                        details={"effect": link.effect},
                    )
        self.working_units = {
            item.name: self._working_unit(item.unit) for item in self.inputs
        }
        self.dimension_vectors = {
            item.name: unit_vector(registry, item.unit)
            for item in self.inputs
        }
        size = len(names)
        self.random = [[Decimal(0) for _ in names] for _ in names]
        self.systematic = [[Decimal(0) for _ in names] for _ in names]
        self.systematic_detail = {}
        self._build_random(names)
        self._build_systematic(names)
        self.explicit = (
            parse_matrix(
                explicit_matrix,
                names,
                [self.dimension_vectors[name] for name in names],
                field="covariance",
            )
            if explicit_matrix is not None
            else [[Decimal(0) for _ in names] for _ in names]
        )
        self.matrix = [
            [
                self.random[i][j]
                + self.systematic[i][j]
                + self.explicit[i][j]
                for j in range(size)
            ]
            for i in range(size)
        ]
        validate_symmetric(self.matrix, size, field="covariance")
        validate_psd(self.matrix, names, field="covariance")
        self.names = names

    def _working_unit(self, unit):
        candidates = [
            definition
            for definition in self.registry.definitions()
            if definition["dimension"] == self.registry.dimension(unit)
            and definition["scale"] == "1"
            and definition["offset"] == "0"
        ]
        if len(candidates) != 1:
            raise InputError(
                "dimension has no unique base unit for uncertainty assembly",
                details={"unit": unit},
            )
        return candidates[0]["symbol"]

    def _to_working(self, value, unit):
        return self.registry.uncertainty(value, unit, self._working_unit(unit))

    def _build_random(self, names):
        for index, name in enumerate(names):
            budget = self.budgets[name]
            converted = self._to_working(
                budget.random_uncertainty, self.model_input(name).unit
            )
            self.random[index][index] = converted * converted

    def _build_systematic(self, names):
        positions = {name: index for index, name in enumerate(names)}
        for effect_id, effect in sorted(self.effects.items()):
            affected = []
            for name in names:
                for link in self.budgets[name].effects:
                    if link.effect == effect_id:
                        affected.append((name, link))
            contributions = {}
            for name, link in affected:
                input_unit = self.model_input(name).unit
                if not self.registry.compatible(effect.unit, input_unit):
                    raise InputError(
                        "systematic effect unit does not match its input dimension",
                        field="effects",
                        details={
                            "effect": effect_id,
                            "effect_unit": effect.unit,
                            "input": name,
                            "input_unit": input_unit,
                        },
                    )
                native = self.registry.uncertainty(
                    effect.uncertainty, effect.unit, input_unit
                )
                contributions[name] = self._to_working(native, input_unit) * link.coefficient
            detail = [[Decimal(0) for _ in names] for _ in names]
            for left_name, left_value in contributions.items():
                for right_name, right_value in contributions.items():
                    entry = left_value * right_value
                    i, j = positions[left_name], positions[right_name]
                    self.systematic[i][j] += entry
                    detail[i][j] += entry
            self.systematic_detail[effect_id] = detail

    def model_input(self, name):
        return next(item for item in self.inputs if item.name == name)

    def working_values(self):
        result = {}
        for item in self.inputs:
            value = self.budgets[item.name].value
            result[item.name] = self.registry.convert(
                value, item.unit, self.working_units[item.name]
            )
        return result

    def output_scale_factor(self, name, output_unit):
        """Factor converting a base-unit input into the output unit's scale.

        Inputs are expressed in the unitary unit of their own dimension.  If
        the declared output unit has scale ``s`` relative to its dimension's
        base unit, an input whose dimension vector contains the output
        dimension with exponent ``e`` contributes as if expressed in a unit
        whose scale is ``s**e``; its numerical value must be divided by that
        scale.  Dimensions absent from the output vector are unaffected.
        """

        output = self.registry.get(output_unit)
        vector = self.dimension_vectors[name]
        exponent = vector.get(output.dimension, 0)
        if exponent == 0:
            return Decimal(1)
        return Decimal(1) / (output.scale ** exponent)

    def native_sensitivity(self, name, base_coefficient):
        """Convert a coefficient against the base working unit back to the
        coefficient against the input's native declared unit.

        If x_native is expressed in a unit with scale ``s`` and x_base in the
        unitary unit (scale 1), then x_base = s * x_native, so
        dy/dx_native = (dy/dx_base) * s.
        """

        unit = self.model_input(name).unit
        scale = self.registry.get(unit).scale
        return base_coefficient * scale

    def provenance(self):
        names = [item.name for item in self.inputs]
        return {
            "inputs": [
                {
                    **self.budgets[item.name].as_dict(),
                    "unit": item.unit,
                    "working_unit": self.working_units[item.name],
                    "observation": item.observation,
                    "certificate": item.certificate,
                    "source_model": item.source_model,
                }
                for item in self.inputs
            ],
            "systematic_effects": [
                effect.as_dict() for effect in self.effects.values()
            ],
            "covariance_matrix": [
                [decimal_text(cell) for cell in row] for row in self.explicit
            ],
            "covariance_matrix_note": (
                "entries are expressed in the product of the working units "
                "of the corresponding inputs"
            ),
            "covariance_entry_units": [
                [
                    self.working_units[names[i]]
                    + "*"
                    + self.working_units[names[j]]
                    for j in range(len(names))
                ]
                for i in range(len(names))
            ],
            "working_covariance": [
                [decimal_text(cell) for cell in row] for row in self.matrix
            ],
            "working_units": dict(self.working_units),
        }
