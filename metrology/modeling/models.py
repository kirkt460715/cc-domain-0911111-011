"""Versionable measurement models.

A model binds named inputs (each carrying a unit, an observation and a
calibration certificate) to one arithmetic expression.  Parsing resolves the
output dimension, rejects illegal expression constructs and verifies that
inputs referenced by earlier model versions cannot form dependency cycles.
Models are content-addressed: identical content reuses an existing version,
changed content opens a new version and old versions stay readable.
"""

from dataclasses import dataclass

from ..errors import InputError, Missing
from ..values import (
    digest,
    identifier,
    label,
    require_object,
    strict_json,
)
from . import dimensions as dims
from .expressions import ModelExpression

DIMENSIONLESS_LABELS = frozenset({"dimensionless", "1", "one", "unitless"})


def unit_vector(registry, unit):
    label_name = registry.dimension(unit)
    if label_name in DIMENSIONLESS_LABELS:
        return {}
    return dims.single(label_name)


@dataclass(frozen=True)
class ModelInput:
    name: str
    unit: str
    observation: str
    certificate: str
    source_model: str = None

    @classmethod
    def parse(cls, data):
        fields = {"name", "unit", "observation", "certificate", "source_model"}
        require_object(data, fields, {"name", "unit", "observation", "certificate"})
        source_model = data.get("source_model")
        return cls(
            identifier(data["name"], "name"),
            identifier(data["unit"], "unit"),
            identifier(data["observation"], "observation"),
            identifier(data["certificate"], "certificate"),
            identifier(source_model, "source_model") if source_model else None,
        )

    def as_dict(self):
        return {
            "name": self.name,
            "unit": self.unit,
            "observation": self.observation,
            "certificate": self.certificate,
            "source_model": self.source_model,
        }


@dataclass(frozen=True)
class MeasurementModel:
    id: str
    version: int
    description: str
    expression: ModelExpression
    inputs: tuple
    output_unit: str
    output_dimension: dict

    @classmethod
    def parse(cls, data, registry, *, version, existing=None):
        fields = {
            "id",
            "description",
            "expression",
            "inputs",
            "output_unit",
        }
        require_object(data, fields, {"id", "expression", "inputs"})
        model_id = identifier(data["id"])
        if not isinstance(data["inputs"], list) or not data["inputs"]:
            raise InputError("model must declare one or more inputs")
        if len(data["inputs"]) > 100:
            raise InputError("model supports at most one hundred inputs")
        inputs = tuple(ModelInput.parse(item) for item in data["inputs"])
        names = [item.name for item in inputs]
        if len(set(names)) != len(names):
            raise InputError("model input names must be unique")
        for item in inputs:
            registry.get(item.unit)
        expression = ModelExpression.parse(data["expression"])
        unknown = set(expression.variables) - set(names)
        if unknown:
            raise InputError(
                "expression references undeclared inputs",
                details={"inputs": sorted(unknown)},
            )
        unused = set(names) - set(expression.variables)
        if unused:
            raise InputError(
                "declared inputs are not used by the expression",
                details={"inputs": sorted(unused)},
            )
        input_dimensions = {
            item.name: unit_vector(registry, item.unit) for item in inputs
        }
        output_dimension = expression.check_dimensions(input_dimensions)
        output_unit = data.get("output_unit")
        if output_unit is None:
            output_unit = _default_unit(output_dimension, registry)
        else:
            output_unit = identifier(output_unit, "output_unit")
            registry.get(output_unit)
            target = unit_vector(registry, output_unit)
            if not dims.equal(target, output_dimension):
                raise InputError(
                    "declared output unit does not match expression dimension",
                    details={
                        "unit_dimension": dims.describe(target),
                        "expression_dimension": dims.describe(output_dimension),
                    },
                )
        if existing is not None:
            _check_cycles(model_id, inputs, existing)
        description = data.get("description", "")
        if description:
            description = label(description, "description", 2000)
        return cls(
            model_id,
            version,
            description,
            expression,
            inputs,
            output_unit,
            output_dimension,
        )

    def as_dict(self):
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "expression": self.expression.text,
            "inputs": [item.as_dict() for item in self.inputs],
            "output_unit": self.output_unit,
            "output_dimension": dims.describe(self.output_dimension),
        }

    @property
    def fingerprint(self):
        body = {
            "id": self.id,
            "description": self.description,
            "expression": self.expression.text,
            "inputs": [item.as_dict() for item in self.inputs],
            "output_unit": self.output_unit,
        }
        return digest(body)

    @classmethod
    def restore(cls, payload, registry):
        data = strict_json(payload)
        expression = ModelExpression.parse(data["expression"])
        inputs = tuple(ModelInput.parse(item) for item in data["inputs"])
        output_dimension = expression.check_dimensions(
            {
                item.name: unit_vector(registry, item.unit)
                for item in inputs
            }
        )
        return cls(
            data["id"],
            data["version"],
            data["description"],
            expression,
            inputs,
            data["output_unit"],
            output_dimension,
        )


def _default_unit(output_dimension, registry):
    entries = [(name, exponent) for name, exponent in output_dimension.items()]
    if not entries:
        raise InputError(
            "dimensionless model output requires an explicit output_unit",
            field="output_unit",
        )
    if len(entries) != 1 or entries[0][1] != 1:
        raise InputError(
            "derived output dimension requires an explicit output_unit",
            field="output_unit",
        )
    dimension = entries[0][0]
    matches = [
        unit
        for unit in registry.definitions()
        if unit["dimension"] == dimension
        and unit["scale"] == "1"
        and unit["offset"] == "0"
    ]
    if len(matches) != 1:
        raise InputError(
            "output unit is ambiguous for expression dimension",
            field="output_unit",
            details={"dimension": dimension},
        )
    return matches[0]["symbol"]


def _check_cycles(model_id, inputs, existing):
    """Reject a new version whose inputs would make models depend on themselves.

    The candidate model's own outgoing references come from ``inputs``; every
    already registered model's references come from ``existing``.
    """

    def references(current):
        if current == model_id:
            return [item.source_model for item in inputs if item.source_model]
        model = existing.get(current)
        if model is None:
            raise Missing("source model not registered", field=current)
        return [item.source_model for item in model.inputs if item.source_model]

    stack = list(references(model_id))
    seen = set()
    while stack:
        current = stack.pop()
        if current == model_id:
            raise InputError(
                "model inputs create a dependency cycle",
                details={"model": model_id},
            )
        if current in seen:
            continue
        seen.add(current)
        stack.extend(references(current))
