"""Measurement-model evaluation and immutable uncertainty reports.

An evaluation resolves each input's documented value and uncertainty budget,
propagates them with the first-order GUM method and stores an immutable
report.  Coverage factors are reported as coverage factors, never described
as confidence levels.  A certificate correction does not modify an existing
report: the corrected evidence produces a new report that references the one
it supersedes.
"""

import datetime as dt
from decimal import Decimal, localcontext

from ..errors import InputError, Missing
from ..values import (
    decimal_text,
    digest,
    identifier,
    number,
    require_object,
    strict_json,
    time_text,
)
from .gum import combined_variance, sensitivity
from .sampling import SamplingCheck, compare
from .uncertainty import (
    InputBudget,
    SystematicEffect,
    UncertaintyBudget,
)


class ModelEvaluation:
    def __init__(self, database, catalog):
        self.db = database
        self.catalog = catalog

    def _measurement(self, observation_id):
        row = self.db.one(
            """SELECT r.payload FROM observations o JOIN results r
            ON r.observation=o.id WHERE o.id=? AND r.active=1 AND r.payload IS NOT NULL""",
            (observation_id,),
        )
        if row is None:
            # Distinguish a missing/rejected measurement from a plain miss.
            state = self.db.one(
                "SELECT payload FROM results WHERE observation=? AND active=1",
                (observation_id,),
            )
            if state is None:
                raise Missing(
                    "observation has no active measurement", field=observation_id
                )
            raise Missing(
                "observation measurement failed processing", field=observation_id
            )
        payload = strict_json(row["payload"])
        if not payload["accepted"]:
            raise InputError(
                "observation measurement was rejected",
                field=observation_id,
            )
        return payload

    def _resolve_inputs(self, model, request_inputs):
        request_inputs = request_inputs or {}
        if not isinstance(request_inputs, dict):
            raise InputError("inputs must be an object keyed by input name")
        unknown = set(request_inputs) - {item.name for item in model.inputs}
        if unknown:
            raise InputError(
                "unknown model inputs", details={"inputs": sorted(unknown)}
            )
        budgets = []
        resolved = []
        registry = self.catalog.registry()
        for item in model.inputs:
            override = request_inputs.get(item.name, {})
            require_object(override)
            measurement = self._measurement(item.observation)
            certificate_row = self.db.one(
                "SELECT revoked FROM certificates WHERE id=?",
                (item.certificate,),
            )
            if certificate_row is None:
                raise Missing(
                    "calibration certificate not registered",
                    field=item.certificate,
                )
            if certificate_row["revoked"]:
                raise InputError(
                    "calibration certificate has been revoked",
                    field=item.certificate,
                    details={"certificate": item.certificate},
                )
            if measurement["certificate"] != item.certificate:
                raise InputError(
                    "observation certificate does not match model input",
                    field=item.certificate,
                    details={
                        "model_certificate": item.certificate,
                        "measurement_certificate": measurement["certificate"],
                    },
                )
            if not registry.compatible(measurement["unit"], item.unit):
                raise InputError(
                    "measurement unit does not match model input dimension",
                    field=item.name,
                )
            native_value = registry.convert(
                number(measurement["corrected_value"]),
                measurement["unit"],
                item.unit,
            )
            if "value" in override:
                native_value = number(override["value"], "value")
            native_uncertainty = registry.uncertainty(
                number(measurement["corrected_uncertainty"]),
                measurement["unit"],
                item.unit,
            )
            data = {
                "name": item.name,
                "value": native_value,
                "random_uncertainty": override.get(
                    "random_uncertainty", native_uncertainty
                ),
                "effects": override.get("effects", []),
            }
            budgets.append(InputBudget.parse(data))
            resolved.append(
                {
                    "name": item.name,
                    "value": decimal_text(native_value),
                    "unit": item.unit,
                    "observation": item.observation,
                    "certificate": item.certificate,
                    "measurement_revision": self._revision(item.observation),
                }
            )
        return budgets, resolved

    def _revision(self, observation_id):
        row = self.db.one(
            "SELECT revision FROM results WHERE observation=? AND active=1",
            (observation_id,),
        )
        return row["revision"] if row else None

    def evaluate(self, model, request, *, report_id, created_at, supersedes=None):
        require_object(
            request,
            {
                "effects",
                "covariance",
                "coverage_factor",
                "inputs",
                "monte_carlo",
            },
        )
        if not isinstance(created_at, dt.datetime):
            raise InputError("created_at must be a timezone-qualified timestamp")
        registry = self.catalog.registry()
        coverage = number(
            request.get("coverage_factor", 2), "coverage_factor", positive=True
        )
        effects = [
            SystematicEffect.parse(item, registry)
            for item in request.get("effects", [])
        ]
        budgets, resolved_inputs = self._resolve_inputs(
            model, request.get("inputs")
        )
        budget = UncertaintyBudget(
            model, budgets, effects, request.get("covariance"), registry
        )
        variables = list(model.expression.variables)
        base_values = budget.working_values()
        # Rescale every input so that evaluating the expression directly
        # yields a result expressed in the declared output unit (offsets do
        # not apply to delta/derived arithmetic, only scales do).  The same
        # per-input scale must be applied to the covariance matrix.
        scales = {
            name: budget.output_scale_factor(name, model.output_unit)
            for name in variables
        }
        working_values = {
            name: base_values[name] * scales[name] for name in variables
        }
        scaled_covariance = [
            [
                budget.matrix[i][j]
                * scales[variables[i]]
                * scales[variables[j]]
                for j in range(len(variables))
            ]
            for i in range(len(variables))
        ]
        value_working, working_coefficients = sensitivity(
            model.expression.tree, variables, working_values
        )
        variance_working = combined_variance(
            [working_coefficients[name] for name in variables],
            scaled_covariance,
        )
        with localcontext() as ctx:
            ctx.prec = 40
            standard_working = variance_working.sqrt()
            output_value = value_working
            standard = standard_working
            expanded = standard * coverage
        contributions = self._contributions(
            model, budget, variables, working_coefficients, scales
        )
        sensitivities = [
            {
                "input": name,
                "coefficient": decimal_text(working_coefficients[name]),
                "coefficient_native_unit": decimal_text(
                    budget.native_sensitivity(name, working_coefficients[name])
                ),
                "unit": model.output_unit + "/" + budget.working_units[name],
            }
            for name in variables
        ]
        nonlinear = not model.expression.is_linear()
        sampling = None
        mc_request = request.get("monte_carlo")
        if mc_request is not None:
            require_object(
                mc_request, {"seed", "samples"}, {"seed", "samples"}
            )
            check = SamplingCheck(
                model,
                model.expression.tree,
                variables,
                working_values,
                scaled_covariance,
                mc_request["seed"],
            )
            sampling = check.run(int(mc_request.get("samples", 10000)))
            sampling["comparison"] = compare(
                value_working, standard_working, sampling
            )
        report = {
            "id": identifier(report_id),
            "model_id": model.id,
            "model_version": model.version,
            "model_fingerprint": model.fingerprint,
            "created_at": time_text(created_at),
            "combined_value": decimal_text(output_value),
            "output_unit": model.output_unit,
            "sensitivities": sensitivities,
            "contributions": contributions,
            "standard_uncertainty": decimal_text(standard),
            "coverage_factor": decimal_text(coverage),
            "expanded_uncertainty": decimal_text(expanded),
            "coverage_note": (
                "the coverage factor k is not a confidence level and no "
                "coverage probability is claimed"
            ),
            "first_order_nonlinear": nonlinear,
            "monte_carlo": sampling,
            "inputs": resolved_inputs,
            "budget": budget.provenance(),
            "supersedes": supersedes,
        }
        report["fingerprint"] = digest(
            {key: value for key, value in report.items() if key != "fingerprint"}
        )
        return report

    def _contributions(
        self, model, budget, variables, working_coefficients, scales
    ):
        def part(matrix):
            total = Decimal(0)
            entries = []
            size = len(variables)
            for i in range(size):
                for j in range(size):
                    term = (
                        working_coefficients[variables[i]]
                        * working_coefficients[variables[j]]
                        * matrix[i][j]
                        * scales[variables[i]]
                        * scales[variables[j]]
                    )
                    total += term
                    if term != 0:
                        entries.append(
                            {
                                "left": variables[i],
                                "right": variables[j],
                                "variance": decimal_text(term),
                            }
                        )
            return total, entries

        variance_unit = model.output_unit + "^2"
        result = {"random": [], "systematic": [], "covariance": []}
        random_total, random_entries = part(budget.random)
        result["random"] = {
            "variance": decimal_text(random_total),
            "variance_unit": variance_unit,
            "entries": random_entries,
        }
        systematic_entries = []
        systematic_total = Decimal(0)
        for effect_id, matrix in sorted(budget.systematic_detail.items()):
            total, entries = part(matrix)
            if total == 0 and not entries:
                continue
            systematic_total += total
            effect = budget.effects[effect_id]
            systematic_entries.append(
                {
                    "effect": effect_id,
                    "certificate": effect.certificate,
                    "variance": decimal_text(total),
                    "variance_unit": variance_unit,
                    "entries": entries,
                }
            )
        result["systematic"] = {
            "variance": decimal_text(systematic_total),
            "variance_unit": variance_unit,
            "effects": systematic_entries,
        }
        explicit_total, explicit_entries = part(budget.explicit)
        result["covariance"] = {
            "variance": decimal_text(explicit_total),
            "variance_unit": variance_unit,
            "entries": explicit_entries,
        }
        total = random_total + systematic_total + explicit_total
        if total > 0:
            result["random"]["fraction"] = decimal_text(
                result_random_fraction(random_total, total)
            )
            result["systematic"]["fraction"] = decimal_text(
                result_random_fraction(systematic_total, total)
            )
            result["covariance"]["fraction"] = decimal_text(
                result_random_fraction(explicit_total, total)
            )
        else:
            for key in result:
                result[key]["fraction"] = "0"
        return result


def result_random_fraction(part, total):
    if total == 0:
        return Decimal(0)
    return part / total
