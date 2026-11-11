import unittest
from decimal import Decimal

from metrology.errors import Conflict, InputError, Missing
from metrology.ingress.readers import InputRow
from metrology.observations import Observation
from metrology.console import dispatch
from support import RECEIVED, application, certificate, observation


def second_channel(app, instrument="i2", channel="q", value="20", obs_id="r1"):
    app.catalog.add_instrument(
        {"id": instrument, "label": "Second instrument", "channels": {channel: "kPa"}}
    )
    app.catalog.add_certificate(
        certificate(id="c2", instrument=instrument, channel=channel)
    )
    app.catalog.add_policy(
        {"id": "q2", "unit": "kPa", "lower": "0", "upper": "40"}
    )
    app.catalog.bind(instrument, channel, "q2")
    item = Observation.parse(
        {
            "id": obs_id,
            "source": "s1",
            "sequence": 1,
            "instrument": instrument,
            "channel": channel,
            "event_time": "2026-11-11T00:00:01Z",
            "value": value,
            "unit": "kPa",
            "tags": {"lot": "A"},
        }
    )
    app.pipeline.ingest(
        "b2", [InputRow(1, observation=item)], received_at=RECEIVED
    )
    return item


def sum_model(model_id="m1", expression="p + q"):
    return {
        "id": model_id,
        "expression": expression,
        "output_unit": "kPa",
        "inputs": [
            {"name": "p", "unit": "kPa", "observation": "r0", "certificate": "c1"},
            {"name": "q", "unit": "kPa", "observation": "r1", "certificate": "c2"},
        ],
    }


class ModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.app = application()
        self.addCleanup(self.app.close)
        self.app.pipeline.ingest(
            "b", [InputRow(1, observation=observation(0))], received_at=RECEIVED
        )
        second_channel(self.app)

    def _register(self, **changes):
        data = sum_model()
        data.update(changes)
        return self.app.models.register_model(
            data, created_at="2026-11-11T01:00:00Z"
        )

    def test_register_and_evaluate_linear_sum(self):
        self._register()
        report = self.app.models.evaluate(
            "m1",
            {"inputs": {"p": {}, "q": {}}},
            report_id="rep1",
            created_at="2026-11-11T02:00:00Z",
        )
        self.assertEqual(report["combined_value"], "30")
        self.assertEqual(report["output_unit"], "kPa")
        self.assertEqual(len(report["sensitivities"]), 2)
        self.assertEqual(report["sensitivities"][0]["coefficient"], "1")
        self.assertEqual(report["coverage_factor"], "2")
        self.assertIn("not a confidence level", report["coverage_note"])

    def test_versioning_reuses_identical_content(self):
        first = self._register()
        again = self._register()
        self.assertEqual(first["version"], 1)
        self.assertFalse(again["created"])
        changed = self._register(expression="p + 2*q")
        self.assertEqual(changed["version"], 2)
        self.assertTrue(changed["created"])
        self.assertEqual(
            self.app.models.model("m1", 1)["expression"], "p + q"
        )
        self.assertEqual(self.app.models.model("m1")["expression"], "p + 2*q")

    def test_dimension_mismatch_rejected_at_registration(self):
        with self.assertRaises(InputError):
            self._register(expression="p * q", output_unit="kPa")

    def test_dependency_cycle_rejected(self):
        self.app.models.register_model(
            {
                "id": "mB",
                "expression": "p",
                "output_unit": "kPa",
                "inputs": [
                    {
                        "name": "p",
                        "unit": "kPa",
                        "observation": "r0",
                        "certificate": "c1",
                    }
                ],
            },
            created_at="2026-11-11T01:00:00Z",
        )
        self.app.models.register_model(
            {
                "id": "mA",
                "expression": "x",
                "output_unit": "kPa",
                "inputs": [
                    {
                        "name": "x",
                        "unit": "kPa",
                        "observation": "r0",
                        "certificate": "c1",
                        "source_model": "mB",
                    }
                ],
            },
            created_at="2026-11-11T01:00:00Z",
        )
        with self.assertRaises(InputError):
            self.app.models.register_model(
                {
                    "id": "mB",
                    "expression": "y",
                    "output_unit": "kPa",
                    "inputs": [
                        {
                            "name": "y",
                            "unit": "kPa",
                            "observation": "r0",
                            "certificate": "c1",
                            "source_model": "mA",
                        }
                    ],
                },
                created_at="2026-11-11T02:00:00Z",
            )

    def test_certificate_must_match_measurement(self):
        self._register()
        # A new model version may bind an input to a different registered
        # certificate, but evaluation must reject it when the stored
        # measurement was actually produced by another certificate.
        self.app.catalog.add_certificate(
            certificate(
                id="c3",
                instrument="i2",
                channel="q",
                valid_from="2030-01-01T00:00:00Z",
                valid_to="2031-01-01T00:00:00Z",
            )
        )
        data = sum_model("m1b")
        data["inputs"][1]["certificate"] = "c3"
        self.app.models.register_model(
            data, created_at="2026-11-11T01:00:00Z"
        )
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "m1b",
                {"inputs": {}},
                report_id="rep-bad",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_revoked_certificate_blocks_evaluation(self):
        self._register()
        self.app.catalog.revoke_certificate("c1", "correction issued")
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "m1",
                {"inputs": {}},
                report_id="rep-revoked",
                created_at="2026-11-11T02:00:00Z",
            )


class UncertaintyBudgetTests(unittest.TestCase):
    def setUp(self):
        self.app = application()
        self.addCleanup(self.app.close)
        self.app.pipeline.ingest(
            "b", [InputRow(1, observation=observation(0))], received_at=RECEIVED
        )
        second_channel(self.app)
        self.app.models.register_model(
            sum_model(), created_at="2026-11-11T01:00:00Z"
        )

    def test_shared_systematic_effect_creates_covariance(self):
        report = self.app.models.evaluate(
            "m1",
            {
                "effects": [
                    {
                        "id": "ref",
                        "uncertainty": "0.5",
                        "unit": "kPa",
                        "certificate": "c1",
                    }
                ],
                "inputs": {
                    "p": {
                        "random_uncertainty": "0.1",
                        "effects": [{"effect": "ref"}],
                    },
                    "q": {
                        "random_uncertainty": "0.2",
                        "effects": [{"effect": "ref"}],
                    },
                },
            },
            report_id="rep1",
            created_at="2026-11-11T02:00:00Z",
        )
        # u^2 = .1^2 + .2^2 + 2*.5^2 = 1.05
        self.assertAlmostEqual(
            float(report["standard_uncertainty"]),
            float(Decimal("1.05").sqrt()),
            places=9,
        )
        systematic = report["contributions"]["systematic"]
        self.assertEqual(systematic["variance"], "1")
        self.assertEqual(
            systematic["effects"][0]["certificate"], "c1"
        )
        fractions = (
            Decimal(report["contributions"]["random"]["fraction"])
            + Decimal(report["contributions"]["systematic"]["fraction"])
            + Decimal(report["contributions"]["covariance"]["fraction"])
        )
        self.assertEqual(fractions, Decimal(1))

    def test_asymmetric_matrix_rejected(self):
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "m1",
                {
                    "covariance": [["1", "0.5"], ["0.2", "1"]],
                    "inputs": {"p": {}, "q": {}},
                },
                report_id="bad1",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_non_psd_matrix_rejected(self):
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "m1",
                {
                    "covariance": [["1", "2"], ["2", "1"]],
                    "inputs": {"p": {}, "q": {}},
                },
                report_id="bad2",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_covariance_dimension_mismatch_rejected(self):
        # Register a third, voltage-dimension input and confirm a covariance
        # entry across different dimensions is refused.
        self.app.catalog.add_instrument(
            {"id": "iv", "label": "Volt", "channels": {"v": "V"}}
        )
        self.app.catalog.add_certificate(
            certificate(id="cv", instrument="iv", channel="v",
                        input_unit="V", output_unit="V",
                        minimum="0", maximum="10")
        )
        self.app.catalog.add_policy(
            {"id": "pv", "unit": "V", "lower": "0", "upper": "10"}
        )
        self.app.catalog.bind("iv", "v", "pv")
        item = Observation.parse(
            {
                "id": "rv",
                "source": "s1",
                "sequence": 2,
                "instrument": "iv",
                "channel": "v",
                "event_time": "2026-11-11T00:00:02Z",
                "value": "5",
                "unit": "V",
                "tags": {"lot": "A"},
            }
        )
        self.app.pipeline.ingest(
            "bv", [InputRow(1, observation=item)], received_at=RECEIVED
        )
        self.app.models.register_model(
            {
                "id": "mmix",
                "expression": "(p + q) * v / v",
                "output_unit": "kPa",
                "inputs": [
                    {
                        "name": "p",
                        "unit": "kPa",
                        "observation": "r0",
                        "certificate": "c1",
                    },
                    {
                        "name": "q",
                        "unit": "kPa",
                        "observation": "r1",
                        "certificate": "c2",
                    },
                    {
                        "name": "v",
                        "unit": "V",
                        "observation": "rv",
                        "certificate": "cv",
                    },
                ],
            },
            created_at="2026-11-11T01:00:00Z",
        )
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "mmix",
                {
                    "covariance": [
                        ["0", "0", "0.1"],
                        ["0", "0", "0"],
                        ["0.1", "0", "0"],
                    ],
                    "inputs": {"p": {}, "q": {}, "v": {}},
                },
                report_id="bad-dim",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_unit_mismatch_effect_rejected(self):
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "m1",
                {
                    "effects": [
                        {
                            "id": "ref",
                            "uncertainty": "1",
                            "unit": "V",
                            "certificate": "c1",
                        }
                    ],
                    "inputs": {
                        "p": {"effects": [{"effect": "ref"}]},
                        "q": {},
                    },
                },
                report_id="bad3",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_unknown_effect_rejected(self):
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "m1",
                {"inputs": {"p": {"effects": [{"effect": "ghost"}]}, "q": {}}},
                report_id="bad4",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_unit_scaling_in_budget(self):
        # Express p in Pa while the model unit is kPa; the numeric value and
        # uncertainty must be converted before propagation so the combined
        # value stays identical to the all-kPa model.
        data = sum_model("m-pa")
        data["inputs"][0]["unit"] = "Pa"
        self.app.models.register_model(
            data, created_at="2026-11-11T01:00:00Z"
        )
        report = self.app.models.evaluate(
            "m-pa",
            {"inputs": {"p": {}, "q": {}}},
            report_id="rep-pa",
            created_at="2026-11-11T02:00:00Z",
        )
        self.assertEqual(report["combined_value"], "30")
        self.assertEqual(
            report["inputs"][0]["unit"], "Pa"
        )


class ReportAndSamplingTests(unittest.TestCase):
    def setUp(self):
        self.app = application()
        self.addCleanup(self.app.close)
        self.app.catalog.add_unit(
            {
                "symbol": "one",
                "dimension": "dimensionless",
                "scale": "1",
                "offset": "0",
            }
        )
        self.app.pipeline.ingest(
            "b", [InputRow(1, observation=observation(0))], received_at=RECEIVED
        )
        second_channel(self.app)
        self.app.models.register_model(
            {
                "id": "mr",
                "expression": "p / q",
                "output_unit": "one",
                "inputs": [
                    {
                        "name": "p",
                        "unit": "kPa",
                        "observation": "r0",
                        "certificate": "c1",
                    },
                    {
                        "name": "q",
                        "unit": "kPa",
                        "observation": "r1",
                        "certificate": "c2",
                    },
                ],
            },
            created_at="2026-11-11T01:00:00Z",
        )

    def _request(self, seed=42):
        return {
            "inputs": {
                "p": {"random_uncertainty": "0.1"},
                "q": {"random_uncertainty": "0.2"},
            },
            "monte_carlo": {"seed": seed, "samples": 5000},
        }

    def test_nonlinear_flag_and_monte_carlo(self):
        report = self.app.models.evaluate(
            "mr",
            self._request(),
            report_id="rep1",
            created_at="2026-11-11T02:00:00Z",
        )
        self.assertTrue(report["first_order_nonlinear"])
        mc = report["monte_carlo"]
        self.assertEqual(mc["seed"], 42)
        self.assertEqual(mc["valid_samples"], 5000)
        self.assertEqual(mc["invalid_samples"], 0)
        self.assertIn("p02_5", mc["distribution"])
        self.assertIn("comparison", mc)
        self.assertNotIn("confidence", mc["comparison"])
        # Analytic first-order result: p/q = 0.5
        self.assertEqual(report["combined_value"], "0.5")

    def test_fixed_seed_is_deterministic(self):
        first = self.app.models.evaluate(
            "mr",
            self._request(),
            report_id="rep-a",
            created_at="2026-11-11T02:00:00Z",
        )
        second = self.app.models.evaluate(
            "mr",
            self._request(),
            report_id="rep-b",
            created_at="2026-11-11T02:01:00Z",
        )
        self.assertEqual(
            first["monte_carlo"]["distribution"],
            second["monte_carlo"]["distribution"],
        )

    def test_invalid_samples_are_counted_not_dropped(self):
        # Force the denominator's documented value to zero with zero variance
        # so every Monte Carlo draw divides by zero; the run must fail with a
        # domain error rather than returning silently truncated statistics.
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "mr",
                {
                    "inputs": {
                        "p": {"random_uncertainty": "0.01"},
                        "q": {"value": "0", "random_uncertainty": "0"},
                    },
                    "monte_carlo": {"seed": 3, "samples": 200},
                },
                report_id="rep-invalid",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_division_by_zero_at_nominal_point_rejected(self):
        report_model = {
            "id": "mzero",
            "expression": "p / q",
            "output_unit": "one",
            "inputs": [
                {
                    "name": "p",
                    "unit": "kPa",
                    "observation": "r0",
                    "certificate": "c1",
                },
                {
                    "name": "q",
                    "unit": "kPa",
                    "observation": "r1",
                    "certificate": "c2",
                },
            ],
        }
        self.app.models.register_model(
            report_model, created_at="2026-11-11T01:00:00Z"
        )
        with self.assertRaises(InputError):
            self.app.models.evaluate(
                "mzero",
                {"inputs": {"p": {}, "q": {"value": "0"}}},
                report_id="rep-zero",
                created_at="2026-11-11T02:00:00Z",
            )

    def test_report_immutability_and_correction(self):
        self.app.models.evaluate(
            "mr",
            self._request(),
            report_id="rep1",
            created_at="2026-11-11T02:00:00Z",
        )
        with self.assertRaises(Conflict):
            self.app.models.evaluate(
                "mr",
                self._request(),
                report_id="rep1",
                created_at="2026-11-11T02:05:00Z",
            )
        corrected = self.app.models.corrected_report(
            "mr",
            {
                "inputs": {
                    "p": {"random_uncertainty": "0.12"},
                    "q": {"random_uncertainty": "0.2"},
                }
            },
            report_id="rep2",
            supersedes="rep1",
            created_at="2026-11-11T03:00:00Z",
            reason="certificate repeatability corrected",
        )
        original = self.app.models.report("rep1")
        self.assertEqual(corrected["supersedes"], "rep1")
        self.assertEqual(corrected["correction_reason"],
                         "certificate repeatability corrected")
        self.assertNotIn("correction_reason", original)
        self.assertEqual(
            self.app.models.certificate_reports("c1")[0]["report"], "rep1"
        )

    def test_supersede_unknown_report_rejected(self):
        with self.assertRaises(Missing):
            self.app.models.evaluate(
                "mr",
                self._request(),
                report_id="rep-x",
                supersedes="missing",
                created_at="2026-11-11T03:00:00Z",
            )

    def test_console_dispatch(self):
        result = dispatch(
            self.app,
            {
                "action": "model_evaluate",
                "model_id": "mr",
                "report_id": "rep-console",
                "created_at": "2026-11-11T02:00:00Z",
                "request": {
                    "inputs": {
                        "p": {"random_uncertainty": "0.1"},
                        "q": {"random_uncertainty": "0.2"},
                    }
                },
            },
        )
        self.assertEqual(result["id"], "rep-console")
        listing = dispatch(self.app, {"action": "models"})
        self.assertTrue(any(m["id"] == "mr" for m in listing["models"]))


if __name__ == "__main__":
    unittest.main()
