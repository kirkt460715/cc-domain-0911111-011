import unittest
from decimal import Decimal

from metrology.errors import InputError
from metrology.modeling import dimensions as dims
from metrology.modeling.expressions import ModelExpression
from metrology.modeling.gum import (
    Dual,
    combined_variance,
    evaluate_dual,
    parse_matrix,
    validate_psd,
    validate_symmetric,
)
from metrology.modeling.sampling import SamplingCheck, percentile


class ExpressionTests(unittest.TestCase):
    def test_accepts_supported_grammar(self):
        for text in (
            "a + b",
            "2*a - 3*b",
            "a/b",
            "a^2",
            "a^-3",
            "(a + b)*c",
            "-a + +b",
            "3.14*a^2 + b/2",
        ):
            expression = ModelExpression.parse(text)
            self.assertTrue(expression.variables)

    def test_rejects_illegal_constructs(self):
        for text in (
            "__import__('os')",
            "a > b",
            "a ** 1.5",
            "lambda x: x",
            "a & b",
            "a | b",
            "~a",
            "a // b",
            "a % b",
            "f(a)",
            "a if b else c",
            "[a]",
            '{"k": a}',
        ):
            with self.assertRaises(InputError):
                ModelExpression.parse(text)

    def test_power_requires_integer_literal(self):
        with self.assertRaises(InputError):
            ModelExpression.parse("a^b")
        with self.assertRaises(InputError):
            ModelExpression.parse("a^(1+1)")

    def test_caret_maps_to_power(self):
        expression = ModelExpression.parse("a^2")
        self.assertEqual(expression.variables, ("a",))
        result = evaluate_dual(
            expression.tree,
            {"a": Dual(Decimal("5"), (Decimal("1"),))},
        )
        self.assertEqual(result.value, Decimal("25"))
        self.assertEqual(result.partials[0], Decimal("10"))

    def test_undeclared_and_unused_inputs(self):
        expression = ModelExpression.parse("a + b")
        with self.assertRaises(InputError):
            expression.check_dimensions({"a": dims.single("pressure")})

    def test_dimension_tracking(self):
        expression = ModelExpression.parse("a*b + c^2")
        pressure = dims.single("pressure")
        result = expression.check_dimensions(
            {"a": pressure, "b": pressure, "c": pressure}
        )
        self.assertEqual(result, {"pressure": 2})

    def test_additive_dimension_mismatch_rejected(self):
        expression = ModelExpression.parse("a + b")
        with self.assertRaises(InputError):
            expression.check_dimensions(
                {"a": dims.single("pressure"), "b": dims.single("length")}
            )

    def test_linearity_detection(self):
        self.assertTrue(ModelExpression.parse("2*a + b - 3").is_linear())
        self.assertFalse(ModelExpression.parse("a*b").is_linear())
        self.assertFalse(ModelExpression.parse("a/b").is_linear())
        self.assertFalse(ModelExpression.parse("a^2").is_linear())


class DimensionTests(unittest.TestCase):
    def test_product_quotient_and_power(self):
        pressure = dims.single("pressure")
        length = dims.single("length")
        self.assertEqual(
            dims.product(pressure, length), {"pressure": 1, "length": 1}
        )
        self.assertEqual(
            dims.quotient(pressure, length), {"pressure": 1, "length": -1}
        )
        self.assertEqual(dims.power(pressure, 3), {"pressure": 3})

    def test_zero_exponent_is_dimensionless(self):
        self.assertEqual(dims.power(dims.single("pressure"), 0), {})

    def test_exponent_range(self):
        with self.assertRaises(InputError):
            dims.power(dims.single("pressure"), 99)


class GumMathTests(unittest.TestCase):
    def test_dual_arithmetic_partials(self):
        expression = ModelExpression.parse("a*b + c^2 - d/2")
        names = {
            "a": Dual(Decimal("2"), (Decimal("1"), Decimal("0"), Decimal("0"), Decimal("0"))),
            "b": Dual(Decimal("3"), (Decimal("0"), Decimal("1"), Decimal("0"), Decimal("0"))),
            "c": Dual(Decimal("5"), (Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0"))),
            "d": Dual(Decimal("8"), (Decimal("0"), Decimal("0"), Decimal("0"), Decimal("1"))),
        }
        result = evaluate_dual(expression.tree, names)
        # 2*3 + 25 - 4 = 27
        self.assertEqual(result.value, Decimal("27"))
        # da = b = 3, db = a = 2, dc = 2c = 10, dd = -1/2
        self.assertEqual(
            result.partials,
            (Decimal("3"), Decimal("2"), Decimal("10"), Decimal("-0.5")),
        )

    def test_symmetric_validation(self):
        with self.assertRaises(InputError):
            validate_symmetric(
                [[Decimal(1), Decimal(2)], [Decimal(3), Decimal(1)]], 2
            )
        validate_symmetric([[Decimal(1), Decimal(2)], [Decimal(2), Decimal(1)]], 2)

    def test_psd_validation(self):
        validate_psd(
            [[Decimal(2), Decimal(1)], [Decimal(1), Decimal(2)]], ["a", "b"]
        )
        with self.assertRaises(InputError):
            validate_psd(
                [[Decimal(1), Decimal(2)], [Decimal(2), Decimal(1)]], ["a", "b"]
            )

    def test_parse_matrix_dimension_check(self):
        labels = ["a", "b"]
        vectors = [dims.single("pressure"), dims.single("length")]
        with self.assertRaises(InputError):
            parse_matrix(
                [["1", "0.5"], ["0.5", "1"]], labels, vectors
            )
        same = [dims.single("pressure"), dims.single("pressure")]
        matrix = parse_matrix([["1", "0"], ["0", "1"]], labels, same)
        self.assertEqual(matrix[0][0], Decimal(1))

    def test_combined_variance_with_correlation(self):
        coefficients = [Decimal(1), Decimal(1)]
        covariance = [
            [Decimal("0.01"), Decimal("0.005")],
            [Decimal("0.005"), Decimal("0.04")],
        ]
        variance = combined_variance(coefficients, covariance)
        # .01 + .04 + 2*.005 = .06
        self.assertEqual(variance, Decimal("0.06"))


class SamplingTests(unittest.TestCase):
    def test_zero_variance_zero_denominator_marks_invalid(self):
        # Deterministic input: denominator is fixed at zero, so every draw
        # divides by zero and must be recorded as an invalid sample.
        expression = ModelExpression.parse("a / b")
        check = SamplingCheck(
            model=None,
            tree=expression.tree,
            variables=["a", "b"],
            working_values={"a": Decimal("1"), "b": Decimal("0")},
            covariance=[[Decimal("0.01"), Decimal(0)], [Decimal(0), Decimal(0)]],
            seed=11,
        )
        with self.assertRaises(InputError):
            check.run(100)

    def test_seed_determines_distribution(self):
        expression = ModelExpression.parse("a + b")
        build = lambda seed: SamplingCheck(
            model=None,
            tree=expression.tree,
            variables=["a", "b"],
            working_values={"a": Decimal("10"), "b": Decimal("20")},
            covariance=[[Decimal("0.04"), Decimal(0)], [Decimal(0), Decimal("0.09")]],
            seed=seed,
        ).run(2000)
        self.assertEqual(build(5)["distribution"], build(5)["distribution"])
        self.assertNotEqual(
            build(5)["distribution"], build(6)["distribution"]
        )

    def test_percentile_edges(self):
        self.assertEqual(percentile([7.0], 2.5), 7.0)
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 50), 2.5)


if __name__ == "__main__":
    unittest.main()
