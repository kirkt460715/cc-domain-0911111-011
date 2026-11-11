"""Safe expression language for versioned measurement models.

The grammar deliberately covers only linear combinations, multiplication,
division and integer powers::

    expression := term (('+' | '-') term)*
    term       := factor (('*' | '/') factor)*
    factor     := ('+' | '-') factor | primary ('^' integer)?
    primary    := number | identifier | '(' expression ')'

Anything else (function calls, names of builtins, non-integer powers,
comparisons, booleans, ...) is rejected at parse time so stored models can
never execute arbitrary code.
"""

import ast
from decimal import Decimal

from ..errors import InputError
from ..values import IDENTIFIER
from . import dimensions as dims

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)
_MAX_POWER = 12


class ModelExpression:
    def __init__(self, text, tree, variables):
        self.text = text
        self.tree = tree
        self.variables = tuple(variables)

    @classmethod
    def parse(cls, text):
        if not isinstance(text, str) or not text.strip():
            raise InputError("expression must be nonempty text", field="expression")
        if len(text) > 1000:
            raise InputError("expression is too long", field="expression")
        # The model language uses the conventional metrology caret for
        # exponentiation; Python's grammar would read it as bitwise XOR.
        normalized = text.replace("^", "**")
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise InputError("malformed expression", field="expression") from exc
        variables = []
        cls._validate(tree.body, variables)
        variables = tuple(sorted(set(variables)))
        return cls(text, tree.body, variables)

    @classmethod
    def _validate(cls, node, variables):
        if not isinstance(node, _ALLOWED_NODES):
            raise InputError(
                "unsupported expression construct",
                field="expression",
                details={"construct": type(node).__name__},
            )
        if isinstance(node, ast.BinOp):
            if not isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
            ):
                raise InputError(
                    "unsupported expression operator",
                    field="expression",
                    details={"operator": type(node.op).__name__},
                )
            cls._validate(node.left, variables)
            cls._validate(node.right, variables)
            if isinstance(node.op, ast.Pow):
                cls._check_power(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise InputError(
                    "unsupported expression operator",
                    field="expression",
                    details={"operator": type(node.op).__name__},
                )
            cls._validate(node.operand, variables)
        elif isinstance(node, ast.Name):
            if not IDENTIFIER.fullmatch(node.id):
                raise InputError("invalid input name", field=node.id)
            variables.append(node.id)
        elif isinstance(node, ast.Constant):
            cls._check_constant(node)

    @staticmethod
    def _check_constant(node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise InputError(
                "only numeric constants are allowed", field="expression"
            )
        if not Decimal(str(node.value)).is_finite():
            raise InputError("numeric constant must be finite", field="expression")

    @staticmethod
    def _check_power(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            target = node.operand
            sign = -1 if isinstance(node.op, ast.USub) else 1
        else:
            target = node
            sign = 1
        if not isinstance(target, ast.Constant) or not isinstance(target.value, int):
            raise InputError("power exponent must be an integer literal")
        exponent = sign * target.value
        if not -_MAX_POWER <= exponent <= _MAX_POWER:
            raise InputError(
                "integer power exponent outside supported range",
                details={"exponent": exponent, "maximum": _MAX_POWER},
            )

    def as_dict(self):
        return {"text": self.text, "variables": list(self.variables)}

    def check_dimensions(self, input_dimensions):
        """Resolve the output dimension vector from declared input dimensions."""
        missing = [name for name in self.variables if name not in input_dimensions]
        if missing:
            raise InputError(
                "expression references undeclared inputs",
                details={"inputs": sorted(missing)},
            )
        return self._dimension(self.tree, input_dimensions)

    def _dimension(self, node, input_dimensions):
        if isinstance(node, ast.Name):
            return dict(input_dimensions[node.id])
        if isinstance(node, ast.Constant):
            return {}
        if isinstance(node, ast.UnaryOp):
            return self._dimension(node.operand, input_dimensions)
        left = self._dimension(node.left, input_dimensions)
        right = self._dimension(node.right, input_dimensions)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            dims.require_additive(left, right)
            return left
        if isinstance(node.op, ast.Mult):
            return dims.product(left, right)
        if isinstance(node.op, ast.Div):
            return dims.quotient(left, right)
        if isinstance(node.op, ast.Pow):
            return dims.power(left, self._exponent(node.right))
        raise InputError("unsupported expression operator")

    @staticmethod
    def _exponent(node):
        if isinstance(node, ast.UnaryOp):
            inner = ModelExpression._exponent(node.operand)
            return -inner if isinstance(node.op, ast.USub) else inner
        return int(node.value)

    def is_linear(self):
        """True when every variable appears to the first power only.

        Multiplication or division is nonlinear if both operands reference a
        variable; multiplication by a numeric constant and division by a
        numeric constant remain linear.  Any integer power other than one is
        nonlinear.
        """

        return self._linear(self.tree)

    def _linear(self, node):
        if isinstance(node, ast.Name) or isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.UnaryOp):
            return self._linear(node.operand)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return self._linear(node.left) and self._linear(node.right)
        if isinstance(node.op, ast.Mult):
            return not (
                self._references(node.left) and self._references(node.right)
            ) and self._linear(node.left) and self._linear(node.right)
        if isinstance(node.op, ast.Div):
            if self._references(node.right):
                return False
            return self._linear(node.left)
        if isinstance(node.op, ast.Pow):
            return False
        return False

    def _references(self, node):
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.UnaryOp):
            return self._references(node.operand)
        if isinstance(node, ast.BinOp):
            return self._references(node.left) or self._references(node.right)
        return False
