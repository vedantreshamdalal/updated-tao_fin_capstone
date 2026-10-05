# backend/app/services/calculator.py

"""
Calculator

Safe arithmetic for financial checks -- no eval(). We only ever
allow numbers and + - * / ( ) via an AST walk, so nothing the LLM
or a user emits can execute arbitrary code.
"""

import ast
import operator
import re
from typing import Optional


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    pass


def safe_eval(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculatorError(f"Could not parse expression: {expression}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise CalculatorError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise CalculatorError(f"Unsupported expression element: {ast.dump(node)}")

    try:
        return _eval(tree)
    except ZeroDivisionError as e:
        raise CalculatorError("Division by zero") from e


def growth_rate(old_value: float, new_value: float) -> float:
    if old_value == 0:
        raise CalculatorError("Cannot compute growth rate from a zero base value")
    return (new_value - old_value) / old_value * 100


def margin(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise CalculatorError("Cannot compute margin with a zero denominator")
    return numerator / denominator * 100


def cagr(begin_value: float, end_value: float, years: float) -> float:
    if begin_value <= 0 or years <= 0:
        raise CalculatorError("CAGR requires a positive base value and positive years")
    return ((end_value / begin_value) ** (1 / years) - 1) * 100


_NUMBER_PATTERN = re.compile(
    r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?"
)


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER_PATTERN.findall(text):
        cleaned = match.replace("$", "").replace(",", "").replace("%", "")
        if cleaned in ("", "-", "+", "."):
            continue
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def numbers_approximately_match(a: float, b: float, tolerance: float = 0.01) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tolerance