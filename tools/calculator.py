import ast
import operator
from typing import Any, Dict
from tools.base import BaseTool

class CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. Useful for doing calculations or conversions. "
            "Input must be a standard mathematical expression string like '500 * 133.33' or '(20 + 30) / 5'."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate (e.g., '500 * 133.4')"
                }
            },
            "required": ["expression"]
        }

    def _execute(self, expression: str, **kwargs) -> float:
        if not expression:
            raise ValueError("No expression provided.")
            
        # Clean expression of potential hazards
        expression = "".join(c for c in expression if c in "0123456789.+-*/^() ")
        
        # Replace caret with double asterisk for exponentiation in standard python
        expression = expression.replace("^", "**")
        
        # Supported operators
        safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Num):  # Python < 3.8 fallback
                return node.n
            elif isinstance(node, ast.Constant):  # Python >= 3.8
                return node.value
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type in safe_operators:
                    return safe_operators[op_type](left, right)
                raise TypeError(f"Unsupported binary operator: {op_type.__name__}")
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_type = type(node.op)
                if op_type in safe_operators:
                    return safe_operators[op_type](operand)
                raise TypeError(f"Unsupported unary operator: {op_type.__name__}")
            else:
                raise TypeError(f"Unsupported expression syntax element: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval(tree)
            return float(result)
        except Exception as e:
            raise ValueError(f"Invalid math expression '{expression}': {str(e)}")
