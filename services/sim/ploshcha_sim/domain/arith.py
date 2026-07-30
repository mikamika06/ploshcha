import ast
import operator

MAX_EXPONENT = 64
MAX_ABS = 10 ** 15
MAX_NODES = 200

BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

CONTRACT = "приймаю лише числа й дії + - * / // % ** ( )"


class ArithError(ValueError):
    pass


def evaluate(expr: str) -> float:
    """Арифметика без `eval` (борг 9, K10).

    Раніше калькулятор робив `eval(вираз)` із символьним фільтром. Фільтр не рятує: він пропускає
    те, що складається з дозволених символів, а не те, що безпечно. Тут парситься AST і виконуються
    ЛИШЕ арифметичні вузли — імена, виклики, атрибути, індексація неможливі структурно, а не за
    списком заборон. Плюс стелі на показник і величину, бо `9**9**9` — це DoS, а не вразливість.
    """
    if not expr or not expr.strip():
        raise ArithError(f"порожній вираз; {CONTRACT}")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ArithError(f"не арифметичний вираз: {exc.msg}; {CONTRACT}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ArithError(f"вираз завеликий ({len(nodes)} вузлів); {CONTRACT}")
    return _eval(tree.body)


def _eval(node) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ArithError(f"дозволені лише числа, отримано {type(node.value).__name__}")
        return _guard(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return _guard(UNARY[type(node.op)](_eval(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ArithError(f"показник понад {MAX_EXPONENT}")
            if right != int(right):
                raise ArithError("дробовий показник не підтримується")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ArithError("ділення на нуль")
        return _guard(BINARY[type(node.op)](left, right))
    raise ArithError(f"заборонена конструкція {type(node).__name__}; {CONTRACT}")


def _guard(value) -> float:
    if isinstance(value, complex) or abs(value) > MAX_ABS:
        raise ArithError(f"число поза межами ±{MAX_ABS}")
    return value
