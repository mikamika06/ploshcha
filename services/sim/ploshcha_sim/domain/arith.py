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

AGGREGATES = {
    "sum": lambda xs: sum(xs),
    "min": lambda xs: min(xs),
    "max": lambda xs: max(xs),
    "len": lambda xs: float(len(xs)),
}
MAX_ITEMS = 500

CONTRACT = ("приймаю лише числа й дії + - * / // % ** ( ), "
            "а також згортки sum/min/max/len над списком чисел, напр. sum([29, 58, 91])")


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
    if isinstance(node, ast.Call):
        return _aggregate(node)
    raise ArithError(f"заборонена конструкція {type(node).__name__}; {CONTRACT}")


def _aggregate(node) -> float:
    if not isinstance(node.func, ast.Name):
        raise ArithError(f"заборонена конструкція Call; {CONTRACT}")
    if node.func.id not in AGGREGATES:
        raise ArithError(f"невідома згортка {node.func.id}; {CONTRACT}")
    if node.keywords or len(node.args) != 1:
        raise ArithError(f"згортка бере рівно один список; {CONTRACT}")
    arg = node.args[0]
    if not isinstance(arg, (ast.List, ast.Tuple)):
        raise ArithError(f"аргумент згортки мусить бути списком чисел; {CONTRACT}")
    if len(arg.elts) > MAX_ITEMS:
        raise ArithError(f"у списку понад {MAX_ITEMS} елементів")
    if not arg.elts:
        raise ArithError("порожній список")
    values = [_eval(el) for el in arg.elts]
    return _guard(AGGREGATES[node.func.id](values))


def _guard(value) -> float:
    if isinstance(value, complex) or abs(value) > MAX_ABS:
        raise ArithError(f"число поза межами ±{MAX_ABS}")
    return value
