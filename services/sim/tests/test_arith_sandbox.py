import pytest

from ploshcha_sim.compose import build_toolbox
from ploshcha_sim.domain.arith import MAX_ABS, MAX_EXPONENT, ArithError, evaluate
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.ports.tool import ToolCall

ATTACKS = [
    "__import__('os').system('echo pwned')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "globals()",
    "[x for x in range(10)]",
    "lambda: 1",
    "1 if True else 2",
    "print(1)",
    "os.system('ls')",
    "'a'*10",
    "9**9**9",
    "1/0",
]


@pytest.mark.parametrize("expr", ATTACKS)
def test_every_non_arithmetic_expression_is_refused(expr):
    """Борг 9 з першого спринта: раніше тут стояв eval() із символьним фільтром."""
    with pytest.raises(ArithError):
        evaluate(expr)


def test_the_expressions_the_items_actually_need_still_work():
    assert evaluate("347*892") == 309524
    assert evaluate("12345 + 67890") == 80235
    assert evaluate("144**2") == 20736, "badexpr-01 очікує саме це"
    assert evaluate("2**10") == 1024, "badexpr-02 очікує саме це"
    assert evaluate("1918-1648") == 270
    assert evaluate("(62+41+88)/2") == 95.5
    assert evaluate("-5 + 3") == -2


def test_a_character_filter_would_have_let_this_through():
    """Чому фільтр символів не є захистом: усе це складається з «дозволених» символів."""
    allowed = set("0123456789+-*/(). ")
    for expr in ("9**9**9", "1/0", "((((((1))))))" * 30):
        assert set(expr) <= allowed, expr
        with pytest.raises(ArithError):
            evaluate(expr)


def test_dos_ceilings_are_explicit():
    with pytest.raises(ArithError, match="показник"):
        evaluate(f"2**{MAX_EXPONENT + 1}")
    with pytest.raises(ArithError, match="межами"):
        evaluate(f"{MAX_ABS} * 10")
    with pytest.raises(ArithError, match="завеликий"):
        evaluate("1+" * 120 + "1")
    assert evaluate("2**40") == 2 ** 40, "показник у межах і результат у межах — рахуємо"
    with pytest.raises(ArithError, match="межами"):
        evaluate(f"2**{MAX_EXPONENT}"), "дві стелі незалежні: показник ок, величина ні"


def test_division_and_empty_are_named_not_crashed():
    with pytest.raises(ArithError, match="нуль"):
        evaluate("10 % 0")
    with pytest.raises(ArithError, match="порожній"):
        evaluate("   ")


def test_booleans_and_strings_are_not_numbers():
    with pytest.raises(ArithError):
        evaluate("True + 1")
    with pytest.raises(ArithError):
        evaluate("'2' + '2'")


@pytest.mark.parametrize("toolset,tool,arg", [
    ("default", "calc", "expr"),
    ("ua", "обчислити", "вираз"),
    ("registry", "обчислити", "вираз"),
    ("registry_teach", "обчислити", "вираз"),
])
def test_no_calculator_in_the_grid_still_executes_code(toolset, tool, arg):
    box = build_toolbox(AppSpec().with_(toolset=toolset))
    attack = box.call(ToolCall(tool=tool, args={arg: "__import__('os').system('ls')"}))
    assert not attack.ok, "виклик функції мусить бути відкинутий"
    assert "заборонена конструкція Call" in (attack.error or "")
    good = box.call(ToolCall(tool=tool, args={arg: "2+2"}))
    assert good.ok and list(good.value.values())[0] == 4


def test_the_teaching_variant_keeps_teaching():
    box = build_toolbox(AppSpec().with_(toolset="registry_teach"))
    bad = box.call(ToolCall(tool="обчислити", args={"вираз": "['зп-1', 'зп-2']"}))
    assert not bad.ok
    assert "62+41+88" in bad.error, "контракт мусить лишитись у тексті помилки"


def test_no_eval_left_in_the_adapters():
    import pathlib
    adapters = pathlib.Path(__file__).parents[1] / "ploshcha_sim" / "adapters"
    offenders = []
    for path in adapters.glob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if "eval(" in stripped and "evaluate(" not in stripped:
                offenders.append(f"{path.name}:{n}")
    assert not offenders, offenders
