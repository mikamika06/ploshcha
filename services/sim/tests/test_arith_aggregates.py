import pytest

from ploshcha_sim.domain.arith import MAX_ITEMS, ArithError, evaluate


@pytest.mark.parametrize("expr,expected", [
    ("sum([29, 58, 91])", 178),
    ("sum([29,27,31])", 87),
    ("min([5, 2, 9])", 2),
    ("max([5, 2, 9])", 9),
    ("len([1, 2, 3])", 3),
    ("sum((1, 2, 3))", 6),
    ("sum([1.5, 2.5])", 4.0),
    ("sum([29, 58, 91]) - 78", 100),
    ("sum([2, 3]) * len([1, 1, 1])", 15),
    ("sum([-1, -2])", -3),
])
def test_aggregates_work(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os')",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "sum",
    "abs(-1)",
    "print([1])",
    "sum([1].__class__)",
    "sum([1]).real",
    "[1, 2, 3]",
    "sum([1, 2], [3])",
    "sum(x=[1])",
    "sum([1, 'два'])",
    "sum([True, False])",
    "sum(range(3))",
    "sum([sum])",
    "getattr(sum, 'x')",
])
def test_dangerous_and_malformed_still_rejected(expr):
    with pytest.raises(ArithError):
        evaluate(expr)


def test_empty_list_rejected():
    with pytest.raises(ArithError):
        evaluate("sum([])")


def test_too_many_items_rejected():
    with pytest.raises(ArithError):
        evaluate("sum([" + ",".join("1" for _ in range(MAX_ITEMS + 1)) + "])")


def test_nested_aggregate_of_numbers_is_allowed():
    assert evaluate("sum([sum([1, 2]), 3])") == 6


def test_error_message_teaches_the_contract():
    with pytest.raises(ArithError) as exc:
        evaluate("total(записи)")
    text = str(exc.value)
    assert "sum" in text
    assert "sum([29, 58, 91])" in text


def test_plain_arithmetic_unchanged():
    assert evaluate("144*144") == 20736
    assert evaluate("29 + 58 + 91") == 178
    with pytest.raises(ArithError):
        evaluate("144^2")
