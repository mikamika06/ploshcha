import pytest

from ploshcha_sim.adapters.llm_openai import RETRIABLE, OpenAICompatLlm


class Boom(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def _llm(retries=4):
    llm = OpenAICompatLlm.__new__(OpenAICompatLlm)
    llm.retries = retries
    llm.retried = 0
    llm.slept = []
    llm._sleep = llm.slept.append
    return llm


@pytest.mark.parametrize("status", RETRIABLE)
def test_transient_gateway_errors_are_retried(status):
    """Один 502 уже вбив два набори регрес-свіпу — саме це й лікуємо."""
    llm = _llm()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise Boom(status)
        return "готово"

    assert llm._with_retry(flaky) == "готово"
    assert len(calls) == 3 and llm.retried == 2


@pytest.mark.parametrize("status", (400, 401, 403, 404, 422))
def test_contract_errors_are_not_retried(status):
    """4xx означає НАШУ помилку; глушити її ретраями гірше, ніж упасти."""
    llm = _llm()
    with pytest.raises(Boom):
        llm._with_retry(lambda: (_ for _ in ()).throw(Boom(status)))
    assert llm.retried == 0


def test_retries_are_bounded_and_backoff_grows():
    llm = _llm(retries=3)
    with pytest.raises(Boom):
        llm._with_retry(lambda: (_ for _ in ()).throw(Boom(502)))
    assert llm.retried == 3, "рівно стільки спроб, скільки оголошено"
    assert llm.slept == [2.0, 4.0, 8.0], llm.slept


def test_success_on_the_first_call_sleeps_nothing():
    llm = _llm()
    assert llm._with_retry(lambda: 42) == 42
    assert llm.retried == 0 and llm.slept == []


def test_an_error_without_a_status_is_not_swallowed():
    llm = _llm()
    with pytest.raises(ValueError):
        llm._with_retry(lambda: (_ for _ in ()).throw(ValueError("щось інше")))
    assert llm.retried == 0
