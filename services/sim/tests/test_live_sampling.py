import hashlib
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[3]


def _load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _llm():
    _load_env()
    base_url = os.environ.get("PLOSHCHA_LIVE_URL") or os.environ.get("LAPA_BASE_URL")
    model = os.environ.get("PLOSHCHA_LIVE_MODEL") or os.environ.get("LAPA_MODEL")
    key = os.environ.get("LAPA_API_KEY", "EMPTY")
    if not base_url or not model:
        pytest.skip("нема endpoint: задай PLOSHCHA_LIVE_URL/PLOSHCHA_LIVE_MODEL або LAPA_*")
    from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
    return OpenAICompatLlm(model=model, base_url=base_url, api_key=key, timeout=300.0)


PROMPT = "Опиши одним реченням українське село навесні."


def _h(llm, temperature, seed):
    return hashlib.sha256(
        llm.generate(PROMPT, temperature=temperature, max_tokens=64, seed=seed).text.encode()
    ).hexdigest()


def test_greedy_ignores_seed():
    llm = _llm()
    assert _h(llm, 0.0, 1) == _h(llm, 0.0, 2) == _h(llm, 0.0, 3)


def test_sampling_same_seed_reproduces():
    llm = _llm()
    assert _h(llm, 0.8, 7) == _h(llm, 0.8, 7)


def test_sampling_different_seeds_diverge():
    llm = _llm()
    assert len({_h(llm, 0.8, s) for s in (1, 2, 3, 4)}) > 1
