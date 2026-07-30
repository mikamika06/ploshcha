"""OpenAI-сумісний адаптер (vLLM: Lapa/Mamay/Gemma; або хмара)."""

import time

from ..ports.llm import LlmPort, LlmResult, LlmUsage

RETRIES = 4
BACKOFF = 2.0
RETRIABLE = (429, 500, 502, 503, 504)


class OpenAICompatLlm(LlmPort):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        structured_mode: str = "json_schema",
        guided_backend: str | None = "xgrammar",
        retries: int = RETRIES,
        sleep=time.sleep,
    ):
        """structured_mode: json_schema (працює й на Lapathoniia) | json_object | guided (vLLM) | none."""
        from openai import OpenAI  # лінивий імпорт: тести не потребують пакета

        self.model = model
        self.structured_mode = structured_mode
        self.guided_backend = guided_backend
        self.retries = retries
        self.retried = 0
        self._sleep = sleep
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _call(self, prompt, system, temperature, max_tokens, extra_body=None, response_format=None, seed=None) -> LlmResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        t0 = time.perf_counter()
        resp = self._with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or {},
                **({"seed": seed} if seed is not None else {}),
                **({"response_format": response_format} if response_format else {}),
            )
        )
        latency = int((time.perf_counter() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        return LlmResult(
            text=resp.choices[0].message.content or "",
            model=self.model,
            usage=LlmUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency,
            structured=bool(extra_body or response_format),
            finish_reason=resp.choices[0].finish_reason,
        )

    def _with_retry(self, call):
        """Один транзієнтний 502 не має вбивати півгодинний прогін.

        Спільний шлюз Lapathoniia періодично віддає 429/5xx. Без ретраю ми вже втратили два набори
        регрес-свіпу; при цьому НЕ ретраїмо 4xx-помилки контракту (400/401/404) — вони означають
        нашу помилку, і глушити їх було б гірше, ніж упасти.
        """
        delay = BACKOFF
        for attempt in range(self.retries + 1):
            try:
                return call()
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None)
                if status not in RETRIABLE or attempt == self.retries:
                    raise
                self.retried += 1
                self._sleep(delay)
                delay *= BACKOFF

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512, seed=None) -> LlmResult:
        return self._call(prompt, system, temperature, max_tokens, None, seed=seed)

    def generate_structured(self, prompt, schema, *, system=None, temperature=0.0, max_tokens=512, seed=None) -> LlmResult:
        extra: dict = {}
        rf: dict | None = None
        if self.structured_mode == "json_schema":
            rf = {
                "type": "json_schema",
                "json_schema": {"name": "action", "schema": schema, "strict": True},
            }
        elif self.structured_mode == "json_object":
            rf = {"type": "json_object"}
        elif self.structured_mode == "guided":
            extra["guided_json"] = schema
            if self.guided_backend:
                extra["guided_decoding_backend"] = self.guided_backend
        return self._call(prompt, system, temperature, max_tokens, extra, rf, seed=seed)
