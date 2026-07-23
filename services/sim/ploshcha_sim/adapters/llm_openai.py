"""OpenAI-сумісний адаптер (vLLM: Lapa/Mamay/Gemma; або хмара)."""

import time

from ..ports.llm import LlmPort, LlmResult, LlmUsage


class OpenAICompatLlm(LlmPort):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        structured_mode: str = "json_schema",
        guided_backend: str | None = "xgrammar",
    ):
        """structured_mode: json_schema (працює й на Lapathoniia) | json_object | guided (vLLM) | none."""
        from openai import OpenAI  # лінивий імпорт: тести не потребують пакета

        self.model = model
        self.structured_mode = structured_mode
        self.guided_backend = guided_backend
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _call(self, prompt, system, temperature, max_tokens, extra_body=None, response_format=None) -> LlmResult:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or {},
            **({"response_format": response_format} if response_format else {}),
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

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512) -> LlmResult:
        return self._call(prompt, system, temperature, max_tokens, None)

    def generate_structured(self, prompt, schema, *, system=None, temperature=0.0, max_tokens=512) -> LlmResult:
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
        return self._call(prompt, system, temperature, max_tokens, extra, rf)
