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

    def _call(self, prompt, system, temperature, max_tokens, extra_body=None, response_format=None,
              seed=None, repetition_penalty=None) -> LlmResult:
        """★ `repetition_penalty` ДОЇЖДЖАЄ ДО ШЛЮЗУ — і старий запис про це був артефактом кешу.

        Доти в трьох місцях репозиторію стояло, що шлюз Lapathoniia ковтає всі штрафи семплера
        («вивід не змінився ані на символ»). Для `frequency_penalty`, `presence_penalty`,
        `no_repeat_ngram_size`, `bad_words` і `dry_*` це підтвердилось удруге, а для
        `repetition_penalty` — ні, і причина в самому способі заміру: **шлюз КЕШУЄ відповіді, а
        `extra_body` у ключ кешу не входить**. Наївна проба (той самий пакет віча, `temperature=0`,
        `seed=1`, девʼять плечей — `rp` 1.0/1.15/1.3/2.0, `top_k`, `min_p`, вигаданий параметр)
        дала одну й ту саму суму `42fb002637e9ca2e` на всіх девʼятьох, зокрема на невалідних
        `rp=0.0` і `rp=-1.0`, і без жодного 400. Кеш доведено й часом: той самий запит удруге —
        2252 мс проти 85 мс, побайтово однаково.

        Проба з розбитим кешем (мітка часу в промпті, примусовий повтор) показала протилежне:
        `rp=5.0` першим викликом дало 6 «вовків» і побиту абетку («Вовк، Вовк， вовк ، …»), `rp=1.0`
        першим — 10 «вовків» чистим повтором, а ДРУГИЙ виклик у кожній парі вертав байт-у-байт
        перший, хоч штраф у ньому був інший. Тобто вивід залежить від штрафу ПЕРШОГО виклику на
        свіжому промпті — важіль живий, ковтає його кеш, а не шлюз. Драбина на свіжих промптах
        (Lapa 12B): 1.0-1.3 не міняють нічого, кусати починає з 1.5.

        Тією ж пробою: `stop` теж доїжджає, а `guided_json`/`guided_choice` — ні, МОВЧКИ, тобто
        `structured_mode="guided"` на цьому шлюзі мертва гілка (прод на `json_schema`).

        Поле лишається вимкненим (`None` — нічого не слати, запит байт-у-байт той самий): на
        справжніх пакетах віча плата за зменшення дослівного повернення виявилась один-до-одного,
        тож умикати його в прод-умові `viche` нема за що (`evalkit/conditions.py`).
        """
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        # Штраф лягає ПОРУЧ зі схемою, а не замість неї: `extra_body` тут єдиний канал і для
        # constrained decoding, і для важелів семплера, тож змішувати їх треба тут, а не в порті.
        body = dict(extra_body or {})
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty
        t0 = time.perf_counter()
        resp = self._with_retry(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=body,
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
            # Структурованість міряється СХЕМОЮ, а не повнотою `extra_body`: інакше сам лише
            # штраф семплера робив би вільну генерацію «структурованою» у звіті.
            structured=bool(extra_body or response_format),
            finish_reason=resp.choices[0].finish_reason,
            rendered={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "response_format": response_format,
                "extra_body": body,
            },
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

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512, seed=None,
                 repetition_penalty=None) -> LlmResult:
        return self._call(prompt, system, temperature, max_tokens, None, seed=seed,
                          repetition_penalty=repetition_penalty)

    def generate_structured(self, prompt, schema, *, system=None, temperature=0.0, max_tokens=512,
                            seed=None, repetition_penalty=None) -> LlmResult:
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
        return self._call(prompt, system, temperature, max_tokens, extra, rf, seed=seed,
                          repetition_penalty=repetition_penalty)
