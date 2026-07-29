import json

from ploshcha_sim.ports.llm import LlmPort, LlmResult, LlmUsage

FREE_ANSWER = "Битва під Крутами відбулася 1918 року."
DONE_MARKS = ("Результат:", "Виклик:")
FIRST_CALL = {
    "check_date": '{"tool":"check_date","event":"Битва під Крутами","year":1918}',
    "перевірити_дату": '{"tool":"перевірити_дату","подія":"Битва під Крутами","рік":1918}',
}
FINAL = '{"tool":"final_answer","text":"1918 рік"}'


class RuleLlm(LlmPort):
    """Детермінована модель-правило: вивід залежить ЛИШЕ від промпту й схеми.

    Потрібна для паритетних фікстур: скриптована FakeLlm вичерпується між прогонами,
    а тут стану немає взагалі, тому 17 умов можна прогнати без порядкового звʼязку.
    Назви інструментів приходять у схемі, а не в тексті промпту, тому правило дивиться в схему.
    """

    def __init__(self, model: str = "rule", repeat: bool = False):
        self.model = model
        self.repeat = repeat
        self.calls: list[dict] = []

    def _reply(self, prompt: str, structured: bool, schema: dict | None) -> str:
        if not structured:
            return FREE_ANSWER
        if any(mark in prompt for mark in DONE_MARKS) and not self.repeat:
            return FINAL
        wire = json.dumps(schema, ensure_ascii=False) if schema else ""
        for name, call in FIRST_CALL.items():
            if name in wire:
                return call
        return FINAL

    def _run(self, prompt, system, structured, schema, seed, temperature) -> LlmResult:
        self.calls.append({"structured": structured, "seed": seed, "temperature": temperature})
        text = self._reply(prompt, structured, schema)
        return LlmResult(
            text=text, model=self.model,
            usage=LlmUsage(prompt_tokens=len(prompt.split()), completion_tokens=len(text.split())),
            latency_ms=0, structured=structured, finish_reason="stop",
        )

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=512, seed=None):
        return self._run(prompt, system, False, None, seed, temperature)

    def generate_structured(self, prompt, schema, *, system=None, temperature=0.0,
                            max_tokens=512, seed=None):
        return self._run(prompt, system, True, schema, seed, temperature)
