import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from ploshcha_sim.adapters import FakeToolbox
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm

SYS = {
    "uk": ("Ти агент з інструментами: check_date(year,event), lookup_fact(entity), "
           "calc(expr), final_answer(text). На КОЖНОМУ кроці поверни РІВНО один JSON — або виклик "
           "інструмента, або final_answer. Використовуй результати попередніх викликів. "
           "Якщо жоден інструмент не потрібен — одразу final_answer."),
    "en": ("You are a tool agent: check_date(year,event), lookup_fact(entity), calc(expr), "
           "final_answer(text). At EACH step return EXACTLY one JSON — either a tool call or "
           "final_answer. Use previous results. If no tool is needed — final_answer right away."),
}

DATA_TOOLS = {"check_date", "lookup_fact", "calc"}

SINGLE = {
    "uk": [
        ("Знайди факт про Тараса Шевченка.", "lookup_fact", "Шевченк"),
        ("Що відомо про Івана Мазепу?", "lookup_fact", "Мазеп"),
        ("Перевір, чи Битва під Крутами була 1918 року.", "check_date", "1918"),
        ("Чи 1648 — рік початку Хмельниччини?", "check_date", "1648"),
        ("Скільки буде 17 мінус 9?", "calc", "17"),
        ("Обчисли 6 помножити на 7.", "calc", "6"),
    ],
    "en": [
        ("Find a fact about Taras Shevchenko.", "lookup_fact", "Shevchenko"),
        ("What is known about Ivan Mazepa?", "lookup_fact", "Mazepa"),
        ("Check whether the Battle of Kruty was in 1918.", "check_date", "1918"),
        ("Is 1648 the year Khmelnytsky uprising began?", "check_date", "1648"),
        ("How much is 17 minus 9?", "calc", "17"),
        ("Compute 6 times 7.", "calc", "6"),
    ],
}

ABSTAIN = {
    "uk": ["Як тебе звати?", "Розкажи короткий жарт.", "Яка твоя улюблена пора року?"],
    "en": ["What is your name?", "Tell a short joke.", "What is your favourite season?"],
}

MULTI = {
    "uk": [
        ("Дізнайся факт про Битву під Крутами, а потім заверши відповіддю.", {"lookup_fact"}),
        ("Перевір, чи Хмельниччина почалась 1648, і заверши словом «вірно» або «ні».", {"check_date"}),
    ],
    "en": [
        ("Look up a fact about the Battle of Kruty, then finish with an answer.", {"lookup_fact"}),
        ("Check if the Khmelnytsky uprising began in 1648, then finish.", {"check_date"}),
    ],
}


def safe_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def render(task, history):
    lines = [f"Задача: {task}"]
    for call, result in history:
        lines.append(f"Виклик: {json.dumps({'tool': call.tool, **call.args}, ensure_ascii=False)}")
        lines.append(f"Результат: {json.dumps(result.value, ensure_ascii=False)}")
    lines.append("Наступний крок (один JSON):")
    return "\n".join(lines)


def run_task(llm, box, task, lang, max_steps=4, seed=1):
    history = []
    calls = []
    for _ in range(max_steps):
        res = llm.generate_structured(render(task, history), box.wire_schema(), system=SYS[lang], seed=seed)
        call, reason = box.parse(safe_json(res.text))
        if call is None:
            calls.append(("PARSE_FAIL:" + str(reason), {}))
            break
        calls.append((call.tool, call.args))
        if call.tool == "final_answer":
            break
        result = box.call(call)
        history.append((call, result))
    return calls


def first_data_tool(calls):
    for tool, _ in calls:
        if tool in DATA_TOOLS:
            return tool, _
    return None, None


def run(label, model, base_url, api_key, lang):
    box = FakeToolbox()
    llm = OpenAICompatLlm(model=model, base_url=base_url, api_key=api_key, structured_mode="json_schema")
    print(f"\n{'='*78}\n{label} [{lang}]   {model}\n{'='*78}")

    sel = arg = 0
    for task, gold_tool, gold_sub in SINGLE[lang]:
        calls = run_task(llm, box, task, lang)
        tool, args = first_data_tool(calls)
        sel += tool == gold_tool
        arg += tool == gold_tool and gold_sub.lower() in json.dumps(args, ensure_ascii=False).lower()
    n = len(SINGLE[lang])
    print(f"  single:   вибір тула {sel}/{n}, аргумент грунтований {arg}/{n}")

    abst = 0
    for task in ABSTAIN[lang]:
        calls = run_task(llm, box, task, lang)
        abst += calls[0][0] == "final_answer"
    print(f"  abstain:  одразу final_answer {abst}/{len(ABSTAIN[lang])} (не чіпає data-тули)")

    ms = 0
    for task, need in MULTI[lang]:
        calls = run_task(llm, box, task, lang)
        used = {t for t, _ in calls}
        ms += need <= used and calls[-1][0] == "final_answer"
    print(f"  multi:    ланцюг+final {ms}/{len(MULTI[lang])}")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    for var in ("LAPA_MODEL", "MAMAY_MODEL"):
        if os.environ.get(var):
            for lang in ("uk", "en"):
                run(var, os.environ[var], url, key, lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
