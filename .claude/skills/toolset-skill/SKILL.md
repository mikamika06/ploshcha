---
name: toolset-skill
description: Додати інструмент або цілий набір інструментів у ToolPort ПЛОЩІ — оголошення SkillSpec, реєстрація в TOOLSETS, умова в evalkit і айтеми з gold/foil під гейт валідації.
---

# Новий інструмент у наборі

## Де що лежить

Порт — `services/sim/ploshcha_sim/ports/tool.py`: `ToolSpec`, `ToolCall`, `ToolResult` і сам
`ToolPort`. Там же живуть чотири різні способи віддати схему моделі: `wire_tool_schema`
(пласка), `strict_tool_schema` (`oneOf` по інструментах), `args_tool_schema` (аргументи одного
інструмента) і `choice_schema` (тільки вибір імені). Це не дублювання — це різні режими
виконавця, і `parse_*` до кожного з них свій. Коди помилок фіксовані рядками:
`unknown_tool`, `bad_args`, `not_json`, `no_tool_field`, `tool_field_forbidden`.

Реалізації — `services/sim/ploshcha_sim/adapters/tools_*.py`. Реєстр наборів — словник
`TOOLSETS` у `services/sim/ploshcha_sim/compose.py` (зараз чотирнадцять ключів, включно з
`none`, який лишає тільки `final_answer`).

## Оголошення форми

Інструмент — не лише функція. У `services/sim/ploshcha_sim/domain/skill.py` є `SkillSpec` із
полями `shape` (`scalar` / `aggregate` / `collection`), `side_effect` (`read` / `write`),
`trust` (`trusted` / `untrusted`), `cost_hint` і `max_items`. Оголошення робиться в
`adapters/skills_declared.py`, а обгортка, що лишає реєстр звичайним `ToolPort`, — у
`adapters/skillbox.py`.

Форма `collection` має заміряні наслідки, і їх треба знати заздалегідь. Без відстеження
залишку цикл робить `ITERATION_CEILING = 2` виклики — «у 104 прогонах з 104». З `coverage=True`
стеля `ITERATION_CEILING_COVERAGE = 8`, але «не надійно: 5 повних обходів із 9 задач, на 20
елементах обходу не спостерігалось жодного». Тому `shape_notes` пише причину в звіт **гучно,
але не корективно**: рішення лишається за застосунком.

## Обовʼязковий контракт результату

`ToolResult.found` — не декоративне поле. Воно розрізняє «інструмент відпрацював, але даних
немає» і «інструмент зламався», і на ньому стоять чесна відмова та рішення верифікатора про
наявність доказу. `None` означає «питання незастосовне» — інструмент не шукає, а обчислює.
Не повертай `ok=True, value=None` замість `found=False`.

## Вимірювальний бік

Новий набір без айтемів не існує для приладу. Айтеми — `services/sim/evalkit/items/*.jsonl`,
мапа «набір айтемів → дозволені набори інструментів» — `ITEM_SET_TOOLSETS` у
`services/sim/evalkit/validate.py`. Кожен айтем мусить мати щонайменше один результатний
предикат (інакше `success = all({})` дає `False` **тихо**), мати `gold` і мати `foil`, причому
**кожен** `foil` мусить провалитись: властивість `foil_vacuous` у `ItemReport` існує саме тому,
що доданий правильний антиеталон колись сховав хибний.

Умова, у якій набір міряється, додається в `services/sim/evalkit/conditions.py` через
`BASE.with_(...)`. Памʼятай, що `AppSpec` заморожений і має `sha256`: зміна полів існуючої
умови змінює відбиток і робить старі числа непорівнянними. Нова вісь — нова умова, не правка
старої.

## Заборони

Жодного `tools`-параметра на Gemma-3-похідних: шлюз його тихо викидає, і тул-токенів не буде.
Контракт цього проєкту — текстовий рендер плюс `json_schema`.

Жодного літерала промпту в коді ядра: промпт — окрема абляційна вісь, і йде через `prompt_id`
з реєстру `services/sim/evalkit/prompts/agent.jsonl`.
