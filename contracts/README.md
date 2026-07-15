# Контракт подій ПЛОЩА (v1)

**Це джерело правди** для фронту (`packages/contract-ts`) і майбутнього беку (Pydantic).
Фронт доробляється **проти цього контракту + фікстур** ще до написання беку (ADR-0003);
бек Фази 4 лише «поважає» контракт. Дрейф ловиться CI-гейтом паритету.

- Схема подій: [`ploshcha-events.schema.json`](ploshcha-events.schema.json)
- Схема сцени: [`scene.schema.json`](scene.schema.json)
- Версія протоколу: [`VERSION`](VERSION) (semver) — `1.0.0`

## Конверт (Envelope)

Кожна подія має спільний конверт:

| Поле | Тип | Сенс |
|---|---|---|
| `protocol` | semver-рядок | версія контракту (`1.0.0`) |
| `runId` | рядок `[A-Za-z0-9._-]{1..64}` | ідентифікатор прогону |
| `seq` | ціле ≥0 | **монотонний** лічильник у прогоні → детекція розривів/реордеру |
| `ts` | ISO-8601 | час події |
| `tick` | ціле ≥0 | крок симуляції (час дня) |
| `type` | рядок | **дискримінатор** типу |
| `payload` | обʼєкт | типізований за `type` |
| `note?` | рядок | необовʼязкова примітка |

## 13 типів подій

| type | payload (стисло) | що рендерить |
|---|---|---|
| `run.started` | `config, scene, startedAt` | Оповідач відкриває день; завантаження сцени |
| `casting.begin` | `mode` | екран кастингу |
| `casting.done` | `cast: VillagerPublic[]` | спавн селян за ролями |
| `tick.begin` | `timeOfDay, mood?` | Дзвін відбиває час; погода настрою |
| `plan.formed` | `agentId, summary, steps?` | селянин отримує денний маршрут |
| `agent.moved` | `agentId, to: PlaceRef, activity?` | рух до POI + дія |
| `utterance.spoken` | `agentId, to?, text, place?, tone?` | бульбашка + рядок у чаті |
| `event.happened` | `event: VillageEvent` | Дошка/Дзвін оголошують подію дня |
| `reflection.formed` | `agentId, thought` | серпанок-думка Хору над селянином |
| `report.compiled` | `chronicle: DayChronicle` | хроніка дня + настрій + зміни стосунків |
| `run.degraded` | `stage, reason?` | UI-стан деградації (не креш) |
| `run.done` | `ticks, tokens, counts` | завершення прогону |
| `run.error` | `message` (санітизовано) | UI-стан помилки |

## Дві проєкції (ADR-0002)

- **spectator** (за замовч.) — **видима поведінка**: рух, репліки, події, настрій, і
  один рядок-думка в `reflection.formed`. Без внутрішньої когніції.
- **analytics** (auth-gated) — додає памʼять, повні beliefs і міркування рефлексії,
  повний план, retrieval-сліди. Для дослідження/eval/інспекції.

Схема описує spectator-форму. Analytics — надмножина (додаткові поля present лише в ній);
бек серіалізує на межі `contract/` (`to_spectator()`/`to_analytics()`).

## Правила еволюції

- `type`-ключі, `poi.kind`, `event.kind` — **машинні**; людські підписи — у payload-полях
  (`name`/`label`/`text`/`narration`, українською).
- Зміна форми = бамп `VERSION` (semver). Мінорні — тільки **додавання опційних** полів.
- Кожен рядок фікстури (`packages/fixtures/runs/*.jsonl`) мусить валідуватися проти
  `PloshchaEvent` (zod) — це і є контракт-конформність.

## Транспорт

- Живий бек: SSE. `POST /runs` → `{runId}`; `GET /runs/{runId}/stream` — потік конвертів
  (по одному в `data:`), `event:` дублює `type`.
- Фікстури: той самий конверт, **по одному JSON на рядок** (`*.jsonl`).
- Один інтерфейс `EventSourcePort` на фронті з драйверами `SseDriver` / `FixtureDriver`.
