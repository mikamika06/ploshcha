"""Важелі ДЕКОДУВАННЯ: що з них доїжджає до шлюзу і як воно проведене в ядро.

Файл існує через одну поламку методу, а не через один параметр. Три круги в репозиторії стояв
запис «шлюз ковтає штрафи семплера, вивід не змінився ані на символ» — і для `repetition_penalty`
він виявився АРТЕФАКТОМ КЕШУ: шлюз Lapathoniia кешує відповіді, а `extra_body` у ключ кешу не
входить, тож девʼять плечей на одному пакеті віча (`rp` 1.0/1.15/1.3/2.0, `top_k`, `min_p`,
вигаданий параметр і навіть невалідні `rp=0.0`/`rp=-1.0`) вернули ту саму суму `42fb002637e9ca2e`
без жодного 400. Живого шлюзу тут немає й бути не може, тому тести стережуть те, що взагалі
перевіряється офлайн: важіль доїжджає ДО КАНАЛУ в потрібній формі, а вимкнений — не міняє запиту
ані на поле.
"""

from types import SimpleNamespace

from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
from ploshcha_sim.compose import VICHE_KWARGS, build_viche
from ploshcha_sim.domain.spec import AppSpec
from rule_llm import RuleLlm

SCHEMA = {"type": "object", "properties": {"репліка": {"type": "string"}}}


class Client:
    """Найтонший двійник шлюзу: запамʼятовує тіло запиту, вертає порожню відповідь."""

    def __init__(self):
        self.sent: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.sent.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def _llm(mode="json_schema"):
    llm = OpenAICompatLlm.__new__(OpenAICompatLlm)
    llm.model, llm.structured_mode, llm.guided_backend = "fake", mode, None
    llm.retries, llm.retried, llm._sleep = 0, 0, lambda _: None
    llm._client = Client()
    return llm, llm._client


def test_the_penalty_reaches_the_gateway_inside_extra_body():
    """Форма важлива не менше за сам факт: vLLM бере штраф ТІЛЬКИ через `extra_body`.

    Кеш-розбита проба (мітка часу в промпті, примусовий повтор, `temperature=0`, `seed=1`)
    показала, що саме в цій формі він живий: `rp=5.0` першим викликом дало 6 «вовків» і побиту
    абетку, `rp=1.0` першим — 10 «вовків» чистим повтором, а другий виклик у кожній парі вернув
    байт-у-байт перший. Через кореневі поля запиту (`frequency_penalty`, `presence_penalty`) той
    самий шлюз не бере нічого — це підтверджено вдруге.
    """
    llm, client = _llm()
    llm.generate_structured("п", SCHEMA, repetition_penalty=1.15)
    assert client.sent[0]["extra_body"] == {"repetition_penalty": 1.15}
    assert "repetition_penalty" not in client.sent[0], "кореневим полем шлюз штрафу не бере"
    assert client.sent[0]["response_format"]["type"] == "json_schema", "схема лишається на місці"


def test_the_penalty_rides_next_to_guided_decoding_without_evicting_it():
    """`extra_body` — єдиний канал і для схеми, і для штрафу, тож змішувати їх треба, не міняти."""
    llm, client = _llm(mode="guided")
    llm.generate_structured("п", SCHEMA, repetition_penalty=1.3)
    assert client.sent[0]["extra_body"] == {"guided_json": SCHEMA, "repetition_penalty": 1.3}


def test_the_switch_off_leaves_the_request_byte_for_byte_the_same():
    """Дефолт мусить лишати вже пораховані прогони порівнюваними — тобто не слати поля взагалі.

    Не `repetition_penalty: null` і не `1.0`: обидва — це поле в тілі запиту, а поле в тілі
    запиту міняє те, що бачить шлюз, і тим самим право порівнювати старі звіти з новими.
    """
    llm, client = _llm()
    llm.generate_structured("п", SCHEMA)
    llm.generate("п")
    assert client.sent[0]["extra_body"] == {}
    assert client.sent[1]["extra_body"] == {}


def test_the_free_generation_stays_unstructured_even_with_a_penalty():
    """Структурованість міряється СХЕМОЮ, а не повнотою `extra_body`.

    Інакше сам лише важіль семплера робив би вільну генерацію «структурованою» у звіті, і лічба
    tier-ів у замірах поїхала б від правки, яка до схеми не має стосунку.
    """
    llm, _ = _llm()
    assert llm.generate("п", repetition_penalty=1.15).structured is False
    assert llm.generate_structured("п", SCHEMA, repetition_penalty=1.15).structured is True


def test_the_rendered_request_shows_the_penalty_that_was_actually_sent():
    """`rendered` — єдине, чим замір доводить, ЩО поїхало в шлюз; здогад тут не годиться."""
    llm, _ = _llm()
    res = llm.generate_structured("п", SCHEMA, repetition_penalty=1.15)
    assert res.rendered["extra_body"]["repetition_penalty"] == 1.15


def test_the_penalty_is_an_axis_of_the_run_not_an_ornament():
    """Поле мусить рухати `sha256` умови: інакше два різні прогони звітують під одним іменем."""
    spec = AppSpec(mode="viche")
    assert spec.viche_repetition_penalty is None, "дефолт зберігає теперішню поведінку"
    assert spec.sha256 != spec.with_(viche_repetition_penalty=1.15).sha256


def test_the_composition_root_carries_the_penalty_into_the_viche():
    """Магічного числа всередині агента бути не може: важіль приходить лише зі специфікації."""
    lapa, mamay = RuleLlm("lapa"), RuleLlm("mamay")
    spec = AppSpec(mode="viche")
    assert "repetition_penalty" in VICHE_KWARGS
    assert build_viche(spec, lapa=lapa, mamay=mamay).repetition_penalty is None
    tuned = build_viche(spec.with_(viche_repetition_penalty=1.15), lapa=lapa, mamay=mamay)
    assert tuned.repetition_penalty == 1.15


def test_the_production_condition_leaves_the_lever_off():
    """Вимкнено ЗА ЗАМІРОМ: на свіжих промптах 1.0-1.3 не міняють нічого (кусати починає з 1.5), а
    на справжніх пакетах віча плата один-до-одного. Число сюди дописують після заміру, який
    покаже виграш, а не обмін."""
    from evalkit.conditions import CONDITIONS

    assert CONDITIONS["viche"].viche_repetition_penalty is None
    assert CONDITIONS["viche-notools"].viche_repetition_penalty is None
