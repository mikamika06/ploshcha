import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from evalkit.conditions import CONDITIONS  # noqa: E402
from evalkit.prompts import resolve  # noqa: E402
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.adapters.decisions_sqlite import SqliteDecisions  # noqa: E402
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue  # noqa: E402
from ploshcha_sim.compose import (  # noqa: E402
    build_budget, build_effort, build_graph, build_orchestrator, build_router, build_viche)
from ploshcha_sim.domain.governor import Governor  # noqa: E402
from ploshcha_sim.adapters.village_sqlite import SqliteVillage  # noqa: E402
from ploshcha_sim.agents.forge import forge_village  # noqa: E402
from ploshcha_sim.domain.viche import PERSONAS, public_cast  # noqa: E402
from ploshcha_sim.live import EventBus, LiveRunner, serve  # noqa: E402

SCENE = {"id": "ploshcha", "name": "Площа"}
# Сід села. Змінити — інше село; те саме число — ті самі сусіди.
VILLAGE_SEED = int(os.environ.get("PLOSHCHA_VILLAGE_SEED", "11"))
DEFAULT_CONDITION = "viche"


def parse_args(argv):
    p = argparse.ArgumentParser(description="ПЛОЩА онлайн: SSE-потік живого прогону пари")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--condition", default=DEFAULT_CONDITION,
                   help="умова з evalkit.conditions — профіль пари (дефолт: профіль Я3)")
    p.add_argument("--max-tokens", type=int, default=200_000, help="стеля токенів на весь прогін")
    p.add_argument("--max-usd", type=float, default=1.0)
    p.add_argument("--max-items", type=int, default=50)
    p.add_argument("--db", default=str(ROOT / "docs" / "research" / "eval-runs" / "ploshcha.db"))
    p.add_argument("--kill-file", default=str(ROOT / "STOP"))
    p.add_argument("--seed-topic", action="append", default=[],
                   help="тема, покладена в чергу на старті; можна кілька разів")
    p.add_argument("--static", default=str(ROOT / "apps" / "web" / "dist"),
                   help="корінь збірки фронта; порожньо — не роздавати статику")
    p.add_argument("--resume", action="store_true",
                   help="стартувати одразу; за замовчуванням сервер стоїть на ПАУЗІ")
    return p.parse_args(argv)


def build_live(*, condition: str, max_tokens: int, max_usd: float, max_items: int,
               db: str, kill_file: str | None = None, paused: bool = True):
    """Збірка живого циклу — спільна для сервера і для соак-заміру, щоб проводка не дублювалась."""
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key or not url:
        raise RuntimeError("нема LAPA_API_KEY / LAPA_BASE_URL у .env")
    spec = CONDITIONS.get(condition)
    if spec is None:
        raise KeyError(f"невідома умова {condition!r}")

    lapa = OpenAICompatLlm(model=os.environ["LAPA_MODEL"], base_url=url, api_key=key,
                           structured_mode="json_schema")
    mamay = OpenAICompatLlm(model=os.environ["MAMAY_MODEL"], base_url=url, api_key=key,
                            structured_mode="json_schema")
    variant = resolve(spec.prompt_id)
    system = variant.render_system()
    answer_instruction = resolve(spec.answer_prompt_id).render_system()
    budget = build_budget(spec)

    # Село народжується РАЗ на сід і зберігається; ухвали й стосунки кріпляться до конкретних людей.
    village: list = []
    if spec.mode == "viche":
        store = SqliteVillage(db)
        village = store.load(VILLAGE_SEED)
        if not village:
            # Кажемо вголос: це один виклик Mamay і він триває, а порт доти закритий. Мовчазна
            # пауза на хвилину читається як «не запустилось».
            print(f"  породжую село (сід {VILLAGE_SEED})… перший старт довший", flush=True)
            village = forge_village(
                build_router(spec, lapa=lapa, mamay=mamay), build_effort(spec),
                seed=VILLAGE_SEED, roles=[p.role for p in PERSONAS],
                lenses={p.role: p.lens for p in PERSONAS}, size=spec.max_width + 2,
                system=resolve("viche/forge").render_system())
            store.save(VILLAGE_SEED, village)

    bus = EventBus()
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    queue = SqliteQueue(db)
    governor = Governor(max_tokens=max_tokens, max_usd=max_usd, max_items=max_items,
                        kill_file=kill_file)

    def make_agent(trace, run_id):
        kw = dict(system=system, tail=variant.tail or None, prompt_id=variant.id,
                  prompt_sha=variant.sha256, answer_instruction=answer_instruction)
        if spec.mode == "viche":
            agent = build_viche(spec, lapa=lapa, mamay=mamay, trace=trace, run_id=run_id,
                                system=system, prompt_id=variant.id, prompt_sha=variant.sha256,
                                line_system=variant.render_system(),
                                score_system=resolve("viche/score").render_system(),
                                summary_system=resolve("viche/summary").render_system(),
                                doubt_system=resolve("viche/doubt").render_system(),
                                chronicle_system=resolve("viche/chronicle").render_system(),
                                village=village)
            agent.budget_template = budget
            return agent
        if spec.graph:
            agent = build_graph(spec, lapa=lapa, mamay=mamay, trace=trace, run_id=run_id, **kw)
        else:
            agent = build_orchestrator(spec, lapa=lapa, mamay=mamay, **kw)
            agent.trace = trace
            agent.run_id = run_id
        agent.budget_template = budget
        return agent

    runner = LiveRunner(bus, queue, make_agent, governor=governor, scene=SCENE,
                        paused=paused,
                        cast=public_cast(village) if spec.mode == "viche" else None,
                        decisions=SqliteDecisions(db) if spec.mode == "viche" else None)
    return spec, bus, queue, runner


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        spec, bus, queue, runner = build_live(
            condition=args.condition, max_tokens=args.max_tokens, max_usd=args.max_usd,
            max_items=args.max_items, db=args.db, kill_file=args.kill_file,
            paused=not args.resume)
    except (RuntimeError, KeyError) as exc:
        print(exc)
        return 2

    # Ключ із міткою часу: черга ідемпотентна за ключем (інваріант порту), тому фіксований
    # `seed-0` при перезапуску тихо НЕ запускався б — виглядало б як «сервер нічого не робить».
    stamp = int(time.time())
    for i, topic in enumerate(args.seed_topic):
        queue.put(f"seed-{stamp}-{i}", {"task": topic, "source": "cli"})

    static = Path(args.static) if args.static else None
    if static is not None and not (static / "index.html").is_file():
        # Кажемо ВГОЛОС, а не роздаємо 404: інакше «ядро працює, а сторінка порожня» знову виглядає
        # як поламка ядра.
        print(f"  ⚠ збірки немає: {static}/index.html — зроби `pnpm build`, або став --static ''")
        static = None
    httpd = serve(bus, runner, port=args.port, static=static)
    runner.start()
    print(f"ПЛОЩА онлайн: http://127.0.0.1:{args.port}"
          + ("" if static else "  (лише API — статику не роздаємо)"))
    print(f"  умова     {args.condition} (spec={spec.sha256})")
    print(f"  стелі     токени {args.max_tokens} · ${args.max_usd} · айтемів {args.max_items}")
    print(f"  kill-file {args.kill_file}  (створи файл — цикл зупиниться)")
    state_note = "ПРАЦЮЄ" if args.resume else "ПАУЗА (POST /command kind=resume)"
    print(f"  стан      {state_note}")
    print(f"  черга     {json.dumps(queue.stats(), ensure_ascii=False)}")

    stop = {"now": False}

    def on_signal(_sig, _frm):
        stop["now"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        while not stop["now"]:
            time.sleep(0.3)
    finally:
        runner.stop()
        httpd.shutdown()
        print("\nзупинено:", json.dumps(runner.health(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
