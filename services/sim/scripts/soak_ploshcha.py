import argparse
import collections
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve_ploshcha import ROOT, build_live  # noqa: E402

TOPICS = [
    "Поясни кожне: «мешти», «бараболя».",
    "Поясни кожне: «солонина», «крижі».",
    "Поясни кожне: «нецьки», «повітка».",
    "Поясни кожне: «ковбаня», «шпихлір».",
    "Поясни кожне: «драбина», «ослін».",
    "Поясни кожне: «глечик», «макітра».",
    "Поясни кожне: «рушник», «веретено».",
    "Поясни кожне: «оборіг», «клуня».",
]


def rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if raw > 10 ** 7 else raw / 1024


def parse_args(argv):
    p = argparse.ArgumentParser(description="Я6-B: соак живого циклу на N темах")
    p.add_argument("--condition", default="ploshcha")
    p.add_argument("--topics", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p.add_argument("--max-usd", type=float, default=0.25)
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--width", type=int, default=0,
                   help="замір A: одна тема з k елементами замість соаку")
    return p.parse_args(argv)


WORDS = ["мешти", "бараболя", "солонина", "крижі", "нецьки", "повітка", "ковбаня", "шпихлір",
         "глечик", "макітра", "оборіг", "клуня"]


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    stamp = int(time.time())
    db = str(ROOT / "docs" / "research" / "eval-runs" / f"soak-{stamp}.db")

    if args.width:
        items = ", ".join(f"«{w}»" for w in WORDS[:args.width])
        topics = [f"Поясни кожне: {items}."]
        label = f"ШИРИНА k={args.width}"
    else:
        topics = TOPICS[:args.topics]
        label = f"СОАК {len(topics)} тем"

    spec, bus, queue, runner = build_live(
        condition=args.condition, max_tokens=args.max_tokens, max_usd=args.max_usd,
        max_items=len(topics), db=db, paused=True)
    for i, topic in enumerate(topics):
        queue.put(f"{stamp}-{i}", {"task": topic})

    print(f"{label} · умова {args.condition} (spec={spec.sha256}) · rss {rss_mb():.0f} МБ")
    runner.resume()
    runner.start()

    per_run, last = [], {"tokens": 0, "t": time.time(), "runs": 0, "cursor": 0}
    deadline = time.time() + args.timeout_s
    while time.time() < deadline:
        time.sleep(0.5)
        h = runner.health()
        done = h["runsDone"]
        if done > last["runs"]:
            events, cursor = bus.since(last["cursor"])
            kinds = collections.Counter(e["type"] for e in events)
            per_run.append({
                "n": done, "tokens": h["spend"]["tokens"] - last["tokens"],
                "seconds": round(time.time() - last["t"], 1),
                "events": len(events), "voices": kinds.get("utterance.spoken", 0),
                "tools": kinds.get("tool.called", 0), "rss_mb": round(rss_mb(), 1),
                "outcome": next((e["payload"]["outcome"] for e in reversed(events)
                                 if e["type"] == "task.outcome"), "?"),
            })
            r = per_run[-1]
            print(f"  тема {done}: {r['tokens']:>6} ток · {r['seconds']:>5}s · "
                  f"{r['events']:>3} подій · {r['voices']} голосів · {r['tools']} інстр. · "
                  f"{r['outcome']:8s} · rss {r['rss_mb']} МБ")
            last = {"tokens": h["spend"]["tokens"], "t": time.time(), "runs": done,
                    "cursor": cursor}
        if h["stoppedReason"] or (done >= len(topics) and not h["queue"].get("pending")):
            break
    runner.stop()
    runner.join()

    h = runner.health()
    toks = [r["tokens"] for r in per_run]
    print(f"\nзупинився: {h['stoppedReason'] or 'вичерпав чергу'}")
    print(f"тем виконано: {h['runsDone']}/{len(topics)} · усього токенів {h['spend']['tokens']}")
    if toks:
        drift = (max(toks) - min(toks)) / (sum(toks) / len(toks)) if sum(toks) else 0
        print(f"токенів на тему: мін {min(toks)} · сер {sum(toks)//len(toks)} · макс {max(toks)}"
              f" · розкид {drift:.0%}")
    print(f"подій у шині: {h['events']['nextSeq']} · dropped {h['events']['dropped']}")
    print(f"rss: {rss_mb():.1f} МБ")
    if args.width:
        v = sum(r["voices"] for r in per_run)
        print(f"★ k={args.width}: голосів {v} · токенів {sum(toks)} · "
              f"на елемент {sum(toks)//max(1, args.width)}")

    out = ROOT / "docs" / "research" / "eval-runs" / f"soak-{stamp}.json"
    out.write_text(json.dumps({"label": label, "condition": args.condition,
                               "spec": spec.sha256, "topics": topics, "runs": per_run,
                               "health": h}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"звіт: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
