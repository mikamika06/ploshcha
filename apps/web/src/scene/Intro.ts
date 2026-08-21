import { Container, Sprite, Texture } from "pixi.js";

/** Скільки клубків. На пальцевому пристрої їх менше: сорок напівпрозорих спрайтів на весь екран —
 *  це сорок шарів змішування щокадру, і саме там телефон і затинався на самому вході. */
const PUFFS = matchMedia("(pointer: coarse)").matches ? 14 : 40;

function makeCloudTexture(): Texture {
  const c = document.createElement("canvas");
  // 128 замість 256 на телефоні: хмару однаково розтягують у кілька разів, різниці не видно,
  // а текстура вчетверо легша.
  const side = PUFFS < 20 ? 128 : 256;
  c.width = side;
  c.height = side;
  const x = c.getContext("2d")!;
  const h = side / 2;
  const g = x.createRadialGradient(h, h, side / 32, h, h, h);
  g.addColorStop(0, "rgba(252,252,255,1)");
  g.addColorStop(0.55, "rgba(246,247,251,.92)");
  g.addColorStop(1, "rgba(246,247,251,0)");
  x.fillStyle = g;
  x.beginPath();
  x.arc(h, h, h, 0, Math.PI * 2);
  x.fill();
  return Texture.from(c);
}

interface Puff {
  s: Sprite;
  bx: number;
  by: number;
  sc0: number;
  dx: number;
  dy: number;
}

/**
 * Хмарна завіса. Тримається (ховає село), поки все вантажиться й рендериться; коли готове —
 * `dissipate()` розводить хмари, і село відкривається з-під них уже повністю (з людьми).
 */
export class Intro {
  private layer = new Container();
  private puffs: Puff[] = [];
  private t = 0;
  private parting = false;
  done = false;

  constructor(stage: Container, Wn: number, Hn: number) {
    stage.addChild(this.layer);
    const tex = makeCloudTexture();
    for (let i = 0; i < PUFFS; i++) {
      const s = new Sprite(tex);
      s.anchor.set(0.5);
      const px = Math.random() * Wn;
      const py = Math.random() * Hn;
      s.x = px;
      s.y = py;
      const sc0 = (0.7 + Math.random() * 1.7) * (Wn / 1400);
      s.scale.set(sc0);
      const ang = Math.atan2(py - Hn / 2, px - Wn / 2) + (Math.random() - 0.5) * 0.8;
      const dist = Wn * (0.12 + Math.random() * 0.38);
      this.puffs.push({ s, bx: px, by: py, sc0, dx: Math.cos(ang) * dist, dy: Math.sin(ang) * dist });
      this.layer.addChild(s);
    }
  }

  /** Село готове → почати розводити хмари. */
  dissipate(): void {
    this.parting = true;
  }

  update(dt: number): void {
    if (this.done || !this.parting) return; // ХОЛД: хмари стоять і ховають сцену, поки не готово
    this.t = Math.min(1, this.t + dt / 2.8);
    const e = 1 - Math.pow(1 - this.t, 2.2);
    for (const p of this.puffs) {
      p.s.x = p.bx + p.dx * e;
      p.s.y = p.by + p.dy * e;
      p.s.alpha = Math.max(0, 1 - e * 1.15);
      p.s.scale.set(p.sc0 * (1 + e * 0.6));
    }
    if (this.t >= 1) {
      this.layer.parent?.removeChild(this.layer);
      this.layer.destroy({ children: true });
      this.done = true;
    }
  }
}
