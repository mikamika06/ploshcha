import { Container, Graphics, Sprite, Text, Texture } from "pixi.js";
import type { PlaceRef, POI, VillagerPublic } from "@ploshcha/contract-ts";
import type { WalkGrid } from "./WalkGrid";

type Cell = [number, number];

interface Rec {
  id: string;
  name: string;
  sprite: Sprite;
  shadow: Sprite;
  tex: Texture;
  frames: Texture[] | null;
  sc: number;
  x: number;
  y: number;
  face: number;
  path: Cell[] | null;
  pi: number;
  speed: number;
  bob: number;
  state: "idle" | "walk";
  cell: Cell;
  bubble?: Container;
  bubbleT: number;
  wanderTimer: number;
}

/** Звʼязує селян контракту зі спрайтами: спавн, рух по BFS, бульбашки реплік. */
export class AgentDirector {
  private recs = new Map<string, Rec>();
  private nextTex = 0;

  constructor(
    private world: Container,
    private grid: WalkGrid,
    private pois: Map<string, POI>,
    private charTex: Texture[],
    private roleFrames: Map<string, Texture[]>,
    private shadowTex: Texture,
    private SCL: number,
  ) {}

  spawn(villagers: VillagerPublic[]): void {
    for (const v of villagers) {
      if (this.recs.has(v.id)) continue;
      const cell = this.grid.randCell();
      if (!cell) continue;
      const p = this.grid.cellCenter(cell[0], cell[1]);
      const frames = this.roleFrames.get(v.role) ?? null;
      const tex = frames ? frames[0] : this.charTex[this.nextTex++ % Math.max(1, this.charTex.length)];
      const sprite = new Sprite(tex);
      sprite.anchor.set(0.5, 1);
      const sc = (30 * this.SCL) / (tex.height || 1);
      const shadow = new Sprite(this.shadowTex);
      shadow.anchor.set(0.5, 0.5);
      const dh = tex.height * sc; // висота фігури на екрані
      shadow.width = dh * 0.4;
      shadow.height = dh * 0.13;
      shadow.alpha = 0.85;
      this.world.addChild(shadow);
      this.world.addChild(sprite);
      this.recs.set(v.id, {
        id: v.id, name: v.name, sprite, shadow, tex, frames, sc,
        x: p.x, y: p.y, face: Math.random() < 0.5 ? -1 : 1,
        path: null, pi: 0, speed: 10 + Math.random() * 4, bob: 0, state: "idle",
        cell, bubbleT: 0, wanderTimer: 1 + Math.random() * 3,
      });
    }
  }

  /** Найближчий селянин до світ-точки в межах maxDist (native) — для кліку-інспекту. */
  nearestAt(wx: number, wy: number, maxDist: number): string | null {
    let best: string | null = null;
    let bd = maxDist;
    for (const r of this.recs.values()) {
      const d = Math.hypot(r.x - wx, r.y - wy);
      if (d < bd) {
        bd = d;
        best = r.id;
      }
    }
    return best;
  }

  moveTo(id: string, to: PlaceRef): void {
    const r = this.recs.get(id);
    if (!r) return;
    let nx: number;
    let ny: number;
    if (to.poi && this.pois.has(to.poi)) {
      const poi = this.pois.get(to.poi)!;
      nx = poi.x;
      ny = poi.y;
    } else if (to.x !== undefined && to.y !== undefined) {
      nx = to.x;
      ny = to.y;
    } else {
      return;
    }
    const goal = this.grid.nearWalk(nx, ny);
    if (!goal) return;
    const path = this.grid.bfs(r.cell, goal);
    if (path && path.length > 1) {
      r.path = path;
      r.pi = 1;
      r.state = "walk";
    }
  }

  speak(id: string, text: string): void {
    const r = this.recs.get(id);
    if (!r) return;
    this.clearBubble(r);
    r.bubble = this.makeBubble(text);
    r.bubbleT = 3.6;
    this.world.addChild(r.bubble);
  }

  update(dt: number): void {
    for (const r of this.recs.values()) {
      // амбієнтне блукання: коли стоїть — час від часу йде до близької точки
      if (r.state === "idle") {
        r.wanderTimer -= dt;
        if (r.wanderTimer <= 0) {
          r.wanderTimer = 2 + Math.random() * 4;
          const t = this.grid.randCellNear(r.cell, 14);
          const path = t ? this.grid.bfs(r.cell, t) : null;
          if (path && path.length > 1) {
            r.path = path;
            r.pi = 1;
            r.state = "walk";
          }
        }
      }
      if (r.state === "walk" && r.path) {
        if (r.pi >= r.path.length) {
          r.state = "idle";
          r.path = null;
        } else {
          const c = this.grid.cellCenter(r.path[r.pi][0], r.path[r.pi][1]);
          const dx = c.x - r.x;
          const dy = c.y - r.y;
          const d = Math.hypot(dx, dy);
          const step = r.speed * this.SCL * dt;
          if (d <= step) {
            r.x = c.x;
            r.y = c.y;
            r.cell = r.path[r.pi];
            r.pi++;
          } else {
            r.x += (dx / d) * step;
            r.y += (dy / d) * step;
            if (Math.abs(dx) > 0.5) r.face = dx < 0 ? -1 : 1;
            r.bob += dt * 6.5;
          }
        }
      }
      const walking = r.state === "walk" && r.path !== null;
      if (!walking) r.bob += dt * 2.3; // повільне «дихання» у спокої
      const baseH = r.tex.height * r.sc;
      const s = Math.sin(r.bob);
      // кадрова анімація: спокій = кадр 0, хода чергує кадри 1/2
      if (r.frames) {
        const f = walking ? (s >= 0 ? r.frames[1] : r.frames[2] ?? r.frames[1]) : r.frames[0];
        if (f && r.sprite.texture !== f) r.sprite.texture = f;
      }
      const bounce = walking ? Math.abs(s) * baseH * 0.02 : 0;
      r.sprite.x = r.x;
      r.sprite.y = r.y - bounce;
      r.sprite.rotation = 0;
      r.sprite.scale.set(r.sc * r.face, r.sc);
      r.sprite.zIndex = r.y | 0; // ціле → мікрорух не «бруднить» сортування щокадру
      r.shadow.x = r.x;
      r.shadow.y = r.y;
      r.shadow.alpha = walking ? 0.85 - Math.abs(s) * 0.18 : 0.85;
      r.shadow.zIndex = (r.y | 0) - 1;
      if (r.bubble) {
        r.bubbleT -= dt;
        r.bubble.x = r.x;
        r.bubble.y = r.y - r.tex.height * r.sc - 6 * this.SCL;
        r.bubble.zIndex = r.y + 2;
        if (r.bubbleT <= 0) this.clearBubble(r);
      }
    }
  }

  private clearBubble(r: Rec): void {
    if (r.bubble) {
      this.world.removeChild(r.bubble);
      r.bubble.destroy({ children: true });
      r.bubble = undefined;
    }
  }

  private makeBubble(text: string): Container {
    const c = new Container();
    const label = text.length > 70 ? text.slice(0, 68) + "…" : text;
    const t = new Text(label, {
      fontFamily: "-apple-system, Segoe UI, Roboto, sans-serif",
      fontSize: 17,
      fontWeight: "500",
      fill: 0xf0e8d6,
      wordWrap: true,
      wordWrapWidth: 300,
      align: "left",
      lineHeight: 22,
    });
    const padX = 11;
    const padY = 7;
    const tail = 9;
    const w = t.width + padX * 2;
    const h = t.height + padY * 2;
    const bg = new Graphics();
    bg.beginFill(0x17130d, 0.9);
    bg.lineStyle(1, 0xf2ecdb, 0.16);
    bg.drawRoundedRect(0, 0, w, h, 11);
    bg.endFill();
    bg.lineStyle(0);
    bg.beginFill(0x17130d, 0.9);
    bg.moveTo(w / 2 - 7, h - 1);
    bg.lineTo(w / 2, h + tail);
    bg.lineTo(w / 2 + 7, h - 1);
    bg.endFill();
    t.x = padX;
    t.y = padY;
    c.addChild(bg);
    c.addChild(t);
    c.pivot.set(w / 2, h + tail);
    return c;
  }
}
