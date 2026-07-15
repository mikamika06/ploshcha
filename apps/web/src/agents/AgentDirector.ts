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
    private shadowTex: Texture,
    private SCL: number,
  ) {}

  spawn(villagers: VillagerPublic[]): void {
    for (const v of villagers) {
      if (this.recs.has(v.id)) continue;
      const cell = this.grid.randCell();
      if (!cell) continue;
      const p = this.grid.cellCenter(cell[0], cell[1]);
      const tex = this.charTex[this.nextTex++ % Math.max(1, this.charTex.length)];
      const sprite = new Sprite(tex);
      sprite.anchor.set(0.5, 1);
      const sc = (42 * this.SCL) / (tex.height || 1);
      const shadow = new Sprite(this.shadowTex);
      shadow.anchor.set(0.5, 0.5);
      shadow.width = tex.width * sc * 0.8;
      shadow.height = tex.width * sc * 0.3;
      this.world.addChild(shadow);
      this.world.addChild(sprite);
      this.recs.set(v.id, {
        id: v.id, name: v.name, sprite, shadow, tex, sc,
        x: p.x, y: p.y, face: Math.random() < 0.5 ? -1 : 1,
        path: null, pi: 0, speed: 16 + Math.random() * 8, bob: 0, state: "idle",
        cell, bubbleT: 0,
      });
    }
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
            r.bob += dt * 9;
          }
        }
      }
      const bounce = r.state === "walk" ? Math.abs(Math.sin(r.bob)) * r.tex.height * r.sc * 0.02 : 0;
      r.sprite.x = r.x;
      r.sprite.y = r.y - bounce;
      r.sprite.scale.set(r.sc * r.face, r.sc);
      r.sprite.zIndex = r.y;
      r.shadow.x = r.x;
      r.shadow.y = r.y;
      r.shadow.zIndex = r.y - 0.5;
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
    const label = text.length > 64 ? text.slice(0, 62) + "…" : text;
    const t = new Text(label, {
      fontFamily: "-apple-system, Segoe UI, Roboto, sans-serif",
      fontSize: 26,
      fill: 0x2a1c0e,
      wordWrap: true,
      wordWrapWidth: 460,
      align: "left",
    });
    const padX = 16;
    const padY = 11;
    const w = t.width + padX * 2;
    const h = t.height + padY * 2;
    const bg = new Graphics();
    bg.beginFill(0xfffaf0, 0.96);
    bg.lineStyle(2, 0xc89a2f, 1);
    bg.drawRoundedRect(0, 0, w, h, 14);
    bg.endFill();
    bg.beginFill(0xfffaf0, 0.96);
    bg.lineStyle(0);
    bg.moveTo(22, h - 1);
    bg.lineTo(32, h + 15);
    bg.lineTo(44, h - 1);
    bg.endFill();
    t.x = padX;
    t.y = padY;
    c.addChild(bg);
    c.addChild(t);
    c.pivot.set(w / 2, h + 15);
    return c;
  }
}
