import { assetUrl, readPixels } from "../util/gfx";

type Cell = [number, number];
const NEIGHBORS: Cell[] = [
  [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
];
const NEIGHBORS4: Cell[] = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/** Мінімальна купа (priority, value) з лінивим видаленням — для Дейкстри. */
class MinHeap {
  private ps: number[] = [];
  private vs: number[] = [];
  get size(): number {
    return this.vs.length;
  }
  push(p: number, v: number): void {
    this.ps.push(p);
    this.vs.push(v);
    let i = this.vs.length - 1;
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (this.ps[par] <= this.ps[i]) break;
      this.swap(i, par);
      i = par;
    }
  }
  pop(): { p: number; v: number } {
    const p0 = this.ps[0];
    const v0 = this.vs[0];
    const last = this.vs.length - 1;
    this.ps[0] = this.ps[last];
    this.vs[0] = this.vs[last];
    this.ps.pop();
    this.vs.pop();
    const n = this.vs.length;
    let i = 0;
    for (;;) {
      let sm = i;
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      if (l < n && this.ps[l] < this.ps[sm]) sm = l;
      if (r < n && this.ps[r] < this.ps[sm]) sm = r;
      if (sm === i) break;
      this.swap(i, sm);
      i = sm;
    }
    return { p: p0, v: v0 };
  }
  private swap(a: number, b: number): void {
    const tp = this.ps[a];
    this.ps[a] = this.ps[b];
    this.ps[b] = tp;
    const tv = this.vs[a];
    this.vs[a] = this.vs[b];
    this.vs[b] = tv;
  }
}

/** Сітка прохідності з маски walk. Координати клітинок — у просторі маски; центри — у нативному. */
export class WalkGrid {
  GW = 0;
  GH = 0;
  private grid = new Uint8Array(0);
  private dist = new Int16Array(0); // відстань клітинки до найближчої стіни (центральність)

  constructor(
    private MW: number,
    private MH: number,
    private SCL: number,
    private CELL = 14,
  ) {}

  async load(maskUrl: string, keepoutUrl?: string): Promise<void> {
    const d = await readPixels(assetUrl(maskUrl), this.MW, this.MH);
    // keepout маркує забудову — виключаємо її з прохідності, щоб селяни не ходили по хатах
    const ko = keepoutUrl ? await readPixels(assetUrl(keepoutUrl), this.MW, this.MH) : null;
    this.GW = (this.MW / this.CELL) | 0;
    this.GH = (this.MH / this.CELL) | 0;
    this.grid = new Uint8Array(this.GW * this.GH);
    for (let gy = 0; gy < this.GH; gy++) {
      for (let gx = 0; gx < this.GW; gx++) {
        const px = (gx * this.CELL + 7) | 0;
        const py = (gy * this.CELL + 7) | 0;
        const i = (py * this.MW + px) * 4;
        const walk = d[i] > 100 || d[i + 1] > 100;
        const blocked = ko ? ko[i] > 110 : false;
        this.grid[gy * this.GW + gx] = walk && !blocked ? 1 : 0;
      }
    }
    this.computeDist();
  }

  /** Multi-source BFS від стін → відстань кожної прохідної клітинки до найближчої стіни. */
  private computeDist(): void {
    const N = this.GW * this.GH;
    this.dist = new Int16Array(N).fill(-1);
    const q: number[] = [];
    for (let i = 0; i < N; i++) {
      if (this.grid[i] === 0) {
        this.dist[i] = 0;
        q.push(i);
      }
    }
    let hd = 0;
    while (hd < q.length) {
      const ci = q[hd++];
      const cx = ci % this.GW;
      const cy = (ci / this.GW) | 0;
      const nd = this.dist[ci] + 1;
      for (const [dx, dy] of NEIGHBORS4) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= this.GW || ny >= this.GH) continue;
        const ni = ny * this.GW + nx;
        if (this.dist[ni] === -1) {
          this.dist[ni] = nd;
          q.push(ni);
        }
      }
    }
    for (let i = 0; i < N; i++) if (this.dist[i] < 0) this.dist[i] = 1;
  }

  /**
   * Виключає з прохідності футпринти високих споруд (bbox з objects.json), щоб селяни
   * не заходили ЗА/ПІД хату (де zIndex хати їх ховає й стирчать самі ноги). Викликати ПІСЛЯ load.
   */
  blockObjects(objs: { x: number; y: number; w: number; h: number; baseY: number }[], margin = 6): void {
    for (const o of objs) {
      const tall = o.h >= o.w * 0.6;
      if (!tall) continue;
      const gx0 = Math.max(0, Math.floor((o.x - margin) / this.SCL / this.CELL));
      const gx1 = Math.min(this.GW - 1, Math.floor((o.x + o.w + margin) / this.SCL / this.CELL));
      const gy0 = Math.max(0, Math.floor(o.y / this.SCL / this.CELL));
      const gy1 = Math.min(this.GH - 1, Math.floor((o.baseY + margin) / this.SCL / this.CELL));
      for (let gy = gy0; gy <= gy1; gy++) {
        for (let gx = gx0; gx <= gx1; gx++) this.grid[gy * this.GW + gx] = 0;
      }
    }
    this.computeDist();
  }

  walkable(gx: number, gy: number): boolean {
    return gx >= 0 && gy >= 0 && gx < this.GW && gy < this.GH && this.grid[gy * this.GW + gx] === 1;
  }

  randCell(): Cell | null {
    for (let k = 0; k < 500; k++) {
      const gx = (Math.random() * this.GW) | 0;
      const gy = (Math.random() * this.GH) | 0;
      if (this.walkable(gx, gy)) return [gx, gy];
    }
    return null;
  }

  /** Випадкова прохідна клітинка поблизу (для амбієнтного блукання). */
  randCellNear(center: Cell, radiusCells: number): Cell | null {
    for (let k = 0; k < 80; k++) {
      const gx = center[0] + Math.round((Math.random() * 2 - 1) * radiusCells);
      const gy = center[1] + Math.round((Math.random() * 2 - 1) * radiusCells);
      if (this.walkable(gx, gy)) return [gx, gy];
    }
    return null;
  }

  /** Нативні координати → найближча прохідна клітинка (спіраль). */
  nearWalk(nativeX: number, nativeY: number): Cell | null {
    const gx0 = ((nativeX / this.SCL) / this.CELL) | 0;
    const gy0 = ((nativeY / this.SCL) / this.CELL) | 0;
    if (this.walkable(gx0, gy0)) return [gx0, gy0];
    for (let r = 1; r < 60; r++) {
      for (let dx = -r; dx <= r; dx++) {
        for (let dy = -r; dy <= r; dy++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          if (this.walkable(gx0 + dx, gy0 + dy)) return [gx0 + dx, gy0 + dy];
        }
      }
    }
    return null;
  }

  /** Дейкстра з перевагою центрів стежок (edge-клітинки дорожчі) — щоб не ходили по краях. */
  bfs(s: Cell, g: Cell): Cell[] | null {
    const N = this.GW * this.GH;
    const cost = new Float32Array(N).fill(Infinity);
    const prev = new Int32Array(N).fill(-1);
    const si = s[1] * this.GW + s[0];
    const gi = g[1] * this.GW + g[0];
    cost[si] = 0;
    const heap = new MinHeap();
    heap.push(0, si);
    while (heap.size > 0) {
      const { p: c, v: ci } = heap.pop();
      if (c > cost[ci]) continue;
      if (ci === gi) break;
      const cx = ci % this.GW;
      const cy = (ci / this.GW) | 0;
      for (const [dx, dy] of NEIGHBORS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (!this.walkable(nx, ny)) continue;
        const ni = ny * this.GW + nx;
        const step = dx !== 0 && dy !== 0 ? 1.4142 : 1;
        const penalty = Math.max(0, 3 - this.dist[ni]) * 1.7; // тулитись до центру
        const nc = c + step * (1 + penalty);
        if (nc < cost[ni]) {
          cost[ni] = nc;
          prev[ni] = ci;
          heap.push(nc, ni);
        }
      }
    }
    if (cost[gi] === Infinity && gi !== si) return null;
    const path: Cell[] = [];
    let ci = gi;
    for (;;) {
      path.unshift([ci % this.GW, (ci / this.GW) | 0]);
      if (ci === si) break;
      ci = prev[ci];
      if (ci < 0) return null;
    }
    return path;
  }

  cellCenter(gx: number, gy: number): { x: number; y: number } {
    return { x: (gx * this.CELL + 7) * this.SCL, y: (gy * this.CELL + 7) * this.SCL };
  }
}
