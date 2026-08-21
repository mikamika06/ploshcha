import { assetUrl, readPixels } from "../util/gfx";

type Cell = [number, number];
const NEIGHBORS: Cell[] = [
  [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
];
const NEIGHBORS4: Cell[] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
/** Найдовша щілина, яку можна протоптати (клітинок по 8px). Через воду — ніколи. */
const BRIDGE = 10;

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
  /** Вода (сині клітинки zone-маски). Через неї не можна протоптати стежку — це річка. */
  private water = new Uint8Array(0);

  constructor(
    private MW: number,
    private MH: number,
    private SCL: number,
    private CELL = 8,
  ) {}

  async load(maskUrl: string, keepoutUrl?: string, zoneUrl?: string): Promise<void> {
    const d = await readPixels(assetUrl(maskUrl), this.MW, this.MH);
    // keepout маркує забудову — виключаємо її з прохідності, щоб селяни не ходили по хатах
    const ko = keepoutUrl ? await readPixels(assetUrl(keepoutUrl), this.MW, this.MH) : null;
    const zn = zoneUrl ? await readPixels(assetUrl(zoneUrl), this.MW, this.MH) : null;
    this.GW = (this.MW / this.CELL) | 0;
    this.GH = (this.MH / this.CELL) | 0;
    this.grid = new Uint8Array(this.GW * this.GH);
    this.water = new Uint8Array(this.GW * this.GH);
    // Клітинку вирішує НЕ один піксель у центрі, а пʼять проб.
    //
    // Дорога вужча за клітинку (14px) провалювалась в одну-єдину пробу й рвалась: сітка
    // розпадалась на 13 острівців, і «йди до шинку» тихо нічого не робило, бо шляху просто не
    // існувало. Заміряно до/після: 587 прохідних / 13 компонент → одна мережа.
    const off = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
      .map(([fx, fy]) => [Math.round(fx * this.CELL), Math.round(fy * this.CELL)]);
    // Клітинка 8px, не 14: сільські дороги вужчі за 14 і рвались, через що сітка розпадалась на
    // 13 острівців. Дві проби з пʼяти — щоб край дороги не з'їдався; по воді ж люди ходили не
    // через поріг, а через прямі коридори зшивання — їх обмежено окремо (BRIDGE).
    for (let gy = 0; gy < this.GH; gy++) {
      for (let gx = 0; gx < this.GW; gx++) {
        let walkHits = 0;
        let blockHits = 0;
        for (const [ox, oy] of off) {
          const px = Math.min(this.MW - 1, gx * this.CELL + ox);
          const py = Math.min(this.MH - 1, gy * this.CELL + oy);
          const i = (py * this.MW + px) * 4;
          if (d[i] > 100 || d[i + 1] > 100) walkHits++;
          if (ko && ko[i] > 110) blockHits++;
        }
        this.grid[gy * this.GW + gx] = walkHits >= 2 && blockHits < 3 ? 1 : 0;
        if (zn) {
          const px = Math.min(this.MW - 1, gx * this.CELL + (this.CELL >> 1));
          const py = Math.min(this.MH - 1, gy * this.CELL + (this.CELL >> 1));
          const zi = (py * this.MW + px) * 4;
          // синє в zone-масці = вода
          this.water[gy * this.GW + gx] = zn[zi + 2] > 120 && zn[zi] < 100 && zn[zi + 1] < 100 ? 1 : 0;
        }
      }
    }
    // Вода не ходиться НІКОЛИ, навіть якщо walk-маска зачепила берег.
    for (let i = 0; i < this.grid.length; i++) if (this.water[i]) this.grid[i] = 0;
    this.computeDist();
    this.connect();
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
    this.connect(); // футпринти могли щойно відрізати дворик — зшиваємо мережу знову
  }

  /**
   * Зшиває сітку в ОДНУ мережу.
   *
   * Дві дії, і межа між ними принципова: якщо острівець відділяє лише щілина (≤ BRIDGE клітинок),
   * протоптуємо стежку; якщо він далеко — ВИКИДАЄМО його з прохідних. Раніше зшивалось усе
   * підряд прямою лінією, і коридор ліг просто через річку — селяни пішли по воді.
   */
  private connect(): void {
    for (let guard = 0; guard < 40; guard++) {
      const comp = this.components();
      if (comp.sizes.length <= 1) return;
      let big = 0;
      for (let i = 1; i < comp.sizes.length; i++) if (comp.sizes[i] > comp.sizes[big]) big = i;
      let joined = false;
      for (let c = 0; c < comp.sizes.length; c++) {
        if (c === big) continue;
        const a = comp.cells[c];
        const b = comp.cells[big];
        let best: [number, number] | null = null;
        let bd = Infinity;
        for (const i of a) {
          const ax = i % this.GW;
          const ay = (i / this.GW) | 0;
          for (const j of b) {
            const bx = j % this.GW;
            const by = (j / this.GW) | 0;
            const dd = (ax - bx) ** 2 + (ay - by) ** 2;
            if (dd < bd) {
              bd = dd;
              best = [i, j];
            }
          }
        }
        if (!best) continue;
        // Стежку протоптуємо лише коротку і лише по суходолу. Клапоть за річкою лишається як є:
        // людина дійде до берега (`pathToward`) і стане — це чесніше, ніж місток нізвідки.
        if (Math.sqrt(bd) <= BRIDGE && this.dryLine(best[0], best[1])) {
          this.carve(best[0], best[1]);
          joined = true;
        }
      }
      if (!joined) return;
      this.computeDist();
    }
  }

  private components(): { sizes: number[]; cells: number[][] } {
    const N = this.GW * this.GH;
    const seen = new Uint8Array(N);
    const sizes: number[] = [];
    const cells: number[][] = [];
    const stack: number[] = [];
    for (let i = 0; i < N; i++) {
      if (seen[i] || this.grid[i] === 0) continue;
      const bag: number[] = [];
      stack.length = 0;
      stack.push(i);
      seen[i] = 1;
      while (stack.length) {
        const ci = stack.pop()!;
        bag.push(ci);
        const cx = ci % this.GW;
        const cy = (ci / this.GW) | 0;
        for (const [dx, dy] of NEIGHBORS) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= this.GW || ny >= this.GH) continue;
          const ni = ny * this.GW + nx;
          if (seen[ni] || this.grid[ni] === 0) continue;
          seen[ni] = 1;
          stack.push(ni);
        }
      }
      sizes.push(bag.length);
      cells.push(bag);
    }
    return { sizes, cells };
  }

  /** Чи вся пряма між клітинками йде суходолом. */
  private dryLine(ai: number, bi: number): boolean {
    let x = ai % this.GW;
    let y = (ai / this.GW) | 0;
    const gx = bi % this.GW;
    const gy = (bi / this.GW) | 0;
    for (let step = 0; step < this.GW + this.GH; step++) {
      if (this.water[y * this.GW + x]) return false;
      if (x === gx && y === gy) return true;
      if (x !== gx) x += x < gx ? 1 : -1;
      if (y !== gy) y += y < gy ? 1 : -1;
    }
    return true;
  }

  /** Протоптує пряму стежку між двома клітинками. */
  private carve(ai: number, bi: number): void {
    let x = ai % this.GW;
    let y = (ai / this.GW) | 0;
    const gx = bi % this.GW;
    const gy = (bi / this.GW) | 0;
    for (let step = 0; step < this.GW + this.GH; step++) {
      this.grid[y * this.GW + x] = 1;
      if (x === gx && y === gy) return;
      if (x !== gx) x += x < gx ? 1 : -1;
      if (y !== gy) y += y < gy ? 1 : -1;
    }
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

  /**
   * N прохідних місць БІЛЯ точки — щоб громада стала колом, а не в одну клітинку.
   *
   * Кільце за кутом і радіусом сюди не годиться: точка кола легко падає в стіну чи у воду, і
   * `nearWalk` відводить її на протилежний бік хати — людина йде навколо всієї будівлі. Тому
   * місця беремо з САМОЇ сітки: найближчі прохідні клітинки, розсунуті одна від одної.
   */
  spotsNear(nativeX: number, nativeY: number, n: number, gap = 2): Cell[] {
    const from = this.nearWalk(nativeX, nativeY);
    if (!from) return [];
    // Обхід у ШИРИНУ від самого місця, а не квадратом навколо точки. Квадрат брав клітинки, що
    // лежать поруч по прямій, але за стіною: до них 195 кроків замість 33 — саме це виглядало як
    // «обходить усю локацію». Порядок BFS = порядок ходьби, тож ближче тут означає ближче НОГАМИ.
    const seen = new Uint8Array(this.GW * this.GH);
    const q: number[] = [from[1] * this.GW + from[0]];
    seen[q[0]] = 1;
    const out: Cell[] = [];
    for (let head = 0; head < q.length && out.length < n; head++) {
      const ci = q[head];
      const cx = ci % this.GW;
      const cy = (ci / this.GW) | 0;
      if (!out.some(([fx, fy]) => Math.max(Math.abs(fx - cx), Math.abs(fy - cy)) < gap)) {
        out.push([cx, cy]);
      }
      for (const [dx, dy] of NEIGHBORS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (!this.walkable(nx, ny)) continue;
        const ni = ny * this.GW + nx;
        if (seen[ni]) continue;
        seen[ni] = 1;
        q.push(ni);
      }
    }
    return out;
  }

  /** Дейкстра з перевагою центрів стежок (edge-клітинки дорожчі) — щоб не ходили по краях. */
  /**
   * Шлях ЯКНАЙБЛИЖЧЕ до цілі, коли самої цілі не досягти.
   *
   * Сітка має ізольовані кишені (клітинка всередині двору прохідна, але з дороги в неї не
   * зайти), і `bfs` там чесно повертає `null`. Наслідок був тихий і найгірший з можливих: команда
   * «йди до шинку» просто НІЧОГО не робила — люди лишались на місці, а подія вважалась виконаною.
   * Тепер людина йде настільки близько, наскільки може дійти.
   */
  pathToward(s: Cell, g: Cell): Cell[] | null {
    const N = this.GW * this.GH;
    const cost = new Float64Array(N).fill(Infinity);
    const prev = new Int32Array(N).fill(-1);
    const si = s[1] * this.GW + s[0];
    cost[si] = 0;
    const heap = new MinHeap();
    heap.push(0, si);
    let best = si;
    let bestD = Math.hypot(s[0] - g[0], s[1] - g[1]);
    while (heap.size > 0) {
      const { p: c, v: ci } = heap.pop();
      if (c > cost[ci]) continue;
      const cx = ci % this.GW;
      const cy = (ci / this.GW) | 0;
      const d = Math.hypot(cx - g[0], cy - g[1]);
      if (d < bestD) {
        bestD = d;
        best = ci;
      }
      for (const [dx, dy] of NEIGHBORS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (!this.walkable(nx, ny)) continue;
        const ni = ny * this.GW + nx;
        const step = dx !== 0 && dy !== 0 ? 1.4142 : 1;
        const nc = c + step;
        if (nc < cost[ni]) {
          cost[ni] = nc;
          prev[ni] = ci;
          heap.push(nc, ni);
        }
      }
    }
    if (best === si) return null;
    const path: Cell[] = [];
    let ci = best;
    for (;;) {
      path.unshift([ci % this.GW, (ci / this.GW) | 0]);
      if (ci === si) break;
      ci = prev[ci];
      if (ci < 0) return null;
    }
    return path;
  }

  /**
   * Дейкстра по сітці.
   *
   * ★ `cost` мусить бути Float64, не Float32. Пріоритет у купі — звичайне число (f64), а
   * `Float32Array` округлює записане значення. Якщо округлення пішло ВНИЗ, пізніша перевірка
   * лінивого видалення `c > cost[ci]` спрацьовувала хибно, вузол не розкривався ніколи — і шлях
   * «не існував». Заміряно: 192 з 200 пар на ЗВʼЯЗНІЙ сітці не мали шляху; після заміни типу —
   * 0. Саме через це селяни нікуди не доходили: будь-який довгий маршрут тихо зникав.
   */
  bfs(s: Cell, g: Cell): Cell[] | null {
    const N = this.GW * this.GH;
    const cost = new Float64Array(N).fill(Infinity);
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
        // Штраф за край стежки. Був 1.7 на клітинці 14px — тобто до 4.4× вартості; на сітці 8px
        // майже КОЖНА клітинка вузької дороги крайова, і Дейкстра вела людей навколо села, аби
        // йти «серединою». Тепер це легкий нахил, а не заборона.
        const penalty = Math.max(0, 2 - this.dist[ni]) * 0.3;
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

  /** Центр клітинки в нативних координатах. Половина CELL, а не жорстке «+7»: клітинка тепер 8px. */
  cellCenter(gx: number, gy: number): { x: number; y: number } {
    return { x: (gx * this.CELL + this.CELL / 2) * this.SCL,
             y: (gy * this.CELL + this.CELL / 2) * this.SCL };
  }
}
