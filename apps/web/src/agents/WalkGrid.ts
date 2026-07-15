import { assetUrl, readPixels } from "../util/gfx";

type Cell = [number, number];
const NEIGHBORS: Cell[] = [
  [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
];

/** Сітка прохідності з маски walk. Координати клітинок — у просторі маски; центри — у нативному. */
export class WalkGrid {
  GW = 0;
  GH = 0;
  private grid = new Uint8Array(0);

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

  bfs(s: Cell, g: Cell): Cell[] | null {
    const prev = new Int32Array(this.GW * this.GH).fill(-1);
    const start = s[1] * this.GW + s[0];
    prev[start] = start;
    const q: Cell[] = [s];
    let hd = 0;
    while (hd < q.length) {
      const [cx, cy] = q[hd++];
      if (cx === g[0] && cy === g[1]) break;
      for (const [dx, dy] of NEIGHBORS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (this.walkable(nx, ny) && prev[ny * this.GW + nx] === -1) {
          prev[ny * this.GW + nx] = cy * this.GW + cx;
          q.push([nx, ny]);
        }
      }
    }
    const gi = g[1] * this.GW + g[0];
    if (prev[gi] === -1) return null;
    const path: Cell[] = [];
    let ci = gi;
    for (;;) {
      const cx = ci % this.GW;
      const cy = (ci / this.GW) | 0;
      path.unshift([cx, cy]);
      if (cx === s[0] && cy === s[1]) break;
      ci = prev[ci];
    }
    return path;
  }

  cellCenter(gx: number, gy: number): { x: number; y: number } {
    return { x: (gx * this.CELL + 7) * this.SCL, y: (gy * this.CELL + 7) * this.SCL };
  }
}
