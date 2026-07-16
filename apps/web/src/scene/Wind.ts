import type { VegItem } from "./Vegetation";
import { vnoise } from "./noise";

const WIND = 0.5;
const FREQ = 0.35;

function gustAt(x: number, y: number, wx: number, wy: number, t: number): number {
  const gx = x * 0.0022 - wx * t * 0.35;
  const gy = y * 0.0022 - wy * t * 0.35;
  const n = vnoise(gx, gy) * 0.62 + vnoise(gx * 2.1 + 5, gy * 2.1 + 3) * 0.38;
  const g = Math.max(0, n - 0.5) / 0.5;
  return g * g;
}

/** Колихання рослин через sprite.skew.x за полем вітру + пориви (blast). */
export class Wind {
  private blastStart = -99;

  constructor(private list: VegItem[], private Wn: number) {}

  blast(t: number): void {
    this.blastStart = t;
  }

  update(t: number): void {
    const wa = 0.9 * Math.sin(t * 0.06) + 0.5 * Math.sin(t * 0.017 + 2);
    const wdx = Math.cos(wa);
    const wdy = Math.sin(wa);
    const dtb = t - this.blastStart;
    const blastAmt = dtb >= 0 && dtb < 1.7 ? Math.sin((Math.PI * dtb) / 1.7) : 0;
    const blastPh = Math.min(Math.max(dtb / 1.7, 0), 1);
    for (const w of this.list) {
      if (w.removed) continue;
      let g = gustAt(w.x, w.y, wdx, wdy, t) * (0.3 + 1.4 * FREQ);
      if (blastAmt > 0) {
        const along = w.x / this.Wn;
        g += blastAmt * Math.exp(-Math.pow((along - blastPh) * 2.6, 2)) * 1.3;
      }
      const rustle = 0.03 * Math.sin(t * 2.6 + w.phase);
      w.sprite.skew.x = -((g * WIND) / w.stiff) * wdx - rustle * (g > 0.02 ? 1 : 0.25);
    }
  }
}
