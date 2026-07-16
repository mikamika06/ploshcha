import { Container, Sprite, Texture } from "pixi.js";
import { hash, vnoise } from "./noise";

export interface VegItem {
  sprite: Sprite;
  shadow: Sprite | null;
  x: number;
  y: number;
  stiff: number;
  phase: number;
  removed: boolean;
}

export type VegType = "wheat" | "flower" | "reed" | "grass" | "tree" | "bush";
export type VegTextures = Record<VegType, Texture[]>;

export interface VegParams {
  MW: number;
  MH: number;
  SCL: number;
  DEN: number;
  TREE: number;
  CXf: number;
  CYf: number;
}

/**
 * Процедурний розсів рослин із масок zone (r=пшениця, g=квіти, b=очерет) + keepout.
 * Багатство росте від центру площі до країв. Повертає список для колихання вітром.
 */
export function seedVegetation(
  world: Container,
  zone: Uint8ClampedArray,
  keep: Uint8ClampedArray | null,
  tex: VegTextures,
  shadowTex: Texture,
  o: VegParams,
): VegItem[] {
  const { MW, MH, SCL, DEN, TREE, CXf, CYf } = o;
  const items: VegItem[] = [];
  const cx = CXf * MW;
  const cy = CYf * MH;
  const maxR = Math.hypot(Math.max(cx, MW - cx), Math.max(cy, MH - cy));

  const pick = (ty: VegType, sx: number, sy: number): Texture | undefined => {
    const arr = tex[ty];
    if (!arr.length) return undefined;
    return arr[Math.min(arr.length - 1, (hash(sx * 1.1 + 1, sy * 0.7 + 2) * arr.length) | 0)];
  };

  const addSprite = (t: Texture, x: number, y: number, targetH: number, flip: number): Sprite => {
    const sp = new Sprite(t);
    sp.anchor.set(0.5, 1);
    sp.x = x;
    sp.y = y;
    const sc = (targetH * SCL) / t.height;
    sp.scale.set(sc * flip, sc);
    sp.zIndex = y;
    world.addChild(sp);
    return sp;
  };

  const plant = (ty: VegType, mx: number, my: number, baseH: number, stiff: number, sx: number, sy: number): void => {
    const t = pick(ty, sx, sy);
    if (!t) return;
    const th = baseH * (0.8 + hash(sx * 2.3 + 5, sy * 1.9 + 3) * 0.45);
    const flip = hash(sx * 0.5 + 9, sy * 3.1 + 4) < 0.5 ? -1 : 1;
    const sp = addSprite(t, mx * SCL, my * SCL, th, flip);
    let sh: Sprite | null = null;
    if (ty === "tree" || ty === "bush") {
      sh = new Sprite(shadowTex);
      sh.anchor.set(0.5, 0.5);
      sh.x = mx * SCL;
      sh.y = my * SCL;
      sh.width = sp.width * 0.9;
      sh.height = sp.width * 0.34;
      sh.zIndex = my * SCL - 0.5;
      world.addChild(sh);
    }
    items.push({
      sprite: sp,
      shadow: sh,
      x: mx * SCL,
      y: my * SCL,
      stiff: stiff * (0.85 + hash(sx * 0.9 + 12, sy * 2.1 + 7) * 0.4),
      phase: hash(sx * 1.7 + 6, sy * 2.5 + 8) * 6.28,
      removed: false,
    });
  };

  const base = DEN;
  const treK = TREE;
  for (let y = 0; y < MH; y += base) {
    for (let x = 0; x < MW; x += base) {
      const i = (y * MW + x) * 4;
      if (keep && keep[i] > 110) continue;
      const r = zone[i];
      const g = zone[i + 1];
      const b = zone[i + 2];
      let z: VegType | null = null;
      if (r > 110 && r >= g && r >= b) z = "wheat";
      else if (b > 110 && b >= g * 0.7) z = "reed";
      else if (g > 110) z = "flower";
      if (!z) continue;
      const jx = x + (hash(x * 0.3 + 1, y * 0.5 + 2) - 0.5) * base;
      const jy = y + (hash(x * 0.6 + 3, y * 0.24 + 4) - 0.5) * base;
      const rp = hash(x * 1.7 + 3.1, y * 2.3 + 0.7);
      const rich = Math.min(1, Math.hypot(jx - cx, jy - cy) / maxR);
      const grove = vnoise(x * 0.012 + 4, y * 0.012 + 9);
      if (z === "wheat") {
        if (rp > 0.93) continue;
        plant("wheat", jx, jy, 22, 1.0, x, y);
      } else if (z === "reed") {
        if (rp < 0.3) plant("reed", jx, jy, 36, 0.95, x + 0.2, y);
        else if (rp < 0.62) plant("grass", jx, jy, 20, 0.9, x + 0.4, y);
      } else {
        if (rp < 0.05 + 0.14 * rich) plant("flower", jx, jy, 15, 0.85, x + 0.11, y + 0.13);
        else if (hash(x * 0.9 + 11, y * 1.3 + 5) < 0.06) plant("grass", jx, jy, 17, 0.9, x + 0.23, y + 0.29);
        if (rich > 0.3 && hash(x * 2.7 + 9, y * 1.1 + 4) < 0.03 * treK * (0.3 + grove)) plant("bush", jx, jy, 26, 1.8, x + 0.37, y + 0.41);
        if (rich > 0.42 && hash(x * 3.3 + 7, y * 2.9 + 2) < 0.02 * treK * grove * grove) plant("tree", jx, jy, 56, 3.2, x + 0.71, y + 0.83);
      }
    }
  }
  return items;
}
