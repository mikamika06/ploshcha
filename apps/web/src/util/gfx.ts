import { Texture } from "pixi.js";

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`image load failed: ${url}`));
    img.src = url;
  });
}

// GRD=0.5 (референс): злегка приглушує насиченість для цілісного вигляду.
const GRADE = "saturate(0.83) sepia(0.07) brightness(0.97)";
const texCache = new Map<string, Texture>();

export async function loadGraded(url: string, maxH?: number): Promise<Texture> {
  const key = `${url}|${maxH ?? 0}`;
  const hit = texCache.get(key);
  if (hit) return hit;
  const img = await loadImage(url);
  const nw = img.naturalWidth;
  const nh = img.naturalHeight;
  const sc = maxH ? Math.min(1, maxH / nh) : 1;
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(nw * sc));
  c.height = Math.max(1, Math.round(nh * sc));
  const ctx = c.getContext("2d")!;
  ctx.filter = GRADE;
  ctx.drawImage(img, 0, 0, c.width, c.height);
  const tex = Texture.from(c);
  texCache.set(key, tex);
  return tex;
}

// Піксель-дані маски (для walk-grid): малюємо у canvas і читаємо RGBA.
export async function readPixels(url: string, w: number, h: number): Promise<Uint8ClampedArray> {
  const img = await loadImage(url);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

export function makeShadowTexture(): Texture {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 64;
  const x = c.getContext("2d")!;
  const g = x.createRadialGradient(64, 32, 2, 64, 32, 60);
  g.addColorStop(0, "rgba(20,16,8,0.5)");
  g.addColorStop(1, "rgba(20,16,8,0)");
  x.fillStyle = g;
  x.beginPath();
  x.ellipse(64, 32, 60, 30, 0, 0, Math.PI * 2);
  x.fill();
  return Texture.from(c);
}

// Шляхи зі SceneSpec ("assets/...") → абсолютні від кореня статики ("/assets/...").
export function assetUrl(p: string): string {
  return p.startsWith("/") ? p : "/" + p;
}
