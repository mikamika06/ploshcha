import { Texture } from "pixi.js";

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`image load failed: ${url}`));
    img.src = url;
  });
}

// Легкий грейд для цілісного вигляду — трохи живіший, ніж у референсі.
const GRADE = "saturate(0.96) sepia(0.04) brightness(1.02)";
// Сильніший грейд для персонажів — приглушує, щоб вписати у painterly-фон.
export const GRADE_MUTED = "saturate(0.80) sepia(0.13) brightness(0.94)";
const texCache = new Map<string, Texture>();

export async function loadGraded(url: string, maxH?: number, grade: string = GRADE): Promise<Texture> {
  const key = `${url}|${maxH ?? 0}|${grade}`;
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
  ctx.filter = grade;
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

let softOval: Texture | null = null;

/** Спільна текстура м'якої пласкої тіні-калюжі (радіальний градієнт з розмитим краєм).
 *  Кладеться ПІД основу об'єкта (контактна тінь) — тому об'єкт стоїть на землі, а не «літає». */
export function ovalShadow(): Texture {
  if (softOval) return softOval;
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 128;
  const x = c.getContext("2d")!;
  const g = x.createRadialGradient(64, 64, 3, 64, 64, 62);
  g.addColorStop(0, "rgba(18,13,6,0.5)");
  g.addColorStop(0.55, "rgba(18,13,6,0.28)");
  g.addColorStop(1, "rgba(18,13,6,0)");
  x.fillStyle = g;
  x.beginPath();
  x.arc(64, 64, 62, 0, Math.PI * 2);
  x.fill();
  softOval = Texture.from(c);
  return softOval;
}

// Шляхи зі SceneSpec ("assets/...") → абсолютні від кореня статики ("/assets/...").
/**
 * Шлях до ассета З УРАХУВАННЯМ бази збірки.
 *
 * ★ Жорсткий «/assets/…» ламає будь-яке розгортання в підтеку: на GitHub Pages сайт лежить у
 * `/<репо>/`, і кожен такий шлях віддає 404 — заміряно, село не піднімалось узагалі. `BASE_URL`
 * підставляє Vite: локально це «/», у вітрині — «/ploshcha/».
 */
export function assetUrl(p: string): string {
  const base = import.meta.env.BASE_URL || "/";
  const rel = p.startsWith("/") ? p.slice(1) : p;
  return base.endsWith("/") ? base + rel : `${base}/${rel}`;
}
