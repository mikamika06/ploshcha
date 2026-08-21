/**
 * Хмарна завіса переходу.
 *
 * Село не «перемикається» на локацію — воно ЗАТЯГУЄТЬСЯ хмарами, під ними міняється місце, і
 * хмари розходяться вже над іншим двором. Той самий жест, яким відкривається село на старті
 * (`Intro`), тільки в обидві сторони.
 *
 * Чому DOM, а не Pixi: локації — це DOM-оверлеї поверх полотна, і завіса всередині Pixi лишалась
 * би ПІД ними, тобто ховала б лише мапу, а не перехід.
 */

// На пальцевому пристрої клубків менше: кожен — напівпрозорий шар на весь екран, і сорок таких
// шарів телефон домальовує ривками саме тоді, коли має бути найплавніше.
const PUFFS = matchMedia("(pointer: coarse)").matches ? 14 : 40;
const GATHER_MS = 620;
const PART_MS = 900;

/**
 * Та сама хмара, що в `Intro`: радіальний градієнт, намальований у canvas.
 *
 * CSS-градієнт давав інший спад і читався як «дивна пляма» — завіса переходу мусить бути з тієї
 * ж речовини, що й хмари на заставці, інакше це два різні явища в одному селі.
 */
function cloudUrl(): string {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 256;
  const x = c.getContext("2d")!;
  const g = x.createRadialGradient(128, 128, 8, 128, 128, 128);
  g.addColorStop(0, "rgba(252,252,255,1)");
  g.addColorStop(0.55, "rgba(246,247,251,.92)");
  g.addColorStop(1, "rgba(246,247,251,0)");
  x.fillStyle = g;
  x.beginPath();
  x.arc(128, 128, 128, 0, Math.PI * 2);
  x.fill();
  return c.toDataURL();
}

interface Puff {
  el: HTMLElement;
  ox: number; // куди розлітається (частка екрана від центру)
  oy: number;
}

export class Curtain {
  private root: HTMLElement;
  private puffs: Puff[] = [];
  private busy = false;

  constructor() {
    this.root = document.createElement("div");
    this.root.className = "curtain";
    const puffUrl = cloudUrl();
    for (let i = 0; i < PUFFS; i++) {
      const el = document.createElement("div");
      el.className = "curtain-puff";
      el.style.backgroundImage = `url(${puffUrl})`;
      // Розсипані як в `Intro` — випадково по всьому кадру, а не сіткою: сітка читалась рядами.
      const x = Math.random();
      const y = Math.random();
      el.style.left = `${x * 100}%`;
      el.style.top = `${y * 100}%`;
      const size = (0.7 + Math.random() * 1.7) * 34;
      el.style.width = `${size}vw`;
      el.style.height = `${size}vw`;
      const ang = Math.atan2(y - 0.5, x - 0.5) + (Math.random() - 0.5) * 0.7;
      const dist = 0.45 + Math.random() * 0.5;
      this.puffs.push({ el, ox: Math.cos(ang) * dist, oy: Math.sin(ang) * dist });
      this.root.appendChild(el);
    }
    document.getElementById("stage")!.appendChild(this.root);
    this.scatter(0);
  }

  private scatter(k: number): void {
    // k=1 — хмари розлетілись і зникли; k=0 — зійшлись і ховають усе
    for (const p of this.puffs) {
      p.el.style.transform =
        `translate(-50%, -50%) translate(${p.ox * k * 100}vw, ${p.oy * k * 70}vh) scale(${1 + k * 0.5})`;
      p.el.style.opacity = String(Math.max(0, 1 - k * 1.25));
    }
  }

  /**
   * Затягує хмарами, під ними виконує `swap`, тоді розводить.
   *
   * `swap` викликається саме тоді, коли екран закритий: якщо міняти сцену раніше, глядач бачить
   * стрибок, а завіса перетворюється на прикрасу після факту.
   */
  async sweep(swap: () => void): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    this.root.classList.add("on");
    this.puffs.forEach((p) => (p.el.style.transition = `transform ${GATHER_MS}ms ease-in, opacity ${GATHER_MS}ms ease-in`));
    // окремий кадр — інакше браузер не побачить переходу з початкового стану
    await new Promise((r) => requestAnimationFrame(() => r(null)));
    this.scatter(0);
    await wait(GATHER_MS);
    swap();
    await wait(90);
    this.puffs.forEach((p) => (p.el.style.transition = `transform ${PART_MS}ms cubic-bezier(0.22,0.9,0.3,1), opacity ${PART_MS}ms ease-out`));
    this.scatter(1);
    await wait(PART_MS);
    this.root.classList.remove("on");
    this.busy = false;
  }
}

function wait(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}
