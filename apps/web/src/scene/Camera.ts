/**
 * Пан (перетягування) + зум (колесо, щипок) через CSS-трансформ полотна.
 * Рендер лишається нативним (world scale=1), тож Pixi-фільтр води не зʼїжджає.
 * Полотно внутрішньо worldW×worldH; CSS масштабує/зсуває його в межах рами.
 */
export class Camera {
  private scale = 1;
  private minScale = 1;
  private maxScale = 1;
  private x = 0;
  private y = 0;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private moved = false;
  /**
   * Пальці на полотні. Зум був ТІЛЬКИ на колесі, тобто на телефоні його не існувало взагалі:
   * село відкривалось у кадрі-cover, де видно пʼяту частину його ширини, і наблизити хату чи
   * відступити на все село не було чим.
   */
  private pts = new Map<number, { x: number; y: number }>();
  private pinch = 0; // відстань між пальцями на попередньому кадрі

  constructor(
    private view: HTMLCanvasElement,
    private frame: HTMLElement,
    private worldW: number,
    private worldH: number,
    private focus?: { x: number; y: number },
  ) {
    view.style.transformOrigin = "0 0";
    view.style.touchAction = "none";
    view.style.cursor = "grab";
    view.style.willChange = "transform";
    view.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointermove", this.onMove);
    window.addEventListener("pointerup", this.onUp);
    window.addEventListener("pointercancel", this.onUp);
    view.addEventListener("wheel", this.onWheel, { passive: false });
    // fit після лейауту рами (інакше розміри ще не фінальні → нема пан-простору)
    requestAnimationFrame(() => this.fit());
  }

  private get vw(): number {
    return this.frame.clientWidth;
  }
  private get vh(): number {
    return this.frame.clientHeight;
  }

  fit(): void {
    this.recompute();
    // Стартове наближення. +25% ховає краї полотна на широкому екрані, але на телефоні кадр і
    // без того вузький: cover лишає видимими 24.3% ширини села, а з наближенням — 19.4%.
    // Тобто чверть екрана з'їдалась там, де її й так бракує.
    const boost = this.frame.clientWidth < 700 ? 1 : 1.25;
    this.scale = Math.min(this.maxScale, this.minScale * boost);
    const fx = this.focus ? this.focus.x : this.worldW / 2;
    const fy = this.focus ? this.focus.y : this.worldH / 2;
    this.x = this.vw / 2 - fx * this.scale;
    this.y = this.vh / 2 - fy * this.scale;
    this.apply();
  }

  resize(): void {
    this.recompute();
    this.apply();
  }

  consumeDrag(): boolean {
    const m = this.moved;
    this.moved = false;
    return m;
  }

  private recompute(): void {
    // cover раму; стеля зуму підвищена, щоб «пірнати» в локацію ближче за 1:1
    this.minScale = Math.max(this.vw / this.worldW, this.vh / this.worldH);
    this.maxScale = Math.max(this.minScale, 1.7);
  }

  /** Плавний tween камери (x,y,scale) з ease-out; використовується зануренням/виходом. */
  private raf = 0;
  private tween(toX: number, toY: number, toS: number, ms: number, onDone?: () => void): void {
    cancelAnimationFrame(this.raf);
    const fx = this.x, fy = this.y, fs = this.scale, t0 = performance.now();
    const ease = (t: number): number => 1 - Math.pow(1 - t, 3);
    const step = (now: number): void => {
      const t = Math.min(1, (now - t0) / ms);
      const e = ease(t);
      this.scale = fs + (toS - fs) * e;
      this.x = fx + (toX - fx) * e;
      this.y = fy + (toY - fy) * e;
      this.apply();
      if (t < 1) this.raf = requestAnimationFrame(step);
      else if (onDone) onDone();
    };
    this.raf = requestAnimationFrame(step);
  }

  /** Занурення до точки світу (native-координати POI): наближення + центрування. */
  diveTo(wx: number, wy: number): void {
    this.recompute();
    const s = Math.min(this.maxScale, Math.max(this.minScale * 2.6, this.minScale));
    this.tween(this.vw / 2 - wx * s, this.vh / 2 - wy * s, s, 620);
  }

  /**
   * Мʼяко тримати точку в центрі кадру, НЕ змінюючи зуму.
   *
   * Це не занурення: коли село сходиться на віче, камера просто йде за людьми, як за процесією.
   * Викликається щокадру, тож рухаємось часткою відстані, а не твіном.
   */
  follow(wx: number, wy: number, k = 0.06): void {
    const tx = this.vw / 2 - wx * this.scale;
    const ty = this.vh / 2 - wy * this.scale;
    this.x += (tx - this.x) * k;
    this.y += (ty - this.y) * k;
    this.apply();
  }

  /** Вихід назад у повну діораму. */
  back(): void {
    this.recompute();
    const s = Math.min(this.maxScale, this.minScale * 1.25);
    const fx = this.focus ? this.focus.x : this.worldW / 2;
    const fy = this.focus ? this.focus.y : this.worldH / 2;
    this.tween(this.vw / 2 - fx * s, this.vh / 2 - fy * s, s, 520);
  }

  /** Видима зараз область СВІТУ (native), розширена на margin — для кулінгу рослин вітром. */
  visibleWorldRect(margin = 0): { x0: number; y0: number; x1: number; y1: number } {
    return {
      x0: -this.x / this.scale - margin,
      y0: -this.y / this.scale - margin,
      x1: (this.vw - this.x) / this.scale + margin,
      y1: (this.vh - this.y) / this.scale + margin,
    };
  }

  /** Client-координати → світ (native). Для ручного хіт-тесту (клік по селянину). */
  clientToWorld(cx: number, cy: number): { x: number; y: number } {
    const r = this.frame.getBoundingClientRect();
    return { x: (cx - r.left - this.x) / this.scale, y: (cy - r.top - this.y) / this.scale };
  }

  /** Поточний зум — сцені й директору треба знати, наскільки все наближено. */
  get zoom(): number {
    return this.scale;
  }

  private apply(): void {
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, this.scale));
    const sw = this.worldW * this.scale;
    const sh = this.worldH * this.scale;
    this.x = sw <= this.vw ? (this.vw - sw) / 2 : Math.min(0, Math.max(this.vw - sw, this.x));
    this.y = sh <= this.vh ? (this.vh - sh) / 2 : Math.min(0, Math.max(this.vh - sh, this.y));
    this.view.style.transform = `translate(${this.x.toFixed(1)}px, ${this.y.toFixed(1)}px) scale(${this.scale.toFixed(4)})`;
  }

  private onDown = (e: PointerEvent): void => {
    // Перший дотик жесту («primary») означає, що інших пальців на екрані нема. Прибираємо тут
    // залишки: якщо бодай один `pointerup` загубився (палець зіскочив на панель браузера,
    // система перехопила жест), у списку назавжди лишалось два пальці — і сцена більше не
    // тягалась, бо кожен рух вважався щипком.
    if (e.isPrimary) this.pts.clear();
    this.pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pts.size === 2) {
      // другий палець лягає на екран — далі це щипок, а не тяга
      this.pinch = this.spread();
      this.dragging = false;
      this.moved = true; // щипок не має відкривати локацію під пальцем
      return;
    }
    this.dragging = true;
    this.moved = false;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.view.style.cursor = "grabbing";
  };

  private onMove = (e: PointerEvent): void => {
    if (this.pts.has(e.pointerId)) this.pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pts.size >= 2) {
      const d = this.spread();
      if (this.pinch > 0 && d > 0) {
        const c = this.center();
        this.zoomAt(c.x, c.y, d / this.pinch);
      }
      this.pinch = d;
      return;
    }
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) this.moved = true;
    this.x += dx;
    this.y += dy;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.apply();
  };

  private onUp = (e: PointerEvent): void => {
    this.pts.delete(e.pointerId);
    if (this.pts.size < 2) this.pinch = 0;
    if (!this.dragging) return;
    this.dragging = false;
    this.view.style.cursor = "grab";
  };

  /** Відстань між першими двома пальцями. */
  private spread(): number {
    const [a, b] = [...this.pts.values()];
    return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
  }

  /** Середина між пальцями у координатах рами — саме її щипок тримає на місці. */
  private center(): { x: number; y: number } {
    const [a, b] = [...this.pts.values()];
    const r = this.frame.getBoundingClientRect();
    return { x: (a.x + b.x) / 2 - r.left, y: (a.y + b.y) / 2 - r.top };
  }

  /** Змінити зум, лишивши точку (cx,cy) рами на тому самому місці світу. */
  private zoomAt(cx: number, cy: number, k: number): void {
    const wx = (cx - this.x) / this.scale;
    const wy = (cy - this.y) / this.scale;
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, this.scale * k));
    this.x = cx - wx * this.scale;
    this.y = cy - wy * this.scale;
    this.apply();
  }

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const r = this.frame.getBoundingClientRect();
    this.zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.12 : 1 / 1.12);
  };
}
