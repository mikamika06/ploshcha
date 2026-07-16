/**
 * Пан (перетягування) + зум (колесо) через CSS-трансформ полотна.
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
    this.scale = Math.min(this.maxScale, this.minScale * 1.25);
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
    // cover раму, крізь 1:1 (крипко) як стеля зуму
    this.minScale = Math.max(this.vw / this.worldW, this.vh / this.worldH);
    this.maxScale = Math.max(this.minScale, 1.0);
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
    this.dragging = true;
    this.moved = false;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.view.style.cursor = "grabbing";
  };

  private onMove = (e: PointerEvent): void => {
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

  private onUp = (): void => {
    if (!this.dragging) return;
    this.dragging = false;
    this.view.style.cursor = "grab";
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const r = this.frame.getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    const wx = (cx - this.x) / this.scale;
    const wy = (cy - this.y) / this.scale;
    this.scale *= e.deltaY < 0 ? 1.12 : 1 / 1.12;
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, this.scale));
    this.x = cx - wx * this.scale;
    this.y = cy - wy * this.scale;
    this.apply();
  };
}
