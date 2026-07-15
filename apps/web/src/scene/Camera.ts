import type { Application, Container } from "pixi.js";

/** Пан (перетягування) + зум (колесо) світу, з cover-обмеженням (без порожніх країв). */
export class Camera {
  private scale = 1;
  private minScale = 1;
  private maxScale = 3;
  private x = 0;
  private y = 0;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private moved = false;
  private readonly view: HTMLCanvasElement;

  constructor(
    private app: Application,
    private world: Container,
    private worldW: number,
    private worldH: number,
  ) {
    this.view = app.view as unknown as HTMLCanvasElement;
    this.view.style.cursor = "grab";
    this.view.style.touchAction = "none";
    this.view.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointermove", this.onMove);
    window.addEventListener("pointerup", this.onUp);
    this.view.addEventListener("wheel", this.onWheel, { passive: false });
    this.fit();
  }

  private get vw(): number {
    return this.app.screen.width;
  }
  private get vh(): number {
    return this.app.screen.height;
  }

  /** Стартова рамка: трохи наближено, щоб було куди листати. */
  fit(): void {
    this.recomputeBounds();
    this.scale = this.minScale * 1.4;
    this.x = (this.vw - this.worldW * this.scale) / 2;
    this.y = (this.vh - this.worldH * this.scale) / 2;
    this.apply();
  }

  resize(): void {
    this.recomputeBounds();
    this.apply();
  }

  /** true, якщо останній жест був перетягуванням (щоб клік не зарахувався). */
  consumeDrag(): boolean {
    const m = this.moved;
    this.moved = false;
    return m;
  }

  private recomputeBounds(): void {
    this.minScale = Math.max(this.vw / this.worldW, this.vh / this.worldH);
    this.maxScale = this.minScale * 3.2;
  }

  private apply(): void {
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, this.scale));
    const sw = this.worldW * this.scale;
    const sh = this.worldH * this.scale;
    this.x = sw <= this.vw ? (this.vw - sw) / 2 : Math.min(0, Math.max(this.vw - sw, this.x));
    this.y = sh <= this.vh ? (this.vh - sh) / 2 : Math.min(0, Math.max(this.vh - sh, this.y));
    this.world.scale.set(this.scale);
    this.world.position.set(this.x, this.y);
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
    const rect = this.view.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const wx = (cx - this.x) / this.scale;
    const wy = (cy - this.y) / this.scale;
    this.scale *= e.deltaY < 0 ? 1.12 : 1 / 1.12;
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, this.scale));
    this.x = cx - wx * this.scale;
    this.y = cy - wy * this.scale;
    this.apply();
  };
}
