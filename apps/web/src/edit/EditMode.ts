import { Container, Graphics } from "pixi.js";
import type { WalkGrid } from "../agents/WalkGrid";
import type { Camera } from "../scene/Camera";
import type { VegItem } from "../scene/Vegetation";

const VEG_KEY = "ploshcha.edit.veg.removed";
const WALK_KEY = "ploshcha.edit.walk.blocked";

type Tool = "veg" | "walk";

/**
 * Режим правки: пензлем прибираєш рослини, що налазять, або малюєш «стоп-зони»
 * (щоб селяни не ходили по хатах). Правки зберігаються в localStorage.
 */
export class EditMode {
  private active = false;
  private tool: Tool = "veg";
  private painting = false;
  private readonly brush = 55; // радіус пензля, нативні px
  private removedVeg: Set<string>;
  private blocked: Set<number>;
  private layer = new Graphics();
  private toolbar: HTMLElement;
  private btnVeg!: HTMLButtonElement;
  private btnWalk!: HTMLButtonElement;

  constructor(
    world: Container,
    private grid: WalkGrid,
    private items: VegItem[],
    private camera: Camera,
    private view: HTMLCanvasElement,
  ) {
    this.removedVeg = new Set(loadArr(VEG_KEY));
    this.blocked = new Set(loadArr(WALK_KEY).map(Number));

    // застосувати збережене
    for (const it of items) if (this.removedVeg.has(keyOf(it.x, it.y))) hide(it);
    for (const idx of this.blocked) grid.blockIndex(idx);

    this.layer.zIndex = 1e8;
    this.layer.visible = false;
    world.addChild(this.layer);
    this.drawBlocked();

    this.toolbar = document.createElement("div");
    this.toolbar.className = "edit-toolbar";
    this.toolbar.hidden = true;
    this.toolbar.innerHTML =
      '<span class="et-title">Правка</span>' +
      '<button class="et-tool" data-tool="veg" type="button">прибрати рослини</button>' +
      '<button class="et-tool" data-tool="walk" type="button">закрити прохід</button>' +
      '<button class="et-clear" type="button">скинути все</button>' +
      '<span class="et-hint">клікай або веди по сцені</span>';
    document.body.appendChild(this.toolbar);
    this.btnVeg = this.toolbar.querySelector('[data-tool="veg"]')!;
    this.btnWalk = this.toolbar.querySelector('[data-tool="walk"]')!;
    this.btnVeg.onclick = () => this.setTool("veg");
    this.btnWalk.onclick = () => this.setTool("walk");
    (this.toolbar.querySelector(".et-clear") as HTMLButtonElement).onclick = () => this.clearAll();
    this.setTool("veg");

    view.addEventListener("pointerdown", this.onDown);
    window.addEventListener("pointermove", this.onMove);
    window.addEventListener("pointerup", this.onUp);
  }

  toggle(): boolean {
    this.active = !this.active;
    this.camera.locked = this.active;
    this.toolbar.hidden = !this.active;
    this.layer.visible = this.active;
    this.view.style.cursor = this.active ? "crosshair" : "grab";
    return this.active;
  }

  private setTool(t: Tool): void {
    this.tool = t;
    this.btnVeg.classList.toggle("on", t === "veg");
    this.btnWalk.classList.toggle("on", t === "walk");
  }

  private onDown = (e: PointerEvent): void => {
    if (!this.active) return;
    this.painting = true;
    this.paint(e);
  };
  private onMove = (e: PointerEvent): void => {
    if (this.active && this.painting) this.paint(e);
  };
  private onUp = (): void => {
    if (!this.painting) return;
    this.painting = false;
    this.persist();
  };

  private paint(e: PointerEvent): void {
    const w = this.camera.screenToWorld(e.clientX, e.clientY);
    if (this.tool === "veg") {
      this.removeVegAt(w.x, w.y);
    } else {
      const idxs = this.grid.blockWorld(w.x, w.y, this.brush);
      if (idxs.length) {
        for (const i of idxs) this.blocked.add(i);
        this.drawBlocked();
      }
    }
  }

  private removeVegAt(x: number, y: number): void {
    const r2 = this.brush * this.brush;
    for (const it of this.items) {
      if (it.removed) continue;
      const dx = it.x - x;
      const dy = it.y - y;
      if (dx * dx + dy * dy <= r2) {
        hide(it);
        this.removedVeg.add(keyOf(it.x, it.y));
      }
    }
  }

  private drawBlocked(): void {
    const cw = this.grid.cellWorld;
    this.layer.clear();
    this.layer.beginFill(0xd23b2e, 0.3);
    for (const idx of this.blocked) {
      const gx = idx % this.grid.GW;
      const gy = (idx / this.grid.GW) | 0;
      const c = this.grid.cellCenter(gx, gy);
      this.layer.drawRect(c.x - cw / 2, c.y - cw / 2, cw, cw);
    }
    this.layer.endFill();
  }

  private persist(): void {
    localStorage.setItem(VEG_KEY, JSON.stringify([...this.removedVeg]));
    localStorage.setItem(WALK_KEY, JSON.stringify([...this.blocked]));
  }

  private clearAll(): void {
    localStorage.removeItem(VEG_KEY);
    localStorage.removeItem(WALK_KEY);
    location.reload();
  }
}

function keyOf(x: number, y: number): string {
  return `${Math.round(x)}_${Math.round(y)}`;
}
function hide(it: VegItem): void {
  it.removed = true;
  it.sprite.parent?.removeChild(it.sprite);
  it.shadow?.parent?.removeChild(it.shadow);
}
function loadArr(k: string): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(k) ?? "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}
