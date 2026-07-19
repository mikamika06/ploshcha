import { Circle, Container, type FederatedPointerEvent, Graphics } from "pixi.js";
import type { POI } from "@ploshcha/contract-ts";

/** Радіус хіт-зони порта в native-одиницях (за типом обʼєкта). */
export function radiusFor(kind: string): number {
  switch (kind) {
    case "church":
    case "tavern":
    case "mill":
    case "forge":
      return 90;
    case "square":
      return 120;
    default:
      return 74; // криниця/дошка/дзвіниця/ставок
  }
}

export interface PortEvents {
  onHover: (poi: POI | null, clientX: number, clientY: number) => void;
  onSelect: (poi: POI) => void;
}

/**
 * Робить кожен POI діегетичною «кнопкою»: наведення = шепіт, клік = занурення.
 * Хіт-зони живуть у Pixi-світі (native-координати), тож працюють крізь CSS-камеру.
 */
export class Ports {
  private nodes: Container[] = [];

  constructor(world: Container, pois: POI[], ev: PortEvents) {
    for (const p of pois) {
      const c = new Container();
      c.x = p.x;
      c.y = p.y;
      c.zIndex = 2_000_000_000; // завжди над сценою
      c.eventMode = "static";
      c.cursor = "pointer";
      const r = radiusFor(p.kind);
      c.hitArea = new Circle(0, 0, r);
      // невидиме коло-маркер (для beacon-пульсу в майбутньому; зараз alpha 0)
      const g = new Graphics();
      g.beginFill(0xffffff, 0.001);
      g.drawCircle(0, 0, r);
      g.endFill();
      c.addChild(g);
      c.on("pointerover", (e: FederatedPointerEvent) => ev.onHover(p, e.client.x, e.client.y));
      c.on("pointermove", (e: FederatedPointerEvent) => ev.onHover(p, e.client.x, e.client.y));
      c.on("pointerout", () => ev.onHover(null, 0, 0));
      c.on("pointertap", () => ev.onSelect(p));
      world.addChild(c);
      this.nodes.push(c);
    }
  }

  setEnabled(on: boolean): void {
    for (const n of this.nodes) {
      n.eventMode = on ? "static" : "none";
      n.visible = on;
    }
  }
}
