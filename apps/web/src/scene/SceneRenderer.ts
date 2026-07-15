import { Application, Container, Sprite } from "pixi.js";
import type { SceneSpec } from "@ploshcha/contract-ts";
import { assetUrl, loadGraded } from "../util/gfx";

/** Pixi-сцена: запечена земля у світі з painter-order глибиною (zIndex = y). */
export class SceneRenderer {
  readonly app: Application;
  readonly world: Container;

  constructor(public scene: SceneSpec) {
    this.app = new Application({
      width: scene.size.w,
      height: scene.size.h,
      antialias: true,
      backgroundColor: 0x6f7a45,
      powerPreference: "high-performance",
    });
    this.world = new Container();
    this.world.sortableChildren = true;
    this.app.stage.addChild(this.world);
  }

  mount(el: HTMLElement): void {
    el.insertBefore(this.app.view as unknown as HTMLCanvasElement, el.firstChild);
  }

  async loadGround(): Promise<void> {
    const tex = await loadGraded(assetUrl(this.scene.background));
    const g = new Sprite(tex);
    g.width = this.scene.size.w;
    g.height = this.scene.size.h;
    g.zIndex = -1e9;
    this.world.addChild(g);
  }
}
