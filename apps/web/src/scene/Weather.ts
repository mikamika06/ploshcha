import { Graphics } from "pixi.js";
import type { Application } from "pixi.js";

/**
 * Погода настрою: холодний серпанок над сценою (екранний оверлей) тим густіший,
 * чим нижчий настрій села. Валентність +1 → ясно; −1 → тьмяно й туманно.
 */
export class Weather {
  private overlay = new Graphics();
  private cur = 0;
  private target = 0;
  private readonly color = 0x2a3348;

  constructor(private app: Application) {
    app.stage.addChild(this.overlay);
    this.draw();
  }

  private draw(): void {
    this.overlay.clear();
    this.overlay.beginFill(this.color).drawRect(0, 0, this.app.screen.width, this.app.screen.height).endFill();
    this.overlay.alpha = this.cur;
  }

  resize(): void {
    this.draw();
  }

  setMood(valence: number): void {
    this.target = Math.max(0, Math.min(0.34, 0.05 - valence * 0.34));
  }

  update(dt: number): void {
    this.cur += (this.target - this.cur) * Math.min(1, dt * 1.5);
    this.overlay.alpha = this.cur;
    this.overlay.renderable = this.cur > 0.003; // ясна погода → не блендимо повноекранний прозорий квад
  }
}
