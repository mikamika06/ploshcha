import { Container, Graphics } from "pixi.js";

/**
 * Погода настрою: холодний серпанок над сценою тим густіший, чим нижчий настрій села.
 * Валентність +1 (радість) → ясно; −1 (туга) → тьмяно й туманно.
 */
export class Weather {
  private overlay = new Graphics();
  private cur = 0;
  private target = 0;

  constructor(stage: Container, w: number, h: number) {
    this.overlay.beginFill(0x2a3348).drawRect(0, 0, w, h).endFill();
    this.overlay.alpha = 0;
    stage.addChild(this.overlay);
  }

  setMood(valence: number): void {
    this.target = Math.max(0, Math.min(0.34, 0.05 - valence * 0.34));
  }

  update(dt: number): void {
    this.cur += (this.target - this.cur) * Math.min(1, dt * 1.5);
    this.overlay.alpha = this.cur;
  }
}
