import { Graphics } from "pixi.js";
import type { Application } from "pixi.js";

/**
 * Погода настрою: холодний серпанок над сценою (екранний оверлей) тим густіший,
 * чим нижчий настрій села. Валентність +1 → ясно; −1 → тьмяно й туманно.
 */
/** Стеля серпанку: вище — це вже не настрій, а вимкнене світло. */
const MAX_HAZE = 0.1;

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

  /**
   * Серпанок настрою — ЛЕГКИЙ дотик, не завіса.
   *
   * Було до 0.34 непрозорості холодного синього поверх усієї сцени, і настрій із часом сповзав у
   * мінус: село поступово сіріло, а оновлення сторінки «лагодило» його — бо скидало настрій. Село
   * мусить лишатись яскравим; погода може ледь торкнутись, але не гасити картину.
   */
  setMood(valence: number): void {
    this.target = Math.max(0, Math.min(MAX_HAZE, -valence * MAX_HAZE));
  }

  update(dt: number): void {
    this.cur += (this.target - this.cur) * Math.min(1, dt * 1.5);
    this.overlay.alpha = this.cur;
    this.overlay.renderable = this.cur > 0.003; // ясна погода → не блендимо повноекранний прозорий квад
  }
}
