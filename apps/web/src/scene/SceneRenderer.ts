import { Application, Container, Filter, Sprite, Texture } from "pixi.js";
import type { SceneSpec } from "@ploshcha/contract-ts";
import { assetUrl, loadGraded, readPixels } from "../util/gfx";
import { makeWaterFilter } from "./Water";
import { seedVegetation, type VegItem, type VegTextures } from "./Vegetation";
import { Wind } from "./Wind";
import { Intro } from "./Intro";
import { Weather } from "./Weather";
import { Camera } from "./Camera";

export interface ObjectSpec {
  file: string;
  x: number;
  y: number;
  w: number;
  h: number;
  baseY: number;
}

/**
 * Pixi-сцена нативного розміру (world scale=1 → фільтр води не зʼїжджає).
 * Пан/зум — CSS-трансформ полотна через Camera. Глибина = zIndex(y).
 */
export class SceneRenderer {
  readonly app: Application;
  readonly world: Container;
  weather?: Weather;
  camera?: Camera;

  private t = 0;
  private waterFilter?: Filter;
  private wind?: Wind;
  private intro?: Intro;
  private vegItems: VegItem[] = [];

  constructor(public scene: SceneSpec) {
    this.app = new Application({
      width: scene.size.w,
      height: scene.size.h,
      antialias: true,
      backgroundColor: 0x6f7a45,
      powerPreference: "high-performance",
      autoDensity: true,
      resolution: Math.min(2, window.devicePixelRatio || 1),
    });
    this.world = new Container();
    this.world.sortableChildren = true;
    this.app.stage.addChild(this.world);
  }

  mount(el: HTMLElement): void {
    const canvas = this.app.view as unknown as HTMLCanvasElement;
    el.insertBefore(canvas, el.firstChild);
    const heart =
      this.scene.pois.find((p) => p.kind === "square") ??
      this.scene.pois.find((p) => p.kind === "well");
    const focus = heart ? { x: heart.x, y: heart.y } : undefined;
    this.camera = new Camera(canvas, el, this.scene.size.w, this.scene.size.h, focus);
    window.addEventListener("resize", () => this.camera?.resize());
  }

  async loadGround(): Promise<void> {
    const tex = await loadGraded(assetUrl(this.scene.background));
    const g = new Sprite(tex);
    g.width = this.scene.size.w;
    g.height = this.scene.size.h;
    g.zIndex = -1e9;
    if (this.scene.masks.flow) {
      this.waterFilter = makeWaterFilter(this.scene.masks.flow, this.scene.size.w / this.scene.size.h);
      g.filters = [this.waterFilter];
    }
    this.world.addChild(g);
  }

  /** Будівлі/споруди «розпечено» з фону в окремі спрайти з глибиною (zIndex=низ) — щоб селяни заходили ЗА них. */
  async loadObjects(objects: ObjectSpec[]): Promise<void> {
    await Promise.all(
      objects.map(async (o) => {
        const tex = await loadGraded(assetUrl(`assets/objects/${o.file}`)).catch(() => null);
        if (!tex) return;
        // висока споруда (хата/церква/зруб) заслоняє; пласка (грядки/корита) — на землі.
        // Власна запечена тінь уже всередині вирізаного спрайта.
        const tall = o.h >= o.w * 0.6;
        const sp = new Sprite(tex);
        sp.x = o.x;
        sp.y = o.y;
        sp.zIndex = tall ? o.baseY : -5e8;
        this.world.addChild(sp);
      }),
    );
  }

  async buildVegetation(tex: VegTextures, shadowTex: Texture): Promise<void> {
    const m = this.scene.masks;
    if (!m.zone) return;
    const MW = m.space.w;
    const MH = m.space.h;
    const SCL = this.scene.size.w / MW;
    const zone = await readPixels(assetUrl(m.zone), MW, MH);
    const keep = m.keepout ? await readPixels(assetUrl(m.keepout), MW, MH) : null;
    this.vegItems = seedVegetation(this.world, zone, keep, tex, shadowTex, {
      MW, MH, SCL, DEN: 9, TREE: 1.3, CXf: 782 / 1408, CYf: 377 / 768,
    });
    this.wind = new Wind(this.vegItems, this.scene.size.w);
  }

  initWeather(): void {
    this.weather = new Weather(this.app);
  }

  playIntro(): void {
    this.intro = new Intro(this.app.stage, this.app.screen.width, this.app.screen.height);
  }

  /** Амбієнт кожен кадр: течія води, вітер, інтро, погода. */
  update(dt: number): void {
    this.t += dt;
    if (this.waterFilter) this.waterFilter.uniforms.t = this.t;
    this.wind?.update(this.t);
    this.intro?.update(dt);
    this.weather?.update(dt);
  }

  blast(): void {
    this.wind?.blast(this.t);
  }
}
