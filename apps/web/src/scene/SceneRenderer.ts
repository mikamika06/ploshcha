import { Application, Container, Filter, Sprite, Texture } from "pixi.js";
import type { SceneSpec } from "@ploshcha/contract-ts";
import { assetUrl, loadGraded, readPixels } from "../util/gfx";
import { makeWaterFilter } from "./Water";
import { seedVegetation, type VegTextures } from "./Vegetation";
import { Wind } from "./Wind";
import { Intro } from "./Intro";
import { Weather } from "./Weather";

/** Pixi-сцена: земля+вода, рослинність, вітер, інтро, погода. Глибина = zIndex(y). */
export class SceneRenderer {
  readonly app: Application;
  readonly world: Container;
  weather?: Weather;

  private t = 0;
  private waterFilter?: Filter;
  private wind?: Wind;
  private intro?: Intro;

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
    if (this.scene.masks.flow) {
      this.waterFilter = makeWaterFilter(this.scene.masks.flow, this.scene.size.w / this.scene.size.h);
      g.filters = [this.waterFilter];
    }
    this.world.addChild(g);
  }

  async buildVegetation(tex: VegTextures, shadowTex: Texture): Promise<void> {
    const m = this.scene.masks;
    if (!m.zone) return;
    const MW = m.space.w;
    const MH = m.space.h;
    const SCL = this.scene.size.w / MW;
    const zone = await readPixels(assetUrl(m.zone), MW, MH);
    const keep = m.keepout ? await readPixels(assetUrl(m.keepout), MW, MH) : null;
    const windList = seedVegetation(this.world, zone, keep, tex, shadowTex, {
      MW, MH, SCL, DEN: 9, TREE: 1.3, CXf: 782 / 1408, CYf: 377 / 768,
    });
    this.wind = new Wind(windList, this.scene.size.w);
  }

  initWeather(): void {
    this.weather = new Weather(this.app.stage, this.scene.size.w, this.scene.size.h);
  }

  playIntro(): void {
    this.intro = new Intro(this.app.stage, this.scene.size.w, this.scene.size.h);
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
