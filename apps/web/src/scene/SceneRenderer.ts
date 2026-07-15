import { Application, Container, Filter, Sprite, Texture } from "pixi.js";
import type { SceneSpec } from "@ploshcha/contract-ts";
import { assetUrl, loadGraded, readPixels } from "../util/gfx";
import { makeWaterFilter } from "./Water";
import { seedVegetation, type VegTextures } from "./Vegetation";
import { Wind } from "./Wind";
import { Intro } from "./Intro";
import { Weather } from "./Weather";
import { Camera } from "./Camera";

/** Pixi-сцена: земля+вода, рослинність, вітер, інтро, погода + камера (пан/зум). */
export class SceneRenderer {
  readonly app: Application;
  readonly world: Container;
  weather?: Weather;
  camera?: Camera;

  private t = 0;
  private waterFilter?: Filter;
  private wind?: Wind;
  private intro?: Intro;
  private frameEl?: HTMLElement;

  constructor(public scene: SceneSpec) {
    this.app = new Application({
      antialias: true,
      backgroundColor: 0x6f7a45,
      powerPreference: "high-performance",
      autoDensity: true,
      resolution: Math.min(2, window.devicePixelRatio || 1),
      width: 800,
      height: 600,
    });
    this.world = new Container();
    this.world.sortableChildren = true;
    this.app.stage.addChild(this.world);
  }

  mount(el: HTMLElement): void {
    this.frameEl = el;
    el.insertBefore(this.app.view as unknown as HTMLCanvasElement, el.firstChild);
    this.resizeToFrame();
    this.camera = new Camera(this.app, this.world, this.scene.size.w, this.scene.size.h);
    window.addEventListener("resize", this.onResize);
  }

  private resizeToFrame(): void {
    if (!this.frameEl) return;
    const w = this.frameEl.clientWidth || window.innerWidth;
    const h = this.frameEl.clientHeight || window.innerHeight;
    this.app.renderer.resize(w, h);
  }

  private onResize = (): void => {
    this.resizeToFrame();
    this.camera?.resize();
    this.weather?.resize();
  };

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
