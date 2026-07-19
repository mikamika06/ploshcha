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
    // слабкий пристрій → рендеримо в 1× (у 4 рази менший бекбуфер/філрейт); потужний → до 1.5×
    // (візуально без втрат — полотно все одно CSS-масштабується камерою). antialias зайвий для
    // спрайтового арту (краї — це альфа PNG, а не векторна геометрія).
    const nav = navigator as Navigator & { deviceMemory?: number };
    const weak = (nav.hardwareConcurrency ?? 8) <= 4 || (nav.deviceMemory ?? 8) <= 4;
    this.app = new Application({
      width: scene.size.w,
      height: scene.size.h,
      antialias: false,
      backgroundColor: 0x6f7a45,
      powerPreference: "default",
      autoDensity: true,
      resolution: weak ? 1 : Math.min(1.5, window.devicePixelRatio || 1),
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

  /** Декор-пропси з розкладки (вітряк тощо) — спрайти поверх землі з глибиною (zIndex=y), тож селяни їх обходять. */
  async loadProps(): Promise<void> {
    const props = this.scene.props;
    if (!props?.length) return;
    await Promise.all(
      props.map(async (p) => {
        const tex = await loadGraded(assetUrl(`assets/nb/${p.sprite}.png`)).catch(() => null);
        if (!tex) return;
        const [ax, ay] = p.anchor;
        const sp = new Sprite(tex);
        sp.anchor.set(ax, ay);
        sp.x = p.x;
        sp.y = p.y;
        sp.scale.set(p.scale);
        sp.rotation = (p.rot * Math.PI) / 180;
        sp.zIndex = p.z ?? p.y + tex.height * p.scale * (1 - ay); // глибина за низом спрайта, якщо z не задано
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
    this.wind = new Wind(this.vegItems, this.scene.size.w, this.scene.size.h);
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
    this.wind?.update(this.t, this.camera?.visibleWorldRect(220)); // кулінг рослин поза кадром
    this.intro?.update(dt);
    this.weather?.update(dt);
  }

  blast(): void {
    this.wind?.blast(this.t);
  }
}
