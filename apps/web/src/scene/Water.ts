import { Filter, Texture } from "pixi.js";
import { assetUrl } from "../util/gfx";

// Вода як Pixi-фільтр. flow-мапа: R,G = напрям течії, B = маска води.
// Явно рухомі струмені + мерехтіння + легкий зсув поверхні — щоб ріка «текла».
const waterFrag = `
varying vec2 vTextureCoord; uniform sampler2D uSampler, flowTex; uniform float t, aspect, wa;
float h2(vec2 p){ return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453); }
float n2(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(h2(i), h2(i+vec2(1,0)), f.x), mix(h2(i+vec2(0,1)), h2(i+vec2(1,1)), f.x), f.y); }
void main(){
  vec2 uv = vTextureCoord;
  vec4 base = texture2D(uSampler, uv);
  vec4 fl = texture2D(flowTex, uv);
  // вода лише там, де САМ піксель землі синій (намальована ріка) — flow-мапа лише напрям.
  // так ефект не тече на сушу, навіть якщо мапа трохи не збігається.
  float bluish = smoothstep(-0.01, 0.06, base.b - max(base.r, base.g));
  float wmask = fl.b * bluish;
  if (wmask < 0.04) { gl_FragColor = base; return; }
  vec2 fn = normalize(fl.rg * 2.0 - 1.0 + 1e-5);
  vec2 as = vec2(aspect, 1.0);
  float along  = dot(uv * as, fn);
  float across = dot(uv * as, vec2(-fn.y, fn.x));
  float tt = t * wa;

  // зсув поверхні — вода рухається
  float disp = (n2(vec2(along * 36.0 - tt * 2.4, across * 26.0)) - 0.5) * 0.012;
  vec3 surf = texture2D(uSampler, clamp(uv + fn * disp, 0.001, 0.999)).rgb;
  vec3 water = surf * vec3(0.80, 0.90, 1.03);
  water = mix(water, vec3(0.08, 0.22, 0.31), 0.20);

  // рухомі струмені течії (видимий потік)
  float rip = n2(vec2(along * 22.0 - tt * 2.4, across * 14.0));
  float bands = sin(along * 38.0 - tt * 3.4 + rip * 4.0);
  float streak = smoothstep(0.5, 1.0, bands);
  water += streak * 0.17 * vec3(0.66, 0.83, 0.99);

  // мерехтливі блищики, що пливуть за течією
  float spark = smoothstep(0.76, 0.99, n2(vec2(along * 62.0 - tt * 4.6, across * 34.0 + tt * 0.9)));
  water += spark * 0.34 * vec3(0.88, 0.95, 1.0);

  gl_FragColor = vec4(mix(base.rgb, water, clamp(wmask * 1.35, 0.0, 1.0)), base.a);
}`;

export function makeWaterFilter(flowUrl: string, aspect: number): Filter {
  const flowTex = Texture.from(assetUrl(flowUrl));
  return new Filter(undefined, waterFrag, { flowTex, t: 0, aspect, wa: 1.35 });
}
