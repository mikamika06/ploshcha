import { Filter, Texture } from "pixi.js";
import { assetUrl } from "../util/gfx";

// Вода застосовується до ВСІЄЇ намальованої річки (за чітко-синім пікселем землі),
// а НЕ за flow-мапою — тож ефект завжди точно на річці, по всій довжині, і не їде
// при resize. Flow-мапа дає лише напрям течії (де є), інакше — горизонтальний.
// Поріг «синього» високий, щоб не чіпати сіро-блакитний камінь церкви/хат.
const waterFrag = `
varying vec2 vTextureCoord; uniform sampler2D uSampler,flowTex; uniform float t,aspect,wa;
float h2(vec2 p){return fract(sin(dot(p,vec2(41.3,289.1)))*43758.5453);}
float n2(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(h2(i),h2(i+vec2(1,0)),f.x),mix(h2(i+vec2(0,1)),h2(i+vec2(1,1)),f.x),f.y);}
void main(){ vec2 uv=vTextureCoord; vec4 base=texture2D(uSampler,uv); vec4 fl=texture2D(flowTex,uv);
  float wmask=smoothstep(0.05,0.13, base.b - max(base.r,base.g));   // вся ріка = синій піксель
  if(wmask<0.02){ gl_FragColor=base; return; }
  vec2 fdir=fl.rg*2.0-1.0;
  vec2 fn = (fl.b>0.3) ? normalize(fdir+1e-5) : vec2(1.0,0.0);       // напрям із мапи або горизонт
  vec2 perp=vec2(-fn.y,fn.x); vec2 as=vec2(aspect,1.0);
  float cross=dot(uv*as,perp);
  float ph2=fract(t*0.05*wa*2.0); float amp2=0.050;
  float p1=dot(uv*as,fn)*34.0 - t*2.6*wa;
  float ripple=(n2(vec2(p1,cross*26.0))*0.62+n2(vec2(p1*0.55+11.,cross*13.))*0.38)*2.-1.;
  vec2 rp=perp*ripple*0.0014;
  vec2 uvA=clamp(uv-fn*(ph2*amp2)+rp,0.001,0.999);
  vec2 uvB=clamp(uv-fn*(fract(ph2+0.5)*amp2)+rp,0.001,0.999);
  vec3 wA=texture2D(uSampler,uvA).rgb, wB=texture2D(uSampler,uvB).rgb;
  vec3 water=mix(wA,wB,abs(ph2-0.5)*2.0);
  water*=vec3(0.82,0.90,0.97); water=mix(water,vec3(0.10,0.20,0.26),0.18);
  float sp=n2(vec2(p1+3.,cross*20.)); float streak=smoothstep(0.70,0.96,sp)*wmask;
  vec3 col=mix(base.rgb, water, clamp(wmask*1.15,0.,1.)); col+=streak*0.4*vec3(0.10,0.14,0.17);
  gl_FragColor=vec4(col, base.a);
}`;

export function makeWaterFilter(flowUrl: string, aspect: number): Filter {
  const flowTex = Texture.from(assetUrl(flowUrl));
  const f = new Filter(undefined, waterFrag, { flowTex, t: 0, aspect, wa: 1.0 });
  f.resolution = 0.5; // ріпл м'який і анімований → пів-роздільність невидима, а філ-рейт у 4× менший
  return f;
}
