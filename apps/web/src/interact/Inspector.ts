import type { VillagerState } from "../store/SimStore";

const ROLE_UA: Record<string, string> = {
  koval: "коваль",
  mirosh: "мірошник",
  pip: "піп",
  sheptu: "шептуха-сваха",
  starosta: "староста",
  shynkar: "шинкарка",
  chumak: "чумак",
  diak: "дяк",
  did: "дід-пасічник",
  mati: "молодиця",
  parubok: "парубок",
  divchyna: "дівчина",
};

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}
function list(items: string[], cls: string): string {
  if (!items.length) return `<div class="ins-empty">—</div>`;
  return items.map((x) => `<div class="${cls}">${esc(x)}</div>`).join("");
}

/**
 * Інспектор агента — «зазирнути в голову»: памʼять(думки)/план/сказане/активність.
 * Це analytics-проєкція (прихована когніція). Клік по селянину на діорамі → відкриває.
 */
export class Inspector {
  private root: HTMLElement;
  private body: HTMLElement;

  constructor(private onClose: () => void) {
    this.root = document.createElement("div");
    this.root.className = "inspect";
    this.root.innerHTML = `
      <div class="ins-panel">
        <button class="ins-close" type="button" aria-label="Закрити">✕</button>
        <div class="ins-body"></div>
      </div>`;
    this.body = this.root.querySelector(".ins-body") as HTMLElement;
    (this.root.querySelector(".ins-close") as HTMLElement).textContent = "×";
    this.root.querySelector(".ins-close")!.addEventListener("click", () => this.onClose());
    document.getElementById("stage")!.appendChild(this.root);
  }

  open(v: VillagerState): void {
    const role = ROLE_UA[v.role] ?? v.role;
    // Тут ЛИШЕ сказане вголос. План дня, думки й поточне заняття були службовим шаром: вони
    // пояснювали ядро, а не людину, і читалися як налагоджувальна панель поверх села.
    this.body.innerHTML = `
      <div class="ins-head">
        <div class="ins-name">${esc(v.name)}</div>
        <div class="ins-role">${esc(role)}</div>
      </div>
      <div class="ins-bio">${esc(v.bio)}</div>
      <div class="ins-sec">
        <div class="ins-lbl">Сказав</div>
        ${list(v.said, "ins-said")}
      </div>`;
    this.root.classList.add("on");
  }

  close(): void {
    this.root.classList.remove("on");
  }
}
