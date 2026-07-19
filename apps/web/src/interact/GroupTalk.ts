import type { TalkPart, DiscussionLine } from "./discussion";

/**
 * Гурт-розмова: N учасників (пів-фігури з ходячих спрайтів) рядком унизу, активний мовець
 * підсвічений і піднятий; репліка на плашці; клік = далі. Поверх ЖИВОЇ Площі (усі зійшлись).
 */
export class GroupTalk {
  private root: HTMLElement;
  private cast: HTMLElement;
  private plate: HTMLElement;
  private name: HTMLElement;
  private theme: HTMLElement;
  private lines: DiscussionLine[] = [];
  private i = 0;
  private pEls = new Map<string, HTMLElement>();

  constructor(private onClose: () => void) {
    this.root = document.createElement("div");
    this.root.className = "gtalk";
    this.root.innerHTML = `
      <div class="gtalk-scrim"></div>
      <div class="gtalk-theme"></div>
      <div class="gtalk-cast"></div>
      <div class="gtalk-name"></div>
      <div class="gtalk-plate"></div>
      <button class="gtalk-back" type="button">← до села</button>
      <div class="gtalk-hint">клік — далі</div>`;
    this.cast = this.root.querySelector(".gtalk-cast") as HTMLElement;
    this.plate = this.root.querySelector(".gtalk-plate") as HTMLElement;
    this.name = this.root.querySelector(".gtalk-name") as HTMLElement;
    this.theme = this.root.querySelector(".gtalk-theme") as HTMLElement;
    this.root.querySelector(".gtalk-back")!.addEventListener("click", (e) => {
      e.stopPropagation();
      this.onClose();
    });
    this.root.addEventListener("click", () => this.advance());
    document.getElementById("stage")!.appendChild(this.root);
  }

  open(topic: string, parts: TalkPart[], lines: DiscussionLine[]): void {
    this.theme.textContent = `Тема: ${topic}`;
    this.lines = lines;
    this.i = 0;
    this.cast.innerHTML = "";
    this.pEls.clear();
    // скільки учасників — такий розмір (щоб вміщались рядком)
    this.cast.style.setProperty("--n", String(Math.max(1, parts.length)));
    for (const p of parts) {
      const el = document.createElement("div");
      el.className = "gtalk-p";
      el.style.backgroundImage = `url(/assets/roles/${p.role}/0.png)`;
      this.cast.appendChild(el);
      this.pEls.set(p.id, el);
    }
    this.root.classList.add("on");
    this.render();
  }

  private advance(): void {
    this.i++;
    if (this.i >= this.lines.length) this.onClose();
    else this.render();
  }

  private render(): void {
    const ln = this.lines[this.i];
    if (!ln) return;
    for (const [id, el] of this.pEls) el.classList.toggle("active", id === ln.id);
    this.name.textContent = ln.name;
    this.plate.textContent = ln.text;
  }

  close(): void {
    this.root.classList.remove("on");
  }
}
