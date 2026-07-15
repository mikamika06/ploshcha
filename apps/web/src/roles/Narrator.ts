/** Оповідач/Літописець — голос у лівій колонці нижньої стрічки (кросфейд при зміні). */
export class Narrator {
  private el: HTMLElement;
  private swap: ReturnType<typeof setTimeout> | undefined;

  constructor(elId = "narration") {
    const el = document.getElementById(elId);
    if (!el) throw new Error(`narration element #${elId} not found`);
    this.el = el;
  }

  say(text: string): void {
    if (this.swap) clearTimeout(this.swap);
    if (!this.el.classList.contains("show")) {
      this.el.textContent = text;
      requestAnimationFrame(() => this.el.classList.add("show"));
      return;
    }
    this.el.classList.remove("show");
    this.swap = setTimeout(() => {
      this.el.textContent = text;
      this.el.classList.add("show");
    }, 200);
  }
}
