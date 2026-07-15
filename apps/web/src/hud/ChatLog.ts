/** Бічна хроніка: репліки селян, події, підсумкова хроніка дня. */
export class ChatLog {
  private el: HTMLElement;
  private dayEl: HTMLElement | null;

  constructor() {
    const el = document.getElementById("chat");
    if (!el) throw new Error("chat element #chat not found");
    this.el = el;
    this.dayEl = document.getElementById("daylabel");
  }

  line(name: string, text: string): void {
    this.add(`<div class="bd"><span class="nm">${escapeHtml(name)}</span>${escapeHtml(text)}</div>`, "");
  }

  sys(text: string): void {
    this.add(`<div class="bd">${escapeHtml(text)}</div>`, "sys");
  }

  chronicle(title: string): void {
    this.add(`<div class="bd">📜 ${escapeHtml(title)}</div>`, "chron");
  }

  setDay(label: string): void {
    if (this.dayEl) this.dayEl.textContent = label;
  }

  private add(html: string, cls: string): void {
    const d = document.createElement("div");
    d.className = "msg" + (cls ? " " + cls : "");
    d.innerHTML = html;
    this.el.appendChild(d);
    while (this.el.children.length > 60) this.el.removeChild(this.el.firstChild!);
    requestAnimationFrame(() => d.classList.add("show"));
    this.el.scrollTop = this.el.scrollHeight;
  }
}

function escapeHtml(s: string): string {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
