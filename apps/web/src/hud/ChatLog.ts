/** Хроніка: живий фід реплік/подій у правій колонці стрічки + годинник дня. */
export class ChatLog {
  private feed: HTMLElement;
  private clockEl: HTMLElement | null;
  private readonly MAX = 6;

  constructor() {
    const feed = document.getElementById("chron");
    if (!feed) throw new Error("feed element #chron not found");
    this.feed = feed;
    this.clockEl = document.getElementById("clock");
  }

  line(name: string, text: string): void {
    this.add(`<span class="nm">${escapeHtml(name)}</span> ${escapeHtml(text)}`, "");
  }

  sys(text: string): void {
    this.add(escapeHtml(text), "evt");
  }

  chronicle(title: string): void {
    this.add(escapeHtml(title), "chron");
  }

  setClock(label: string): void {
    if (this.clockEl) this.clockEl.textContent = label;
  }

  private add(html: string, cls: string): void {
    const d = document.createElement("div");
    d.className = "ln" + (cls ? " " + cls : "");
    d.innerHTML = html;
    this.feed.appendChild(d);
    while (this.feed.children.length > this.MAX) this.feed.removeChild(this.feed.firstChild!);
    requestAnimationFrame(() => d.classList.add("show"));
  }
}

function escapeHtml(s: string): string {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
