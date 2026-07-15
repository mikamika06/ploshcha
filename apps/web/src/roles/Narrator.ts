/** Оповідач/Літописець — субтитр-панель знизу. */
export class Narrator {
  private el: HTMLElement;
  private hideTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(elId = "narrator") {
    const el = document.getElementById(elId);
    if (!el) throw new Error(`narrator element #${elId} not found`);
    this.el = el;
  }

  say(text: string, who = "Оповідач", holdMs = 6000): void {
    this.el.innerHTML = `<span class="who">${escapeHtml(who)}:</span>${escapeHtml(text)}`;
    this.el.classList.add("show");
    if (this.hideTimer) clearTimeout(this.hideTimer);
    this.hideTimer = setTimeout(() => this.el.classList.remove("show"), holdMs);
  }
}

function escapeHtml(s: string): string {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
