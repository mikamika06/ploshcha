import { LIVE_URL, IS_LIVE } from "../config";
import { sessionId } from "../net/session";

/**
 * Скарга на баг — цидулкою, а не формою в чужому сервісі.
 *
 * Текст іде на власне ядро й лягає рядком у файл поруч із базою села (`skargy.jsonl`). Жодного
 * зовнішнього сервісу: сторонній формі довелось би довіряти дані гостя, а тут вони не покидають
 * той самий сервер, що й уся ПЛОЩА.
 */
export class Gripe {
  private note: HTMLElement;
  private area: HTMLTextAreaElement;
  private say: HTMLElement;

  constructor(private where: () => string) {
    const host = document.getElementById("stage") ?? document.body;
    const btn = document.createElement("button");
    btn.className = "gripe";
    btn.type = "button";
    btn.textContent = "щось не так?";
    host.appendChild(btn);

    this.note = document.createElement("div");
    this.note.className = "gripe-note";
    this.note.innerHTML = `
      <textarea placeholder="що зламалось або що заважає…" maxlength="2000"></textarea>
      <div class="gripe-row"><span class="gripe-say"></span><button type="button">надіслати</button></div>`;
    host.appendChild(this.note);
    this.area = this.note.querySelector("textarea") as HTMLTextAreaElement;
    this.say = this.note.querySelector(".gripe-say") as HTMLElement;

    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.note.classList.toggle("on");
      if (this.note.classList.contains("on")) this.area.focus();
    });
    this.note.addEventListener("click", (e) => e.stopPropagation());
    (this.note.querySelector("button") as HTMLElement).addEventListener("click", () => void this.send());
    this.area.addEventListener("keydown", (e) => {
      e.stopPropagation(); // інакше пробіл і стрілки керують камерою замість тексту
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void this.send();
    });
  }

  private async send(): Promise<void> {
    const text = this.area.value.trim();
    if (!text) return;
    if (!IS_LIVE) {
      this.say.textContent = "тут без ядра — нікуди слати";
      return;
    }
    this.say.textContent = "надсилаю…";
    try {
      const res = await fetch(`${LIVE_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, sid: sessionId(), where: this.where() }),
      });
      if (!res.ok) throw new Error(String(res.status));
      this.area.value = "";
      this.say.textContent = "записано, дякую";
      window.setTimeout(() => this.note.classList.remove("on"), 1200);
    } catch {
      this.say.textContent = "не дійшло — ядро не відповідає";
    }
  }
}
