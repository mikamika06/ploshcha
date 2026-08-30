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
    // ★ Один кут замість двох. Доти «про це село» й «щось не так?» стояли в різних кутах, різними
    // правилами: ліва рахувала відступ від ширини вікна, права мала жорсткі 20 і 28 пікселів, шрифти
    // різнились на пів пікселя, шари — на девʼять порядків. На широкому екрані вони розʼїжджались,
    // на вузькому сходились, і жодна не знала про безпечні зони телефона.
    const corner = document.createElement("div");
    corner.className = "corner";
    const btn = document.createElement("button");
    btn.className = "corner-dots";
    btn.type = "button";
    btn.setAttribute("aria-label", "Ще");
    btn.textContent = "•••";
    const menu = document.createElement("div");
    menu.className = "corner-menu";
    // Посилання переїжджає СЮДИ з розмітки сторінки, а не твориться тут: у статичному HTML воно
    // потрібне краулерам, які JavaScript не виконують (заміряно: жоден великий AI-краулер не
    // рендерить сторінку), тож зникнути звідти воно не може.
    const nav = document.querySelector(".stovp a");
    if (nav) menu.appendChild(nav);
    const gripeItem = document.createElement("button");
    gripeItem.type = "button";
    gripeItem.className = "corner-item";
    gripeItem.textContent = "щось не так?";
    menu.appendChild(gripeItem);
    corner.append(menu, btn);
    host.appendChild(corner);

    const closeMenu = (): void => corner.classList.remove("on");
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      corner.classList.toggle("on");
      if (!corner.classList.contains("on")) this.note.classList.remove("on");
    });
    corner.addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("click", () => { closeMenu(); this.note.classList.remove("on"); });
    window.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      // Escape закриває спершу меню — і лише якщо його немає, летить далі до сцени.
      if (corner.classList.contains("on") || this.note.classList.contains("on")) {
        e.stopPropagation();
        closeMenu();
        this.note.classList.remove("on");
      }
    }, true);

    this.note = document.createElement("div");
    this.note.className = "gripe-note";
    this.note.innerHTML = `
      <textarea placeholder="що зламалось або що заважає…" maxlength="2000"></textarea>
      <div class="gripe-row"><span class="gripe-say"></span><button type="button">надіслати</button></div>`;
    host.appendChild(this.note);
    this.area = this.note.querySelector("textarea") as HTMLTextAreaElement;
    this.say = this.note.querySelector(".gripe-say") as HTMLElement;

    gripeItem.addEventListener("click", (e) => {
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
