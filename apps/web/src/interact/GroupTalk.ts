import { assetUrl } from "../util/gfx";
import type { TalkPart, DiscussionLine } from "./discussion";

type Side = "left" | "right";

/** Хто на якому боці стоїть ЗАРАЗ. Один слот на бік: сперечаються двоє, решта чекає черги. */
type Standing = { id: string; el: HTMLElement };

/**
 * Віче як СУПЕРЕЧКА, а не ряд облич: хто сказав тезу — з одного боку, хто заперечує — з
 * протилежного, наступний знову перекидається. Текст лежить у центральній колоні й фігури його не
 * перекривають — раніше ряд учасників унизу затуляв плашку саме там, де читають.
 *
 * Слот один на бік: у кадрі рівно двоє співрозмовників, решта чекає своєї черги. Той, хто щойно
 * говорив, лишається притемнений — видно, кому саме відповідають.
 */
export class GroupTalk {
  private root: HTMLElement;
  private sides: Record<Side, HTMLElement>;
  private plate: HTMLElement;
  private name: HTMLElement;
  private theme: HTMLElement;
  private count: HTMLElement;
  private lines: DiscussionLine[] = [];
  private i = 0;
  private pEls = new Map<string, HTMLElement>();
  private parts = new Map<string, TalkPart>();
  private live = false;
  private status = "";
  private input: HTMLInputElement;
  private form: HTMLElement;
  private standing: Partial<Record<Side, Standing>> = {};
  private sideOf = new Map<string, Side>();
  private lastSide: Side = "right";

  constructor(
    private onClose: () => void,
    private onSay: (text: string) => void = () => {},
  ) {
    this.root = document.createElement("div");
    this.root.className = "gtalk";
    this.root.innerHTML = `
      <div class="gtalk-scrim"></div>
      <div class="gtalk-theme"></div>
      <div class="gtalk-side left"></div>
      <div class="gtalk-side right"></div>
      <div class="gtalk-box">
        <div class="tag gtalk-name"></div>
        <div class="gtalk-plate"></div>
      </div>
      <button class="tag gtalk-back" type="button">← до села</button>
      <form class="gtalk-say">
        <input class="gtalk-input" type="text" maxlength="300" autocomplete="off"
               placeholder="сказати своє селу вголос…">
        <button class="tag gtalk-send" type="button">Сказати</button>
      </form>
      <div class="gtalk-count"></div>
      <div class="gtalk-hint">клік — далі</div>`;
    this.sides = {
      left: this.root.querySelector(".gtalk-side.left") as HTMLElement,
      right: this.root.querySelector(".gtalk-side.right") as HTMLElement,
    };
    this.plate = this.root.querySelector(".gtalk-plate") as HTMLElement;
    this.name = this.root.querySelector(".gtalk-name") as HTMLElement;
    this.theme = this.root.querySelector(".gtalk-theme") as HTMLElement;
    this.count = this.root.querySelector(".gtalk-count") as HTMLElement;
    this.root.querySelector(".gtalk-back")!.addEventListener("click", (e) => {
      e.stopPropagation();
      this.onClose();
    });
    this.input = this.root.querySelector(".gtalk-input") as HTMLInputElement;
    this.form = this.root.querySelector(".gtalk-say") as HTMLElement;
    // Поле не має ловити клік-«далі»: інакше кожна спроба щось написати гортала б розмову.
    this.form.addEventListener("click", (e) => e.stopPropagation());
    this.root.querySelector(".gtalk-send")!.addEventListener("click", (e) => {
      e.stopPropagation();
      this.send();
    });
    this.input.addEventListener("keydown", (e) => {
      // Як на Дошці й у кімнаті: Escape із набраним словом відпускає поле, порожнє — виходить.
      if (e.key === "Escape") {
        if (!this.input.value.trim()) return;
        e.stopPropagation();
        this.input.blur();
        return;
      }
      e.stopPropagation();
      if (e.key === "Enter") {
        e.preventDefault();
        this.send();
      }
    });
    this.root.addEventListener("click", () => this.advance());
    document.getElementById("stage")!.appendChild(this.root);
  }

  open(topic: string, parts: TalkPart[], lines: DiscussionLine[]): void {
    this.theme.textContent = `Тема: ${topic}`;
    this.lines = lines;
    this.i = 0;
    this.live = false;
    this.status = "";
    this.parts = new Map(parts.map((p) => [p.id, p]));
    this.sides.left.innerHTML = "";
    this.sides.right.innerHTML = "";
    this.pEls.clear();
    this.standing = {};
    this.sideOf.clear();
    this.lastSide = "right";
    this.root.classList.add("on");
    this.render();
  }

  /**
   * Живий режим: розмови ЩЕ НЕМА. Ядро думає десятки секунд, тому вікно відкривається порожнім, а
   * репліки доливаються через `push`. Раніше тут викликався генератор-резерв — і на живому ядрі
   * користувач бачив заготовані репліки, тобто фікцію замість прогону.
   */
  openLive(topic: string, status: string): void {
    this.open(topic, [], []);
    this.live = true;
    this.setStatus(status);
  }

  get isLive(): boolean {
    return this.live && this.root.classList.contains("on");
  }

  setStatus(text: string): void {
    this.status = text;
    if (this.i >= this.lines.length) this.render();
  }

  /** Доливає справжню репліку. Якщо глядач стоїть на кінці — показуємо одразу. */
  push(part: TalkPart, line: DiscussionLine): void {
    const atTail = this.i >= this.lines.length;
    this.parts.set(part.id, part);
    this.lines.push(line);
    if (atTail) {
      this.i = this.lines.length - 1;
      this.render();
    }
  }

  /**
   * Новий мовець виходить на бік, ПРОТИЛЕЖНИЙ попередньому: теза ліворуч — заперечення праворуч,
   * далі знову ліворуч. Той самий чоловік поспіль лишається на своєму боці, інакше він би стрибав
   * туди-сюди без причини.
   */
  private place(p: TalkPart): void {
    const known = this.sideOf.get(p.id);
    const side: Side = this.standing[this.lastSide]?.id === p.id
      ? this.lastSide
      : known !== undefined && this.standing[known]?.id === p.id
        ? known
        : this.lastSide === "left" ? "right" : "left";

    const current = this.standing[side];
    if (current?.id !== p.id) {
      if (current) {
        // Підсвітку знімаємо ДО видалення з мапи: інакше `render` більше не бачить цей вузол і
        // той, хто пішов, лишається «активним» — спрайт світиться під чужим імʼям.
        current.el.classList.remove("active", "here");
        current.el.classList.add("gone");
        const stale = current.el;
        setTimeout(() => stale.remove(), 400);
        this.pEls.delete(current.id);
      }
      const el = document.createElement("div");
      el.className = "gtalk-p";
      el.style.backgroundImage = `url(${assetUrl(`/assets/roles/${p.role}/0.webp`)})`;
      this.sides[side].appendChild(el);
      // окремий кадр — інакше браузер не побачить переходу з початкового стану в `here`
      requestAnimationFrame(() => el.classList.add("here"));
      this.standing[side] = { id: p.id, el };
      this.pEls.set(p.id, el);
    }
    this.sideOf.set(p.id, side);
    this.lastSide = side;
  }

  private advance(): void {
    // У живому режимі кінець списку ще не кінець розмови: ядро може дати наступну репліку.
    if (this.i + 1 >= this.lines.length) {
      if (this.live) return;
      this.onClose();
      return;
    }
    this.i++;
    this.render();
  }

  private render(): void {
    // Скільки вже сказано: у живому режимі кінець невідомий, тож показуємо лічильник, а не «з N».
    this.count.textContent = this.lines.length
      ? `${this.i + 1} / ${this.lines.length}${this.live ? " · гомонять" : ""}`
      : "";
    const ln = this.lines[this.i];
    if (!ln) {
      for (const el of this.pEls.values()) el.classList.remove("active");
      this.name.textContent = "";
      this.plate.textContent = this.status;
      return;
    }
    this.place(this.parts.get(ln.id) ?? { id: ln.id, name: ln.name, role: ln.role });
    for (const [id, el] of this.pEls) el.classList.toggle("active", id === ln.id);
    this.name.textContent = ln.name;
    this.plate.textContent = ln.text;
  }

  private send(): void {
    const text = this.input.value.trim();
    if (!text) return;
    this.input.value = "";
    this.onSay(text);
  }

  /** Прогін завершився: далі реплік не буде, тож клік знову означає «вийти». */
  finish(status: string): void {
    this.live = false;
    if (this.i >= this.lines.length) this.setStatus(status);
  }

  close(): void {
    this.live = false;
    this.root.classList.remove("on");
  }
}
