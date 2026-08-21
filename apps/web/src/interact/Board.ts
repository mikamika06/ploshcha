import { assetUrl } from "../util/gfx";
export type Heat = "hot" | "warm" | "cold" | "sealed";

/** Де гомоніти. Місце — не декорація: у шинку немає старости, у церкві розмова віч-на-віч. */
const PLACES: { id: string; label: string }[] = [
  { id: "ploshcha", label: "на Площі" },
  { id: "shynok", label: "у шинку" },
  { id: "tserkva", label: "у церкві" },
  { id: "kuznya", label: "у кузні" },
  { id: "mlyn", label: "у млині" },
  { id: "dzvin", label: "б’ючи в дзвін" },
];
export interface Topic {
  id: string;
  text: string;
  heat: Heat;
  author?: string;
}

// Підпис жару («тепла», «палає») з листка прибрано: він займав рядок на маленькій цидулці й нічого
// не додавав — гарячу тему й так видно теплим світінням, а холодна просто не клікається.

// Зелена панель з маски loc_board_mask.png — де чіпляються цидулки (нормовані частки кадру).
const PANEL = { x0: 0.233, y0: 0.207, x1: 0.767, y1: 0.767 };

/**
 * Дошка-вісник — діегетична сцена: дерев'яна дошка (loc_board.jpg), а теми-цидулки
 * пришпилені паперовими листками на дощатій панелі (за маскою). Клік по ГАРЯЧІЙ → розмова.
 */
export class Board {
  private root: HTMLElement;
  private notes: HTMLElement;
  private input: HTMLTextAreaElement;
  private topics: Topic[] = [];
  /** Куди піде наступна тема. Місце їде РАЗОМ із нею — це різні процеси, не інтерʼєри.
   *  Назва `where`, а не `place`: `place()` тут уже зайнятий розкидом цидулок. */
  where = "ploshcha";
  private seq = 0;
  private jit = new Map<string, { cx: number; cy: number; rot: number }>(); // сталий розкид+нахил на цидулку

  constructor(
    private onOpenTalk: (t: Topic) => void,
    private onClose: () => void,
  ) {
    this.root = document.createElement("div");
    this.root.className = "board";
    this.root.innerHTML = `
      <div class="board-stage">
        <img class="board-bg" alt="" src="${assetUrl("/assets/locations/loc_board.jpg")}">
        <div class="board-notes"></div>
      </div>
      <div class="board-title">Дошка-вісник</div>
      <button class="tag board-back" type="button">← до села</button>
      <div class="board-places"></div>
      <form class="board-write">
        <textarea class="board-ta" rows="1" maxlength="500" placeholder="Кинь селу тему для розмови…"></textarea>
        <button class="tag board-pin" type="button">Пришпилити</button>
      </form>`;
    this.notes = this.root.querySelector(".board-notes") as HTMLElement;
    this.input = this.root.querySelector(".board-ta") as HTMLTextAreaElement;
    const places = this.root.querySelector(".board-places") as HTMLElement;
    for (const p of PLACES) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = `tag board-place${p.id === this.where ? " on" : ""}`;
      el.textContent = p.label;
      el.addEventListener("click", () => {
        this.where = p.id;
        for (const other of places.children) other.classList.remove("on");
        el.classList.add("on");
      });
      places.appendChild(el);
    }

    // Місце контейнера цидулок рахує `fit()`: воно різне на широкому екрані (над зеленою панеллю)
    // і на телефоні (стовпчик на всю дошку), а поворот телефона перекидає одне в інше.
    window.addEventListener("resize", () => {
      this.fit();
      this.tuck();
    });
    // стейдж бере аспект самої дошки
    const bg = this.root.querySelector(".board-bg") as HTMLImageElement;
    const stage = this.root.querySelector(".board-stage") as HTMLElement;
    // Аспект віддаємо стилям числом (`--ar`), бо ним CSS рахує ще й ГРАНИЧНУ ширину: дошка має
    // влізти і вшир, і вгору. Ставити тут `aspectRatio` було замало — низ дошки зрізало.
    bg.onload = (): void => {
      if (bg.naturalWidth) stage.style.setProperty("--ar", String(bg.naturalWidth / bg.naturalHeight));
    };

    this.root.querySelector(".board-back")!.addEventListener("click", () => this.onClose());
    this.root.querySelector(".board-pin")!.addEventListener("click", () => this.pin());
    this.input.addEventListener("keydown", (e) => {
      // Escape у полі спершу лише ВІДПУСКАЄ поле, а не викидає з Дошки: інакше набрана тема
      // зникала разом із нею з першого натискання (заміряно: `.board` втрачав `on`, текст
      // пропадав). Друге Escape — поле вже не у фокусі — виходить до села, як і всюди.
      if (e.key === "Escape") {
        if (!this.input.value.trim()) return;
        e.stopPropagation();
        this.input.blur();
        return;
      }
      e.stopPropagation();
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.pin();
      }
    });
    document.getElementById("stage")!.appendChild(this.root);

    // сідинг кількох тем села
    this.addTopic({ text: "Чи брати чумаків крам у борг", heat: "hot" });
    this.addTopic({ text: "Кажуть, за річкою вовки виють", heat: "warm" });
    this.addTopic({ text: "Стара гребля знову протікає", heat: "cold" });
  }

  private pin(): void {
    const text = this.input.value.trim();
    if (!text) return;
    this.input.value = "";
    // Пришпилити — лише локально. У чергу ядра тема йде на КЛІК по цидулці (`onOpenTalk`), бо
    // інакше засіяні теми не мали б прогону взагалі, а власна мала б їх ДВА: один на пришпилення,
    // другий на клік. Один шлях постановки роботи = одна витрата.
    this.addTopic({ text, heat: "hot", author: "ти" });
  }

  /** Партитура лишається в ядрі: на Дошці вона була стіною службового тексту збоку. */
  pinPlan(_summary: string, _steps: string[]): void {}

  addTopic(t: Omit<Topic, "id">): Topic {
    const topic: Topic = { id: `t${++this.seq}`, ...t };
    this.jit.set(topic.id, this.place());
    this.topics.unshift(topic);
    // Межа: живий бек стрімить теми безперервно → не даємо списку/DOM рости нескінченно. На
    // вузькому екрані листків менше — не тому, що не влазять, а тому, що двадцять чотири
    // папірці на долоні читаються як мотлох, а не як стіна оголошень.
    const CAP = window.innerWidth < 840 ? 6 : 24;
    while (this.topics.length > CAP) {
      const gone = this.topics.pop();
      if (gone) this.jit.delete(gone.id);
    }
    this.render();
    return topic;
  }

  /** Рандомна позиція листка з уникненням сильного перекриття (best-of-N по дистанції до інших). */
  private place(): { cx: number; cy: number; rot: number } {
    const pts = [...this.jit.values()];
    let best = { cx: 0.5, cy: 0.5 };
    let bestD = -1;
    for (let k = 0; k < 45; k++) {
      const cx = 0.09 + Math.random() * 0.82;
      const cy = 0.08 + Math.random() * 0.62; // уся панель, але не під ряд бирок унизу
      let d = 9;
      for (const p of pts) d = Math.min(d, Math.hypot(cx - p.cx, (cy - p.cy) * 1.35));
      if (d > bestD) {
        bestD = d;
        best = { cx, cy };
      }
      if (d > 0.2) break; // достатньо далеко від інших — беремо
    }
    return { cx: best.cx, cy: best.cy, rot: (Math.random() * 2 - 1) * 7 };
  }

  private render(): void {
    this.notes.innerHTML = "";
    const n = this.topics.length;
    this.topics.forEach((t, i) => {
      const j = this.jit.get(t.id) ?? { cx: 0.5, cy: 0.5, rot: 0 };
      const el = document.createElement("button");
      el.type = "button";
      el.className = `note note-${t.heat}${t.heat === "hot" ? " note-live" : ""}`;
      el.style.left = `${j.cx * 100}%`;
      el.style.top = `${j.cy * 100}%`;
      el.style.setProperty("--rot", `${j.rot}deg`);
      // Порядок листків: новіші зверху, але ГАРЯЧІ — над усіма. Гаряча цидулка це єдиний спосіб
      // пустити тему в ядро, тож накрита гаряча тема = тема, яку не запустити. На повній дошці
      // (24 листки у низькому вікні) розсування саме по собі цього не гарантує — поле замале.
      el.style.setProperty("--z", String((t.heat === "hot" ? 100 : 0) + n - i));
      el.title = t.text; // повний текст при наведенні (на листку — лише початок)
      el.innerHTML = `
        <span class="note-text">${escapeHtml(t.text)}</span>
        ${t.author ? `<span class="note-foot"><span class="note-by">— ${escapeHtml(t.author)}</span></span>` : ""}`;
      // Миша: наведення розгортає, клік починає розмову. Палець: наведення НЕ ІСНУЄ, тож перший
      // тап розгортає (прочитати), і лише другий кличе село гомоніти — інакше на телефоні
      // прочитати тему було неможливо, а промах одразу витрачав прогін ядра.
      el.addEventListener("click", () => {
        if (this.touch() && !el.classList.contains("open")) {
          for (const o of Array.from(this.notes.querySelectorAll(".note.open"))) o.classList.remove("open");
          el.classList.add("open");
          return;
        }
        if (t.heat === "hot") this.onOpenTalk(t);
      });
      this.notes.appendChild(el);
    });
    // Цидулку кладемо ЦЕНТРОМ, тож у низькому вікні її половина вилазила на бирки. Після
    // вставлення знаємо справжню висоту — і підтягуємо всередину поля.
    requestAnimationFrame(() => {
      this.fit(); // спершу межа поля, тоді підтягування: інакше tuck рахує зі старої висоти
      this.tuck();
    });
  }

  /** Пальцевий пристрій: наведення не існує, тож жести мусять мати тапову пару. */
  private touch(): boolean {
    return !window.matchMedia("(hover: hover)").matches;
  }

  /** Вузьке або низьке вікно: панель дошки завужча за саму цидулку, тож теми йдуть стовпчиком. */
  private narrow(): boolean {
    return window.matchMedia("(max-width: 840px), (max-height: 520px)").matches;
  }

  /**
   * Підтягує листки в поле й розсуває так, щоб СЕРЕДИНА кожного лишалась вільною.
   *
   * Випадковий розкид цього не давав: заміряно 5 накритих центрів із 16 листків і 8 із 24 — до
   * такої цидулки не доходить ані клік, ані наведення. Найстарішу «гарячу» тему браузер просто
   * не міг навести: її повністю накривала пізніша. А гаряча цидулка — єдиний шлях пустити тему в
   * ядро, тож накритий листок означає тему, яку не запустити.
   *
   * Розвести повністю не вийде й не треба: 24 листки більші за панель. Умова слабша й саме тому
   * досяжна — центр листка не має потрапляти під прямокутник сусіда.
   */
  private tuck(): void {
    const box = this.notes.getBoundingClientRect();
    if (!box.height || !box.width) return;
    const els = Array.from(this.notes.children) as HTMLElement[];
    if (!els.length) return;

    const items = els.map((el) => {
      const r = el.getBoundingClientRect();
      return {
        el,
        hot: el.classList.contains("note-hot"),
        w: r.width,
        h: r.height,
        x: (parseFloat(el.style.left) / 100) * box.width,
        y: (parseFloat(el.style.top) / 100) * box.height,
      };
    });
    type Note = (typeof items)[number];
    const hold = (it: Note): void => {
      it.x = Math.max(it.w / 2, Math.min(box.width - it.w / 2, it.x));
      it.y = Math.max(it.h / 2 + 4, Math.min(box.height - it.h / 2 - 4, it.y));
    };
    for (const it of items) hold(it);

    // Проходів більше, ніж здається потрібним: пари розсуваються послідовно, тож пізніша пара
    // зсуває те, що щойно розвела попередня. На 24 листках 14 проходів лишали 2 накритих центри,
    // 48 не лишають жодного.
    const spread = (set: Note[], passes: number): void => {
      for (let pass = 0; pass < passes; pass++) {
        let moved = false;
        for (let i = 0; i < set.length; i++) {
          for (let j = i + 1; j < set.length; j++) {
            const a = set[i];
            const b = set[j];
            const needX = (a.w + b.w) / 4 - Math.abs(b.x - a.x);
            const needY = (a.h + b.h) / 4 - Math.abs(b.y - a.y);
            if (needX <= 0 || needY <= 0) continue; // центр уже поза сусідом — не чіпаємо
            moved = true;
            // Штовхаємо по ТІЙ осі, де до звільнення ближче: так листки лишаються біля своїх
            // випадкових місць, а не збиваються в рядок.
            if (needX <= needY) {
              const s = (b.x - a.x >= 0 ? 1 : -1) * (needX / 2 + 0.5);
              a.x -= s;
              b.x += s;
            } else {
              const s = (b.y - a.y >= 0 ? 1 : -1) * (needY / 2 + 0.5);
              a.y -= s;
              b.y += s;
            }
            hold(a);
            hold(b);
          }
        }
        if (!moved) break;
      }
    };
    spread(items, 48);
    // Повна дошка в низькому вікні фізично не влазить: 24 листки більші за панель, і кілька
    // центрів лишаються накритими. Гарячі мусять лишитись доступними ЗАВЖДИ — вони єдині, що
    // пускають тему в ядро — тож їх розводимо ще раз, уже тільки між собою (їх утричі менше, і
    // місця на них вистачає). Накрити гарячу може лише інша гаряча: решта лежить нижче за неї.
    const hot = items.filter((it) => it.hot);
    spread(hot, 48);
    // Розсування — це збіжність, а не гарантія: пара може впертись у край поля й лишитись
    // накритою (заміряно 1 випадок на 48 гарячих). Тому останній крок жорсткий: кожну ще накриту
    // гарячу садимо у вільну клітинку сітки з кроком у півлистка. Клітинок завжди більше, ніж
    // гарячих тем, тож місце є.
    const covered = (it: Note, others: Note[]): boolean =>
      others.some((o) => o !== it
        && Math.abs(o.x - it.x) < (o.w + it.w) / 4
        && Math.abs(o.y - it.y) < (o.h + it.h) / 4);
    for (const it of hot) {
      if (!covered(it, hot)) continue;
      const cw = it.w / 2 + 1;
      const ch = it.h / 2 + 1;
      let best: { x: number; y: number; d: number } | null = null;
      for (let gy = it.h / 2 + 4; gy <= box.height - it.h / 2 - 4; gy += ch) {
        for (let gx = it.w / 2; gx <= box.width - it.w / 2; gx += cw) {
          let d = Infinity;
          for (const o of hot) {
            if (o === it) continue;
            d = Math.min(d, Math.max(Math.abs(o.x - gx) / cw, Math.abs(o.y - gy) / ch));
          }
          if (!best || d > best.d) best = { x: gx, y: gy, d };
        }
      }
      if (best && best.d >= 1) {
        it.x = best.x;
        it.y = best.y;
      }
    }
    for (const it of items) {
      it.el.style.left = `${(it.x / box.width) * 100}%`;
      it.el.style.top = `${(it.y / box.height) * 100}%`;
    }
  }

  /**
   * Обрізає поле цидулок так, щоб воно кінчалось ВИЩЕ за ряд бирок і поле вводу.
   *
   * Частками панелі це не рахується: дошка масштабується за аспектом, а бирки стоять від низу
   * екрана — у нижчому вікні вони наїжджають одне на одне. Тому міряємо справжні прямокутники.
   */
  private fit(): void {
    const stage = this.root.querySelector(".board-stage") as HTMLElement;
    const places = this.root.querySelector(".board-places") as HTMLElement;
    const write = this.root.querySelector(".board-write") as HTMLElement;
    const rr = this.root.getBoundingClientRect();
    // Ряд бирок стоїть на своїй відстані від низу, а форма під ним на вузькому екрані стає
    // вдвічі вищою (поле теми йде окремим рядком) — і нижній ряд бирок ховався ЗА неї.
    // Тому відлічуємо не від краю, а від справжнього верху форми.
    places.style.bottom = this.narrow() ? `${rr.bottom - write.getBoundingClientRect().top + 10}px` : "";
    const sr = stage.getBoundingClientRect();
    const pr = places.getBoundingClientRect();
    if (!sr.height || !pr.height) return;

    // Листки ЗАВЖДИ в межах намальованої дошки — і на телефоні теж.
    //
    // Двічі пробував інакше, і обидва рази виходило гірше: винесені в стовпчик поверх екрана вони
    // роз'їжджалися повз картину, а розширені «трохи за панель» лягали на дах і на траву. Стіна
    // дощок — це і є межа, за яку шпильку не вбʼєш. На вузькому екрані міняється РОЗМІР листка,
    // а не місце, де він може висіти.
    if (this.notes.parentElement !== stage) stage.appendChild(this.notes);
    const IN = 0.03;
    const x0 = PANEL.x0 + IN;
    const y0 = PANEL.y0 + IN;
    this.notes.style.left = `${x0 * 100}%`;
    this.notes.style.top = `${y0 * 100}%`;
    this.notes.style.width = `${(PANEL.x1 - PANEL.x0 - 2 * IN) * 100}%`;
    const top = y0 * sr.height;
    const limit = pr.top - sr.top - 18 - top;      // 18px просвіту над бирками
    const full = (PANEL.y1 - PANEL.y0 - 2 * IN) * sr.height;
    this.notes.style.height = `${Math.max(80, Math.min(full, limit))}px`;
  }

  open(): void {
    this.root.classList.add("on");
    requestAnimationFrame(() => {
      this.fit();
      this.tuck();
    });
    setTimeout(() => this.input.focus(), 250);
  }
  close(): void {
    this.root.classList.remove("on");
    this.input.blur();
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}
