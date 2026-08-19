export type Heat = "hot" | "warm" | "cold" | "sealed";
export interface Topic {
  id: string;
  text: string;
  heat: Heat;
  author?: string;
}

const HEAT_UA: Record<Heat, string> = { hot: "палає", warm: "тепла", cold: "холоне", sealed: "владнано" };

// Зелена панель з маски loc_board_mask.png — де чіпляються цидулки (нормовані частки кадру).
const PANEL = { x0: 0.233, y0: 0.207, x1: 0.767, y1: 0.767 };

/**
 * Дошка-вісник — діегетична сцена: дерев'яна дошка (loc_board.jpg), а теми-цидулки
 * пришпилені паперовими листками на дощатій панелі (за маскою). Клік по ГАРЯЧІЙ → розмова.
 */
export class Board {
  private root: HTMLElement;
  private notes: HTMLElement;
  private plan: HTMLElement;
  private input: HTMLTextAreaElement;
  private topics: Topic[] = [];
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
        <img class="board-bg" alt="" src="/assets/locations/loc_board.jpg">
        <div class="board-notes"></div>
      </div>
      <div class="board-title">Дошка-вісник</div>
      <div class="board-plan"></div>
      <button class="board-back" type="button">← до села</button>
      <form class="board-write">
        <textarea class="board-ta" rows="1" maxlength="500" placeholder="Нашепни селу тему для розмови…"></textarea>
        <button class="board-pin" type="button">Пришпилити</button>
      </form>`;
    this.notes = this.root.querySelector(".board-notes") as HTMLElement;
    this.plan = this.root.querySelector(".board-plan") as HTMLElement;
    this.input = this.root.querySelector(".board-ta") as HTMLTextAreaElement;

    // контейнер цидулок над зеленою панеллю, з відступом від країв маски
    const IN = 0.03;
    this.notes.style.left = `${(PANEL.x0 + IN) * 100}%`;
    this.notes.style.top = `${(PANEL.y0 + IN) * 100}%`;
    this.notes.style.width = `${(PANEL.x1 - PANEL.x0 - 2 * IN) * 100}%`;
    this.notes.style.height = `${(PANEL.y1 - PANEL.y0 - 2 * IN) * 100}%`;
    // стейдж бере аспект самої дошки
    const bg = this.root.querySelector(".board-bg") as HTMLImageElement;
    const stage = this.root.querySelector(".board-stage") as HTMLElement;
    bg.onload = (): void => {
      if (bg.naturalWidth) stage.style.aspectRatio = `${bg.naturalWidth} / ${bg.naturalHeight}`;
    };

    this.root.querySelector(".board-back")!.addEventListener("click", () => this.onClose());
    this.root.querySelector(".board-pin")!.addEventListener("click", () => this.pin());
    this.input.addEventListener("keydown", (e) => {
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

  /**
   * Партитура віча — предмет на Дошці, а не рядок у лозі.
   *
   * Порядок виступів складає Мамай одним викликом; доти він жив лише всередині ядра, і глядач
   * бачив наслідок (розмову), не бачачи задуму.
   */
  pinPlan(summary: string, steps: string[]): void {
    if (!steps.length) {
      this.plan.classList.remove("on");
      return;
    }
    this.plan.innerHTML = `
      <div class="board-plan-head">${escapeHtml(summary)}</div>
      <ol class="board-plan-steps">${steps.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ol>`;
    this.plan.classList.add("on");
  }

  addTopic(t: Omit<Topic, "id">): Topic {
    const topic: Topic = { id: `t${++this.seq}`, ...t };
    this.jit.set(topic.id, this.place());
    this.topics.unshift(topic);
    // межа: живий бек стрімить теми безперервно → не даємо списку/DOM рости нескінченно
    const CAP = 24;
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
      const cx = 0.1 + Math.random() * 0.8;
      const cy = 0.1 + Math.random() * 0.44; // верхня частина панелі — щоб листки не перекривали форму знизу
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
      el.style.zIndex = String(n - i); // новіші лежать зверху
      el.title = t.text; // повний текст при наведенні (на листку — лише початок)
      el.innerHTML = `
        <span class="note-text">${escapeHtml(t.text)}</span>
        <span class="note-foot">
          <span class="note-heat">${HEAT_UA[t.heat]}</span>
          ${t.author ? `<span class="note-by">— ${escapeHtml(t.author)}</span>` : ""}
        </span>`;
      if (t.heat === "hot") el.addEventListener("click", () => this.onOpenTalk(t));
      this.notes.appendChild(el);
    });
  }

  open(): void {
    this.root.classList.add("on");
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
