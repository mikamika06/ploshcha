/**
 * Літопис — когнітивний шар ядра, сказаний сільською мовою.
 *
 * Ядро від самого початку надсилало `route.decided`, `tool.called`, `tool.result`,
 * `memory.recalled`, `plan.revised`, `run.degraded`, а стор не мав для них жодного `case`: події
 * проходили валідацію, доїжджали й **нічого не робили**. Тобто на екрані було видно, як люди
 * говорять і ходять, і не видно, ЧОМУ. Найгірше — `run.degraded`: цикл ставав, і про це не
 * дізнавався ніхто.
 *
 * HUD-панель рамкою заборонена, тож це тихий рядок унизу. `route.decided` тут свідомо НЕ показуємо:
 * він на кожен крок, і літопис перетворився б на лог — його місце в інспекторі.
 */

/** Дві форми, бо українська: КУДИ пішли («до книги дяка») і ДЕ шукали («у книзі дяка»). Одна
 *  форма давала «у книги дяка того немає» — правильні дані, зіпсована мова. */
const PLACE_OF_TOOL: Record<string, { to: string; at: string }> = {
  словник: { to: "до книги дяка", at: "у книзі дяка" },
  довідка: { to: "до криниці", at: "у криниці" },
  обчислити: { to: "до ковадла", at: "на ковадлі" },
  lookup_fact: { to: "до криниці", at: "у криниці" },
  check_date: { to: "до книги дяка", at: "у книзі дяка" },
  calc: { to: "до ковадла", at: "на ковадлі" },
};

function place(tool: string): { to: string; at: string } {
  return PLACE_OF_TOOL[tool] ?? { to: `по ${tool}`, at: `у ${tool}` };
}

const CAP = 5;

export class Chronicle {
  private root: HTMLElement;

  constructor() {
    this.root = document.createElement("div");
    this.root.className = "chron";
    document.getElementById("stage")!.appendChild(this.root);
  }

  private say(text: string, kind: "" | "dim" | "warn" = ""): void {
    const el = document.createElement("div");
    el.className = `chron-line${kind ? ` ${kind}` : ""}`;
    el.textContent = text;
    this.root.appendChild(el);
    while (this.root.childElementCount > CAP) this.root.firstElementChild?.remove();
    setTimeout(() => el.remove(), 14000);
  }

  called(tool: string): void {
    this.say(`пішли ${place(tool).to}`, "dim");
  }

  resulted(tool: string, ok: boolean, found: boolean | null | undefined): void {
    const { at } = place(tool);
    if (!ok) return this.say(`${at} не озвалось`, "warn");
    // ★ «не знайшов» ≠ «зламався» — ядро тримає це тризначним, і сцена не має права зливати.
    if (found === false) return this.say(`${at} того немає`);
    if (found === true) return this.say(`${at} знайшли`);
    this.say(`спитали ${at}`, "dim");
  }

  recalled(count: number): void {
    if (count > 0) this.say(`пригадали: ${count}`, "dim");
  }

  revised(reason: string): void {
    this.say(`передумали: ${reason}`.slice(0, 90));
  }

  degraded(stage: string, reason?: string): void {
    this.say(`віче стало — ${reason ?? stage}`, "warn");
  }

  outcome(kind: string): void {
    this.say(kind === "answer" ? "віче дійшло згоди" : `віче скінчилось: ${kind}`);
  }

  /** Слово Оповідача — не службовий рядок, тому виділене й тримається довше. */
  chronicle(title: string, narration: string): void {
    const said = [title, narration].filter(Boolean).join(" — ");
    if (said) this.say(said.slice(0, 160));
  }

  thought(name: string): void {
    this.say(`${name} лишився при своєму`, "dim");
  }

  /** Ухвала — єдине, що лишається після віча, тож у літописі вона виділена. */
  decided(label: string): void {
    this.say(`ухвалили: ${label}`.slice(0, 140));
  }

  clear(): void {
    this.root.innerHTML = "";
  }
}
