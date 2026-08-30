import type { ParsedEvent } from "@ploshcha/contract-ts";

export interface EventSourcePort {
  /** Підписатися на потік подій. Повертає функцію відписки. */
  subscribe(onEvent: (ev: ParsedEvent) => void, onEnd?: () => void): () => void;
  /**
   * Забути те, що ще не показано. Є лише там, де є притримка (`LiveDriver`): запис
   * (`FixtureDriver`) відтворюється рівно так, як його зняли, і кидати з нього нічого.
   */
  drop?(): void;
}
