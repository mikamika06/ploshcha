import type { PloshchaEvent } from "@ploshcha/contract-ts";

export interface EventSourcePort {
  /** Підписатися на потік подій. Повертає функцію відписки. */
  subscribe(onEvent: (ev: PloshchaEvent) => void, onEnd?: () => void): () => void;
}
