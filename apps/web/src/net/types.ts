import type { ParsedEvent } from "@ploshcha/contract-ts";

export interface EventSourcePort {
  /** Підписатися на потік подій. Повертає функцію відписки. */
  subscribe(onEvent: (ev: ParsedEvent) => void, onEnd?: () => void): () => void;
}
