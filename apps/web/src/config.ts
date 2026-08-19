// Затримка між подіями при відтворенні фікстури (мс).
export const REPLAY_MS = 1100;

const RAW = (import.meta.env?.VITE_LIVE_URL as string | undefined)?.trim() ?? "";

/**
 * Адреса живого ядра. Три стани, без двозначності:
 *   порожньо        → фронт грає записану фікстуру (нуль токенів)
 *   `same-origin`   → ядро роздає цю ж збірку, тож звертаємось відносно (прод: один процес)
 *   URL             → чуже ядро на іншому порті (розробка з Vite)
 */
export const LIVE_URL: string = RAW === "same-origin" ? "" : RAW;

export const IS_LIVE = RAW.length > 0;
