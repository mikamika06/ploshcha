const KEY = "ploshcha.sid";

/**
 * Ідентифікатор гостя: під ним ядро тримає ОКРЕМЕ село — свою памʼять, свої ухвали, свої чутки.
 *
 * Живе в `localStorage`, а не в памʼяті вкладки, і це головне рішення тут: памʼять села мусить
 * переживати перезавантаження сторінки. Ідентифікатор у змінній означав би, що кожен F5 —
 * новий безрідний гість, тобто памʼяті немає взагалі, скільки б її ядро не зберігало.
 *
 * Абетка вузька навмисно: на боці ядра цей рядок стає ІМʼЯМ ФАЙЛУ, і все, що не проходить його
 * перевірку, тихо трактується як «спільне село».
 */
function mint(): string {
  const source = globalThis.crypto;
  if (source && typeof source.randomUUID === "function") return source.randomUUID();
  // Старий webview без `randomUUID`. Випадковості тут вистачає: це не секрет і не ключ —
  // зіткнення означає спільну памʼять двох гостей, а не доступ до чужого.
  return `s-${Math.random().toString(36).slice(2, 12)}${Date.now().toString(36)}`;
}

let cached: string | undefined;

export function sessionId(): string {
  if (cached !== undefined) return cached;
  try {
    const saved = localStorage.getItem(KEY);
    if (saved) {
      cached = saved;
      return saved;
    }
    const fresh = mint();
    localStorage.setItem(KEY, fresh);
    cached = fresh;
    return fresh;
  } catch {
    // Приватний режим забороняє сховище. Тоді сесія живе рівно вкладку — гірше за памʼять, але
    // краще, ніж падіння на старті через `SecurityError`.
    cached = mint();
    return cached;
  }
}
