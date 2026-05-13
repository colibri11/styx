// UUID validation helper для tool params.
//
// Используется в двух режимах:
//
// 1. **session_id** (Fix 1): silent drop. Если value не UUID-format —
//    возвращаем null, не падаем и не пишем warn. openclaw-cli может
//    передать non-UUID sessionKey через ctx.sessionId, и tool должен
//    деградировать до session-less вызова, а не ломать turn.
//
// 2. **memory_id / entity_id / memory_ids[]** (Fix 4): explicit reject.
//    Tool сам решает что делать с null'ом (обычно — return styxErrorResult
//    с понятным сообщением). Caller получает контролируемый ответ.

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Проверка строки на UUID-format. Не trim'ит — caller должен сделать
 * это сам если есть пробелы.
 */
export function isUuid(value: unknown): value is string {
  return typeof value === "string" && UUID_RE.test(value);
}

/**
 * Coerce session_id-like значение в либо UUID-string, либо null.
 * Принимает unknown (typeof guard внутри) — удобно для toolCtx.sessionId
 * + toolParams.session_id, оба могут быть undefined.
 *
 * Не throw, не log — silent drop по Fix 1.
 */
export function coerceSessionUuid(value: unknown): string | null {
  if (typeof value !== "string") return null;
  return UUID_RE.test(value) ? value : null;
}
