/**
 * Reading the part of a tool's output a view asked for.
 *
 * Kept separate and pure because the interesting behaviour is what happens
 * when the path finds nothing. A plugin author writing `result.items` against
 * a tool that returns `{items: [...]}` gets an empty table, and an empty table
 * is indistinguishable from a tool that legitimately returned nothing — so
 * this reports *which* it was, and the renderer says so.
 */

export type Resolution =
  | { found: true; value: unknown }
  | { found: false; missingAt: string };

export function resolvePath(data: unknown, path: string): Resolution {
  const trimmed = (path || '').trim();
  if (!trimmed) return { found: true, value: data };

  let current: unknown = data;
  const walked: string[] = [];
  for (const segment of trimmed.split('.')) {
    if (!segment) continue;
    if (current === null || typeof current !== 'object') {
      return { found: false, missingAt: walked.join('.') || '(root)' };
    }
    if (!(segment in (current as Record<string, unknown>))) {
      walked.push(segment);
      return { found: false, missingAt: walked.join('.') };
    }
    current = (current as Record<string, unknown>)[segment];
    walked.push(segment);
  }
  return { found: true, value: current };
}

/** Rows for a table, whatever the tool returned. */
export function asRows(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) {
    return value.filter(
      (row): row is Record<string, unknown> =>
        row !== null && typeof row === 'object' && !Array.isArray(row)
    );
  }
  // A single object is one row. Tools that return "the latest" rather than
  // "the list" are common enough that refusing them would be pedantry.
  if (value !== null && typeof value === 'object') {
    return [value as Record<string, unknown>];
  }
  return [];
}

/**
 * A cell, as text.
 *
 * Everything reaching here was written by a plugin author, so it is rendered
 * as a string and never as markup. Objects are JSON rather than
 * "[object Object]", which is the difference between a table you can debug and
 * one you cannot.
 */
export function asCell(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
