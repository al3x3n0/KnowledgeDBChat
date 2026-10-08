/**
 * A URL that came from data, made safe to put in an `href`.
 *
 * React renders `href={url}` whatever the scheme, and `javascript:` runs on
 * click. The URLs here come from ingested papers, inbox items, and
 * knowledge-graph entities a model extracted from documents -- so the value
 * is somebody else's text, not ours. Only http(s) and same-origin paths get
 * through; anything else returns undefined, which renders an anchor that
 * goes nowhere rather than one that runs.
 */
export function safeExternalUrl(url: unknown): string | undefined {
  if (typeof url !== 'string') return undefined;
  const trimmed = url.trim();
  if (!trimmed) return undefined;
  // A path on this origin ("/files/1"), but not a protocol-relative "//host".
  if (trimmed.startsWith('/') && !trimmed.startsWith('//')) return trimmed;
  try {
    const parsed = new URL(trimmed);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? trimmed : undefined;
  } catch {
    return undefined;
  }
}
