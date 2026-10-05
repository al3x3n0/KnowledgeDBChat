import React from 'react';

/**
 * Why an inbox item was surfaced.
 *
 * The monitor profile learns from what you keep and dismiss, and the research
 * runner scores every candidate against it. That opinion used to reach the UI
 * as `Discovery why: source_type:paper:3, token_bias, phrase_bias` — enough to
 * know an opinion existed, not enough to argue with it. The backend now keeps
 * the terms (`research_discovery_signals`), and this renders them.
 *
 * Favourable and unfavourable signals are shown apart on purpose. A term you
 * dismissed explains a ranking *down*, so listing it after "Surfaced because"
 * would state the opposite of what happened.
 */

export interface DiscoverySignal {
  kind: 'phrase' | 'token' | 'source_type';
  term: string;
  /** Signed net over occurrences in kept minus dismissed items — not a count of items. */
  weight: number;
  favourable: boolean;
  label: string;
  text: string;
}

/** Older rows carry only the pre-change debug strings; newer ones may be absent entirely. */
export function readDiscoverySignals(
  metadata: Record<string, unknown> | undefined | null
): DiscoverySignal[] {
  const raw = metadata?.discovery_signals;
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (entry): entry is DiscoverySignal =>
      !!entry &&
      typeof entry === 'object' &&
      typeof (entry as DiscoverySignal).label === 'string' &&
      typeof (entry as DiscoverySignal).weight === 'number'
  );
}

export function readLegacyReasons(
  metadata: Record<string, unknown> | undefined | null
): string[] {
  const raw = metadata?.discovery_reasons;
  if (!Array.isArray(raw)) return [];
  return raw.filter((entry): entry is string => typeof entry === 'string' && !!entry.trim());
}

function Chip({ signal }: { signal: DiscoverySignal }) {
  const sign = signal.weight > 0 ? '+' : '';
  return (
    <span
      title={`${signal.text} (learned weight ${sign}${signal.weight})`}
      className={`inline-flex items-center rounded px-1.5 py-0.5 border ${
        signal.favourable
          ? 'border-emerald-700/40 text-emerald-700'
          : 'border-rose-700/40 text-rose-700'
      }`}
    >
      {signal.favourable ? null : <span className="mr-1" aria-hidden="true">↓</span>}
      {signal.label}
    </span>
  );
}

export function DiscoveryWhy({
  metadata,
  className = '',
}: {
  metadata: Record<string, unknown> | undefined | null;
  className?: string;
}) {
  const signals = readDiscoverySignals(metadata);

  if (signals.length === 0) {
    // Pre-change rows kept their reasons in the old shape. Rendering them as
    // they are beats hiding them: they are ugly, but they are what was known.
    const legacy = readLegacyReasons(metadata);
    if (legacy.length === 0) return null;
    return (
      <p className={`text-xs text-gray-500 mt-2 ${className}`}>
        Why this: {legacy.slice(0, 4).join(', ')}
      </p>
    );
  }

  const favourable = signals.filter((signal) => signal.favourable);
  const against = signals.filter((signal) => !signal.favourable);

  return (
    <p className={`text-xs text-gray-500 mt-2 flex flex-wrap items-center gap-1.5 ${className}`}>
      {favourable.length > 0 ? (
        <>
          <span className="text-gray-400">Surfaced because it matches</span>
          {favourable.map((signal) => (
            <Chip key={`${signal.kind}:${signal.term}`} signal={signal} />
          ))}
        </>
      ) : null}
      {against.length > 0 ? (
        <>
          <span className="text-gray-400">{favourable.length > 0 ? 'despite' : 'Ranked down by'}</span>
          {against.map((signal) => (
            <Chip key={`${signal.kind}:${signal.term}`} signal={signal} />
          ))}
        </>
      ) : null}
    </p>
  );
}

export default DiscoveryWhy;
