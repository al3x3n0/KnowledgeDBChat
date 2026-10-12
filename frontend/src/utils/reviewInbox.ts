/**
 * How long something has been waiting, in the words a person would use.
 *
 * Kept apart from the page so the boundaries can be tested without rendering:
 * a timestamp from the server is UTC and may arrive without a zone suffix,
 * which `new Date` would read as local time.
 */
export function waitingFor(since?: string | null, now: Date = new Date()): string | null {
  if (!since) return null;
  const hasZone = /([zZ]|[+-]\d{2}:?\d{2})$/.test(since);
  const then = new Date(hasZone ? since : `${since}Z`);
  if (Number.isNaN(then.getTime())) return null;

  const minutes = Math.floor((now.getTime() - then.getTime()) / 60000);
  // A clock a little ahead of the server's is not "in the future".
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h`;
  const days = Math.floor(hours / 24);
  return `${days} ${days === 1 ? 'day' : 'days'}`;
}
