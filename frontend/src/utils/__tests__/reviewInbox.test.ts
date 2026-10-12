import { waitingFor } from '../reviewInbox';

const NOW = new Date('2026-10-12T12:00:00Z');

describe('waitingFor', () => {
  it('says nothing when there is no time to go on', () => {
    expect(waitingFor(null, NOW)).toBeNull();
    expect(waitingFor(undefined, NOW)).toBeNull();
    expect(waitingFor('not a date', NOW)).toBeNull();
  });

  it('reads a timestamp with no zone as UTC, which is what the server sends', () => {
    expect(waitingFor('2026-10-12T11:30:00', NOW)).toBe('30 min');
    expect(waitingFor('2026-10-12T11:30:00Z', NOW)).toBe('30 min');
    expect(waitingFor('2026-10-12T13:30:00+02:00', NOW)).toBe('30 min');
  });

  it('moves from minutes to hours to days at the boundaries', () => {
    expect(waitingFor('2026-10-12T11:59:40Z', NOW)).toBe('just now');
    expect(waitingFor('2026-10-12T11:01:00Z', NOW)).toBe('59 min');
    expect(waitingFor('2026-10-12T11:00:00Z', NOW)).toBe('1 h');
    expect(waitingFor('2026-10-11T12:00:01Z', NOW)).toBe('23 h');
    expect(waitingFor('2026-10-11T12:00:00Z', NOW)).toBe('1 day');
    expect(waitingFor('2026-10-03T12:00:00Z', NOW)).toBe('9 days');
  });

  it('does not report a time in the future', () => {
    expect(waitingFor('2026-10-12T12:05:00Z', NOW)).toBe('just now');
  });
});
