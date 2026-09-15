/**
 * Applying a person's own choices to the navigation catalog.
 *
 * Deliberately pure: catalog in, preferences in, doors out. The rules below
 * are the kind that are easy to get subtly wrong -- an ordering that drops
 * entries it was not told about, a hide that removes the page you are
 * standing on -- and a pure function is one a test can pin without rendering
 * anything.
 *
 * Three rules are worth stating because each is a decision rather than an
 * implementation detail:
 *
 * 1. **An order is a preference, not a whitelist.** Entries the stored order
 *    does not mention keep their catalog position after the ones it does.
 *    Otherwise every destination added to the application after you last
 *    touched these settings would be invisible to you, and it would look like
 *    the feature had not shipped.
 *
 * 2. **Hiding never hides the page you are on.** A hidden entry you navigate
 *    to anyway -- from a link, a redirect, a bookmark -- is shown while it is
 *    active, because a sidebar that cannot show where you are is worse than
 *    one showing an entry you asked to remove.
 *
 * 3. **A door with nothing left in it disappears.** But a door is never
 *    itself hidden directly; it goes when its last section does. One less
 *    concept, and no way to end up with a door you cannot open.
 */

import type { NavDoor, NavItem } from './catalog';
import { navKey } from './catalog';

export interface NavPreferences {
  doorOrder?: string[];
  hidden?: string[];
  renamed?: Record<string, string>;
  pinned?: string[];
  landing?: string;
}

export interface UiPreferences {
  nav?: NavPreferences;
}

/** Order `items` by `order`, keeping anything it does not mention. */
function applyOrder<T>(items: T[], order: string[], keyOf: (item: T) => string): T[] {
  if (!order.length) return items;
  const position = new Map(order.map((key, index) => [key, index]));
  const known: T[] = [];
  const rest: T[] = [];
  items.forEach((item) => (position.has(keyOf(item)) ? known : rest).push(item));
  known.sort((a, b) => position.get(keyOf(a))! - position.get(keyOf(b))!);
  return [...known, ...rest];
}

export interface ApplyOptions {
  /** The key of the destination currently open, which is never hidden. */
  activeKey?: string | null;
}

export function applyNavPreferences(
  catalog: NavDoor[],
  prefs: NavPreferences | undefined,
  options: ApplyOptions = {}
): NavDoor[] {
  const hidden = new Set(prefs?.hidden || []);
  const renamed = prefs?.renamed || {};
  const pinned = prefs?.pinned || [];
  const active = options.activeKey || null;

  const doors = catalog
    .map((door) => {
      const sections = door.sections
        .filter((s) => !hidden.has(s.key) || s.key === active)
        .map((s) => (renamed[s.key] ? { ...s, name: renamed[s.key] } : s));
      return { ...door, sections };
    })
    .filter((door) => door.sections.length > 0);

  // Utility doors stay at the bottom whatever the stored order says: the
  // stored order is about the doors you work in, and letting Settings be
  // dragged into the middle of them would be a way to make the nav worse by
  // accident rather than a customization anyone wants.
  const work = doors.filter((d) => !d.utility);
  const utility = doors.filter((d) => d.utility);

  return [
    ...applyOrder(work, prefs?.doorOrder || [], (d) => d.id),
    ...utility,
  ].map((door) => ({
    ...door,
    sections: applyOrder(door.sections, pinned, (s) => s.key),
  }));
}

/** Every destination this user is permitted, flattened, before hiding. */
export function catalogItems(catalog: NavDoor[]): NavItem[] {
  return catalog.flatMap((d) => d.sections);
}

/**
 * Where "/" should land.
 *
 * Falls back to `/chat` when the stored landing page is one this user can no
 * longer reach -- a page they lost admin for, or that was removed -- because
 * the alternative is an account that opens on a 404 with no way back except
 * typing a URL.
 */
export function landingPath(
  catalog: NavDoor[],
  prefs: NavPreferences | undefined,
  fallback = '/chat'
): string {
  const wanted = prefs?.landing;
  if (!wanted) return fallback;
  const reachable = catalogItems(catalog).some((s) => navKey(s.to) === wanted);
  return reachable ? wanted : fallback;
}
