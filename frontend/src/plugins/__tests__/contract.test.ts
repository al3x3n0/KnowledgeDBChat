/**
 * The backend decides what a plugin may declare; this side renders it. Three
 * lists therefore exist in two languages, and each has a failure mode that is
 * silent rather than loud:
 *
 * - a **panel slot** the backend accepts but no page hosts → the panel
 *   installs cleanly and appears nowhere;
 * - a **door** the backend accepts but the catalog does not have → the nav
 *   entry installs cleanly and appears nowhere;
 * - an **icon** the backend accepts but this side cannot resolve → a silent
 *   fallback to a generic box.
 *
 * Reading the Python directly is deliberate. A copy of each list maintained
 * here would be a fourth thing to keep in step, which is the problem rather
 * than the fix.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

import { NAV_CATALOG } from '../../navigation/catalog';
import { PLUGIN_ICONS } from '../navIcons';
import { PANEL_SLOTS } from '../types';

const BACKEND = join(__dirname, '../../../../backend/app/services/plugin_ui.py');

/** The string entries of a top-level `NAME = (...)` tuple in the Python. */
function pythonTuple(source: string, name: string): string[] {
  const match = source.match(new RegExp(`^${name} = \\(([\\s\\S]*?)\\)`, 'm'));
  if (!match) throw new Error(`${name} not found in plugin_ui.py`);
  return Array.from(match[1].matchAll(/"([^"]+)"/g)).map((m) => m[1]);
}

const backendSource = readFileSync(BACKEND, 'utf8');

describe('the manifest contract, across both halves', () => {
  it('every panel slot the backend accepts is hosted by a page', () => {
    const declared = pythonTuple(backendSource, 'PANEL_SLOTS');

    // Search the source for a real host rather than trusting a list.
    const hosted = declared.filter((slot) => {
      const found = execSync(
        `grep -rl 'slot="${slot}"' ${join(__dirname, '../..')} || true`,
        { encoding: 'utf8' }
      ).trim();
      return found.length > 0;
    });

    expect(hosted.sort()).toEqual(declared.sort());
  });

  it('the frontend slot list matches the backend one', () => {
    expect([...PANEL_SLOTS].sort()).toEqual(
      pythonTuple(backendSource, 'PANEL_SLOTS').sort()
    );
  });

  it('every door the backend accepts exists in the catalog', () => {
    const declared = pythonTuple(backendSource, 'NAV_DOORS').sort();
    const actual = NAV_CATALOG.map((d) => d.id).sort();

    expect(declared).toEqual(actual);
  });

  it('every icon the backend accepts resolves to a component', () => {
    const declared = pythonTuple(backendSource, 'ICONS');

    const unresolved = declared.filter((name) => !PLUGIN_ICONS[name]);
    expect(unresolved).toEqual([]);
  });

  it('offers no icon the backend would refuse', () => {
    expect(Object.keys(PLUGIN_ICONS).sort()).toEqual(
      pythonTuple(backendSource, 'ICONS').sort()
    );
  });
});
