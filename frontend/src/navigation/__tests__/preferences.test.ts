import { NAV_CATALOG, buildCatalog, navKey } from '../catalog';
import { applyNavPreferences, landingPath } from '../preferences';

const ctx = { isAdmin: false, latexEnabled: false };
const catalog = () => buildCatalog(ctx);

const keysOf = (doors: ReturnType<typeof catalog>, doorId: string) =>
  doors.find((d) => d.id === doorId)?.sections.map((s) => s.key) ?? [];

describe('the catalog before any preference applies', () => {
  it('hides admin-only destinations from a non-admin', () => {
    const doors = buildCatalog({ isAdmin: false, latexEnabled: false });

    expect(keysOf(doors, 'settings')).not.toContain('/usage');
    expect(keysOf(doors, 'settings')).toContain('/tools');
  });

  it('offers LaTeX Studio to an admin even when the compiler is off', () => {
    // An admin is the person who turns it on; hiding the page from them is
    // hiding the switch.
    const asAdmin = buildCatalog({ isAdmin: true, latexEnabled: false });
    const asUser = buildCatalog({ isAdmin: false, latexEnabled: false });

    expect(keysOf(asAdmin, 'synthesis')).toContain('/latex');
    expect(keysOf(asUser, 'synthesis')).not.toContain('/latex');
  });

  it('identifies the two admin tabs separately', () => {
    // Both are `/admin`; only the search string tells them apart, so a
    // preference stored against the path alone would move or hide both.
    const doors = buildCatalog({ isAdmin: true, latexEnabled: false });
    const keys = keysOf(doors, 'settings');

    expect(keys).toContain('/admin?tab=overview');
    expect(keys).toContain('/admin?tab=agents');
  });
});

describe('ordering', () => {
  it('reorders the doors a person listed', () => {
    const doors = applyNavPreferences(catalog(), {
      doorOrder: ['rnd', 'synthesis'],
    });

    expect(doors.map((d) => d.id).slice(0, 2)).toEqual(['rnd', 'synthesis']);
  });

  it('keeps doors the stored order never mentioned', () => {
    // Otherwise every door added after someone last saved these settings
    // would vanish for them, and read as a feature that never shipped.
    const doors = applyNavPreferences(catalog(), { doorOrder: ['rnd'] });

    expect(doors.map((d) => d.id).sort()).toEqual(
      catalog().map((d) => d.id).sort()
    );
  });

  it('keeps Settings at the bottom however it was ordered', () => {
    const doors = applyNavPreferences(catalog(), {
      doorOrder: ['settings', 'rnd'],
    });

    expect(doors[doors.length - 1].id).toBe('settings');
  });

  it('pins a destination to the top of its door', () => {
    const doors = applyNavPreferences(catalog(), {
      pinned: ['/agent-control-plane'],
    });

    expect(keysOf(doors, 'rnd')[0]).toBe('/agent-control-plane');
  });
});

describe('hiding', () => {
  it('removes a destination a person hid', () => {
    const doors = applyNavPreferences(catalog(), { hidden: ['/papers'] });

    expect(keysOf(doors, 'library')).not.toContain('/papers');
  });

  it('still shows the page you are standing on', () => {
    // Reachable by link, redirect or bookmark. A sidebar that cannot show
    // where you are is worse than one showing an entry you asked to remove.
    const doors = applyNavPreferences(
      catalog(),
      { hidden: ['/papers'] },
      { activeKey: '/papers' }
    );

    expect(keysOf(doors, 'library')).toContain('/papers');
  });

  it('drops a door once its last destination is hidden', () => {
    const chat = catalog().find((d) => d.id === 'chat')!;
    const doors = applyNavPreferences(catalog(), {
      hidden: chat.sections.map((s) => s.key),
    });

    expect(doors.map((d) => d.id)).not.toContain('chat');
  });
});

describe('renaming', () => {
  it('uses the label a person chose', () => {
    const doors = applyNavPreferences(catalog(), {
      renamed: { '/autonomous-agents': 'My Runs' },
    });

    const runs = doors
      .find((d) => d.id === 'rnd')!
      .sections.find((s) => s.key === '/autonomous-agents')!;
    expect(runs.name).toBe('My Runs');
  });

  it('keeps the key stable across a rename, so other preferences still bite', () => {
    const doors = applyNavPreferences(catalog(), {
      renamed: { '/autonomous-agents': 'My Runs' },
      pinned: ['/autonomous-agents'],
    });

    expect(keysOf(doors, 'rnd')[0]).toBe('/autonomous-agents');
  });
});

describe('the landing page', () => {
  it('defaults to chat', () => {
    expect(landingPath(catalog(), undefined)).toBe('/chat');
  });

  it('honours a reachable choice', () => {
    expect(landingPath(catalog(), { landing: '/documents' })).toBe('/documents');
  });

  it('falls back when the choice is no longer reachable', () => {
    // A page they lost admin for, or that was removed. The alternative is an
    // account that opens on a 404 with no way back but typing a URL.
    expect(landingPath(catalog(), { landing: '/usage' })).toBe('/chat');
    expect(landingPath(catalog(), { landing: '/gone' })).toBe('/chat');
  });

  it('allows an admin the page a non-admin cannot land on', () => {
    const adminCatalog = buildCatalog({ isAdmin: true, latexEnabled: false });

    expect(landingPath(adminCatalog, { landing: '/usage' })).toBe('/usage');
  });
});

describe('no preferences at all', () => {
  it('leaves the catalog exactly as it is', () => {
    const before = catalog();
    const after = applyNavPreferences(before, undefined);

    expect(after.map((d) => d.id)).toEqual(before.map((d) => d.id));
    expect(keysOf(after, 'library')).toEqual(keysOf(before, 'library'));
  });

  it('every catalog key is unique', () => {
    // Preferences are stored against keys; two entries sharing one would make
    // hiding or pinning either of them affect both.
    const keys = NAV_CATALOG.flatMap((d) => d.sections.map((s) => navKey(s.to)));

    expect(new Set(keys).size).toBe(keys.length);
  });
});
