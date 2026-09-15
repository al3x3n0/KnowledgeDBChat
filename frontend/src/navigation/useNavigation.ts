/**
 * The navigation this user actually sees, and the means to change it.
 *
 * One hook so that the sidebar, the settings editor and the "/" redirect all
 * read the same resolved answer. They used to disagree by construction: the
 * doors were a literal in `Layout.tsx` and the landing page was a hardcoded
 * `<Navigate to="/chat">` in `App.tsx`, so "where does this account start"
 * had no single place to ask.
 */

import { useCallback, useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';

import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../services/api';
import type { LatexStatusResponse } from '../types';
import { pluginIcon } from '../plugins/navIcons';
import { usePluginUi } from '../plugins/usePluginUi';
import { buildCatalog, navKey, withContributions } from './catalog';
import type { ContributedNavEntry, NavDoor, NavItem } from './catalog';
import { applyNavPreferences, landingPath } from './preferences';
import type { NavPreferences, UiPreferences } from './preferences';

export const NAV_PREFERENCES_QUERY_KEY = ['user-preferences-ui'];

export interface Navigation {
  /** Doors as this person has arranged them. */
  doors: NavDoor[];
  /** Everything they are permitted, before hiding -- what settings lists. */
  catalog: NavDoor[];
  nav: NavPreferences;
  activeItem: NavItem | null;
  activeDoor: NavDoor | null;
  landing: string;
  isLoading: boolean;
  save: (next: NavPreferences) => Promise<unknown>;
  isSaving: boolean;
}

/** Does this entry match where we are? */
export function isActiveNavItem(
  item: NavItem,
  pathname: string,
  search: string
): boolean {
  const target = typeof item.to === 'string' ? item.to : item.to.pathname;
  if (!pathname.startsWith(target)) return false;
  if (typeof item.to === 'string') return true;

  const desiredTab = item.to.search
    ? new URLSearchParams(item.to.search).get('tab')
    : null;
  if (!desiredTab) return true;
  const currentTab = new URLSearchParams(search).get('tab') || 'overview';
  return desiredTab === currentTab;
}

export function useNavigation(): Navigation {
  const { user } = useAuth();
  const location = useLocation();
  const queryClient = useQueryClient();

  const { data: latexStatus } = useQuery<LatexStatusResponse>(
    ['latex-status-nav'],
    () => apiClient.getLatexStatus(),
    { staleTime: 5 * 60 * 1000, retry: 1 }
  );

  const { data: preferences, isLoading } = useQuery(
    NAV_PREFERENCES_QUERY_KEY,
    () => apiClient.getMyPreferences(),
    { staleTime: 5 * 60 * 1000, retry: 1 }
  );

  const nav: NavPreferences = useMemo(() => {
    const ui = (preferences?.ui || {}) as UiPreferences;
    return ui.nav || {};
  }, [preferences]);

  const { contributions } = usePluginUi();

  const contributedNav: ContributedNavEntry[] = useMemo(
    () =>
      contributions.flatMap((plugin) =>
        (plugin.nav || []).map((entry) => ({
          door: entry.door,
          name: entry.name,
          to: `/p/${plugin.slug}/${entry.view}`,
          icon: pluginIcon(entry.icon),
        }))
      ),
    [contributions]
  );

  const catalog = useMemo(
    () =>
      withContributions(
        buildCatalog({
          isAdmin: user?.role === 'admin',
          latexEnabled: Boolean(latexStatus?.enabled),
        }),
        contributedNav
      ),
    [user?.role, latexStatus?.enabled, contributedNav]
  );

  const activeItem = useMemo(() => {
    for (const door of catalog) {
      for (const section of door.sections) {
        if (isActiveNavItem(section, location.pathname, location.search)) {
          return section;
        }
      }
    }
    return null;
  }, [catalog, location.pathname, location.search]);

  const doors = useMemo(
    () =>
      applyNavPreferences(catalog, nav, {
        activeKey: activeItem ? activeItem.key : null,
      }),
    [catalog, nav, activeItem]
  );

  const activeDoor = useMemo(() => {
    if (!activeItem) return doors[0] || null;
    return (
      doors.find((d) => d.sections.some((s) => s.key === activeItem.key)) ||
      doors[0] ||
      null
    );
  }, [activeItem, doors]);

  const mutation = useMutation(
    (next: NavPreferences) =>
      // The whole `ui` document is sent, not a patch: the column is stored
      // wholesale, and a partial write would silently drop the keys it left
      // out.
      apiClient.updateMyPreferences({ ui: { nav: next } }),
    {
      onSuccess: (saved) => {
        queryClient.setQueryData(NAV_PREFERENCES_QUERY_KEY, saved);
      },
    }
  );

  const save = useCallback(
    (next: NavPreferences) => mutation.mutateAsync(next),
    [mutation]
  );

  return {
    doors,
    catalog,
    nav,
    activeItem,
    activeDoor,
    landing: landingPath(catalog, nav),
    isLoading,
    save,
    isSaving: mutation.isLoading,
  };
}

export { navKey };
