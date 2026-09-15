/**
 * What this user's enabled plugins contribute to the interface.
 *
 * One query, shared through the cache by the sidebar, the routes and every
 * panel slot — a page with three slots must not make three requests to learn
 * the same thing.
 */

import { useMemo } from 'react';
import { useQuery } from 'react-query';

import { apiClient } from '../services/api';
import type {
  PanelSlot,
  PluginPanel,
  PluginUiContribution,
  PluginViewSpec,
} from './types';

export const PLUGIN_UI_QUERY_KEY = ['plugin-ui'];

export interface ResolvedPanel {
  slug: string;
  pluginName: string;
  viewId: string;
  view: PluginViewSpec;
  title: string;
}

export interface PluginUi {
  contributions: PluginUiContribution[];
  isLoading: boolean;
  /** The view a route names, or null when nothing contributes it. */
  viewFor: (slug: string, viewId: string) => PluginViewSpec | null;
  /** Every panel contributed to one slot, in install order. */
  panelsFor: (slot: PanelSlot | string) => ResolvedPanel[];
}

export function usePluginUi(): PluginUi {
  const { data, isLoading } = useQuery(
    PLUGIN_UI_QUERY_KEY,
    () => apiClient.getMyPluginUi(),
    { staleTime: 5 * 60 * 1000, retry: 1 }
  );

  const contributions = useMemo(() => data?.plugins || [], [data]);

  return useMemo(
    () => ({
      contributions,
      isLoading,
      viewFor: (slug: string, viewId: string) => {
        const plugin = contributions.find((p) => p.slug === slug);
        return (plugin?.views || {})[viewId] || null;
      },
      panelsFor: (slot: string) =>
        contributions.flatMap((plugin) =>
          (plugin.panels || [])
            .filter((panel: PluginPanel) => panel.slot === slot)
            .map((panel: PluginPanel) => ({
              slug: plugin.slug,
              pluginName: plugin.name,
              viewId: panel.view,
              view: plugin.views[panel.view],
              title: panel.title || plugin.views[panel.view]?.title || plugin.name,
            }))
            // A panel naming a view its own manifest no longer declares cannot
            // happen through install-time validation, but a manifest edited in
            // the database could produce one. Drop it rather than crash a page
            // that is not this plugin's.
            .filter((panel: ResolvedPanel) => Boolean(panel.view))
        ),
    }),
    [contributions, isLoading]
  );
}
