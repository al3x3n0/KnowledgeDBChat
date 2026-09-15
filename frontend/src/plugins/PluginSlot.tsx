/**
 * A named place in a first-party page where plugins may add a panel.
 *
 * Renders nothing at all when no plugin contributes to the slot, which is the
 * normal case — a slot must cost nothing in a deployment with no plugins, in
 * markup as well as in requests.
 */

import { Package } from 'lucide-react';
import React from 'react';

import PluginView from './PluginView';
import { usePluginUi } from './usePluginUi';
import type { PanelSlot } from './types';

export const PluginSlot: React.FC<{
  slot: PanelSlot | string;
  compact?: boolean;
}> = ({ slot, compact = true }) => {
  const { panelsFor } = usePluginUi();
  const panels = panelsFor(slot);

  if (panels.length === 0) return null;

  return (
    <>
      {panels.map((panel) => (
        <div
          key={`${panel.slug}:${panel.viewId}`}
          className="rounded-lg border border-gray-300 bg-gray-100 p-3"
        >
          <div className="mb-2 flex items-center gap-1.5">
            <Package className="h-3.5 w-3.5 text-gray-500" />
            <span className="text-xs font-medium text-gray-900">
              {panel.title}
            </span>
            {/* Say which plugin put this here. A panel in someone else's page
                that does not name its source is indistinguishable from a
                first-party feature that is behaving oddly. */}
            <span className="ml-auto text-[11px] text-gray-500">
              {panel.pluginName}
            </span>
          </div>
          <PluginView
            slug={panel.slug}
            viewId={panel.viewId}
            view={panel.view}
            compact={compact}
          />
        </div>
      ))}
    </>
  );
};

export default PluginSlot;
