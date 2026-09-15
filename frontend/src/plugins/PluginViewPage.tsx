/**
 * The page a contributed navigation entry leads to: `/p/:slug/:viewId`.
 *
 * A plugin gets one route shape rather than the ability to declare paths.
 * Letting a manifest choose its own URL would let a plugin claim `/documents`
 * or `/admin`, and the collision would be resolved by whichever route
 * happened to be registered first — a way for an installed bundle to shadow a
 * first-party page.
 */

import { AlertTriangle } from 'lucide-react';
import React from 'react';
import { useParams } from 'react-router-dom';

import LoadingSpinner from '../components/common/LoadingSpinner';
import PluginView from './PluginView';
import { usePluginUi } from './usePluginUi';

export const PluginViewPage: React.FC = () => {
  const { slug = '', viewId = '' } = useParams();
  const { viewFor, contributions, isLoading } = usePluginUi();

  if (isLoading) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const view = viewFor(slug, viewId);
  const plugin = contributions.find((p) => p.slug === slug);

  if (!view) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-700" />
          <div className="text-sm text-amber-800">
            <p className="font-medium">This page is no longer available.</p>
            <p className="mt-1 text-xs">
              {plugin
                ? `The plugin "${plugin.name}" no longer contributes a view called "${viewId}".`
                : `No enabled plugin called "${slug}" contributes this page. It may have been uninstalled or disabled.`}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl p-6">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-900">
          {view.title || viewId}
        </h1>
        <p className="text-gray-500">
          {view.description || `Contributed by ${plugin?.name || slug}`}
        </p>
      </div>
      <div className="rounded-lg border border-gray-300 bg-white p-4">
        <PluginView slug={slug} viewId={viewId} view={view} />
      </div>
    </div>
  );
};

export default PluginViewPage;
