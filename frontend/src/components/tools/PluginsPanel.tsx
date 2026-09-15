/**
 * Plugins: what this user may run, and what each one would be allowed to do.
 *
 * The listing deliberately leads with each contributed tool's *derived*
 * classification rather than its description. A manifest's author says what a
 * tool is for; the executor type decides what it can reach, and that is the
 * thing someone needs before installing. A webhook described as "read some
 * metrics" still posts to an arbitrary URL, and the row says so.
 *
 * Disabled is not uninstalled, and the two actions are separated for that
 * reason: disabling keeps the settings and is the reversible thing to reach
 * for when a plugin misbehaves.
 */

import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Globe,
  Lock,
  Package,
  Plus,
  Power,
  Trash2,
} from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';
import toast from 'react-hot-toast';

import { apiClient } from '../../services/api';
import type { ContributedToolView, Plugin } from '../../types';

const EXAMPLE_MANIFEST = `{
  "id": "bench",
  "name": "Benchmarks",
  "version": "0.1.0",
  "description": "Record and fetch benchmark numbers",
  "contributes": {
    "tools": [
      {
        "name": "note",
        "description": "Record a benchmark note",
        "tool_type": "transform",
        "parameters_schema": {
          "type": "object",
          "properties": { "text": { "type": "string" } },
          "required": ["text"]
        },
        "config": { "template": "noted: {{ text }}" },
        "job_types": ["research"]
      }
    ]
  }
}`;

const EFFECT_STYLE: Record<string, string> = {
  read: 'bg-gray-200 text-gray-700',
  write: 'bg-amber-100 text-amber-800',
};

const ToolRow: React.FC<{ tool: ContributedToolView }> = ({ tool }) => (
  <div className="flex items-start gap-3 rounded border border-gray-200 bg-white px-3 py-2">
    <div className="min-w-0 flex-1">
      <div className="flex items-center gap-2">
        <code className="text-xs font-medium text-gray-900">{tool.name}</code>
        <span className="text-[11px] text-gray-500">{tool.tool_type}</span>
      </div>
      {tool.description && (
        <p className="mt-0.5 truncate text-xs text-gray-600">{tool.description}</p>
      )}
      {tool.job_types.length > 0 ? (
        <p className="mt-0.5 text-[11px] text-gray-500">
          offered to {tool.job_types.join(', ')} jobs
        </p>
      ) : (
        <p className="mt-0.5 text-[11px] text-amber-700">
          declares no job types, so no job will be offered it
        </p>
      )}
    </div>
    <div className="flex flex-shrink-0 items-center gap-1.5">
      <span
        className={`rounded px-1.5 py-0.5 text-[11px] ${
          EFFECT_STYLE[tool.effects] || 'bg-gray-200 text-gray-700'
        }`}
        title="Derived from the executor type, not from the manifest"
      >
        {tool.effects}
      </span>
      <span
        className="flex items-center gap-1 rounded bg-gray-200 px-1.5 py-0.5 text-[11px] text-gray-700"
        title={
          tool.network === 'external'
            ? 'Can reach off this host'
            : 'Cannot reach the network'
        }
      >
        {tool.network === 'external' ? (
          <Globe className="h-3 w-3" />
        ) : (
          <Lock className="h-3 w-3" />
        )}
        {tool.network}
      </span>
    </div>
  </div>
);

export const PluginsPanel: React.FC = () => {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [composing, setComposing] = useState(false);
  const [manifestText, setManifestText] = useState(EXAMPLE_MANIFEST);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiClient.listPlugins();
      setPlugins(data.items || []);
    } catch {
      // apiClient surfaces the error itself.
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const create = async () => {
    let manifest: Record<string, any>;
    try {
      manifest = JSON.parse(manifestText);
    } catch (err: any) {
      // Say where, not just that: a manifest is long enough that "invalid
      // JSON" is not actionable.
      toast.error(`Manifest is not valid JSON: ${err.message}`);
      return;
    }
    setSaving(true);
    try {
      await apiClient.createPlugin(manifest);
      toast.success('Plugin created');
      setComposing(false);
      await load();
    } catch {
      // The API refuses with the reason and apiClient shows it.
    } finally {
      setSaving(false);
    }
  };

  const install = async (plugin: Plugin) => {
    await apiClient.installPlugin(plugin.id, true);
    toast.success(`${plugin.name} installed`);
    await load();
  };

  const toggle = async (plugin: Plugin) => {
    await apiClient.setPluginEnabled(plugin.id, !plugin.enabled);
    await load();
  };

  const uninstall = async (plugin: Plugin) => {
    await apiClient.uninstallPlugin(plugin.id);
    toast.success(`${plugin.name} uninstalled`);
    await load();
  };

  const remove = async (plugin: Plugin) => {
    await apiClient.deletePlugin(plugin.id);
    toast.success(`${plugin.name} deleted`);
    await load();
  };

  return (
    <div className="mb-6 rounded-lg border border-gray-300 bg-gray-100 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Package className="h-5 w-5 text-primary-600" />
          <h2 className="text-base font-semibold text-gray-900">Plugins</h2>
          <span className="text-xs text-gray-500">
            {plugins.filter((p) => p.enabled).length} enabled of {plugins.length}
          </span>
        </div>
        <button
          onClick={() => setComposing((v) => !v)}
          className="inline-flex items-center gap-1.5 rounded bg-primary-600 px-3 py-1.5 text-sm text-gray-50 hover:bg-primary-700"
        >
          <Plus className="h-4 w-4" />
          New Plugin
        </button>
      </div>

      <p className="mt-1 text-xs text-gray-600">
        A plugin contributes tools an agent job can call by name. What a tool is
        allowed to do is derived from how it runs, not from what its manifest
        claims.
      </p>

      {composing && (
        <div className="mt-3 rounded border border-gray-300 bg-white p-3">
          <label className="block text-xs font-medium uppercase tracking-wide text-gray-500">
            Manifest
          </label>
          <textarea
            value={manifestText}
            onChange={(e) => setManifestText(e.target.value)}
            rows={16}
            spellCheck={false}
            className="mt-1 w-full rounded border border-gray-300 bg-white px-2 py-1.5 font-mono text-xs"
          />
          <div className="mt-2 flex justify-end gap-2">
            <button
              onClick={() => setComposing(false)}
              className="rounded px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={create}
              disabled={saving}
              className="rounded bg-primary-600 px-3 py-1.5 text-sm text-gray-50 hover:bg-primary-700 disabled:opacity-50"
            >
              {saving ? 'Creating…' : 'Create'}
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <p className="mt-3 text-sm text-gray-500">Loading…</p>
      ) : plugins.length === 0 ? (
        <p className="mt-3 text-sm text-gray-500">
          No plugins yet. A plugin bundles tools so they can be installed,
          enabled and disabled as one thing.
        </p>
      ) : (
        <div className="mt-3 space-y-2">
          {plugins.map((plugin) => {
            const open = Boolean(expanded[plugin.id]);
            return (
              <div
                key={plugin.id}
                className="rounded border border-gray-300 bg-white"
              >
                <div className="flex items-center gap-2 px-3 py-2">
                  <button
                    onClick={() =>
                      setExpanded({ ...expanded, [plugin.id]: !open })
                    }
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  >
                    {open ? (
                      <ChevronDown className="h-4 w-4 flex-shrink-0 text-gray-500" />
                    ) : (
                      <ChevronRight className="h-4 w-4 flex-shrink-0 text-gray-500" />
                    )}
                    <span className="truncate text-sm font-medium text-gray-900">
                      {plugin.name}
                    </span>
                    <span className="flex-shrink-0 text-xs text-gray-500">
                      v{plugin.version}
                    </span>
                    {plugin.source === 'builtin' && (
                      <span className="flex-shrink-0 rounded bg-gray-200 px-1.5 py-0.5 text-[11px] text-gray-700">
                        built in
                      </span>
                    )}
                    <span className="flex-shrink-0 text-xs text-gray-500">
                      {plugin.tools.length} tool
                      {plugin.tools.length === 1 ? '' : 's'}
                    </span>
                  </button>

                  {plugin.installed ? (
                    <>
                      <button
                        onClick={() => toggle(plugin)}
                        title={plugin.enabled ? 'Disable' : 'Enable'}
                        className={`flex items-center gap-1 rounded px-2 py-1 text-xs ${
                          plugin.enabled
                            ? 'bg-primary-600 text-gray-50 hover:bg-primary-700'
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        <Power className="h-3 w-3" />
                        {plugin.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                      <button
                        onClick={() => uninstall(plugin)}
                        title="Uninstall (discards settings)"
                        className="rounded p-1 text-gray-500 hover:bg-gray-200 hover:text-gray-900"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => install(plugin)}
                      className="rounded bg-primary-600 px-2 py-1 text-xs text-gray-50 hover:bg-primary-700"
                    >
                      Install
                    </button>
                  )}

                  {plugin.source === 'user' && (
                    <button
                      onClick={() => remove(plugin)}
                      title="Delete this plugin"
                      className="rounded p-1 text-gray-500 hover:bg-gray-200 hover:text-red-600"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>

                {open && (
                  <div className="space-y-1.5 border-t border-gray-200 px-3 py-2">
                    {plugin.description && (
                      <p className="text-xs text-gray-600">{plugin.description}</p>
                    )}
                    {plugin.tools.map((tool) => (
                      <ToolRow key={tool.name} tool={tool} />
                    ))}
                    {plugin.unavailable.length > 0 && (
                      <div className="rounded border border-amber-300 bg-amber-50 px-2 py-1.5">
                        {plugin.unavailable.map((u) => (
                          <p
                            key={u.tool}
                            className="flex items-start gap-1.5 text-[11px] text-amber-800"
                          >
                            <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
                            <span>
                              <code>{u.tool}</code> cannot be offered: {u.reason}
                            </span>
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default PluginsPanel;
