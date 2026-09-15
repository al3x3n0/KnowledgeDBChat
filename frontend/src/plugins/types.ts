/**
 * The shapes a plugin may contribute to the interface.
 *
 * These mirror `backend/app/services/plugin_ui.py`, which is the authority:
 * every field here was validated at install, so the renderer can trust the
 * *shape*. What it must never trust is the *content* — every string was
 * written by whoever authored the plugin and is rendered as text, never as
 * markup.
 */

export type PluginViewKind = 'table' | 'detail' | 'stats' | 'markdown';

export interface PluginViewColumn {
  key: string;
  label: string;
}

export interface PluginViewSource {
  tool: string;
  params: Record<string, unknown>;
}

export interface PluginViewSpec {
  kind: PluginViewKind;
  title: string;
  description: string;
  /** Null for a static view, which has nothing to run. */
  source: PluginViewSource | null;
  /** Dotted path into the tool's output where the data lives. */
  path: string;
  empty: string;
  columns?: PluginViewColumn[];
  text?: string;
}

export interface PluginNavEntry {
  door: string;
  name: string;
  view: string;
  icon: string;
}

export interface PluginPanel {
  slot: string;
  view: string;
  title: string;
}

export interface PluginUiContribution {
  plugin_id: string;
  slug: string;
  name: string;
  nav: PluginNavEntry[];
  views: Record<string, PluginViewSpec>;
  panels: PluginPanel[];
}

export interface PluginUiResponse {
  plugins: PluginUiContribution[];
}

/** The slots a plugin may add a panel to. Mirrors PANEL_SLOTS on the backend. */
export const PANEL_SLOTS = ['job.detail', 'runs.header'] as const;

export type PanelSlot = (typeof PANEL_SLOTS)[number];
