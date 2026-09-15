/**
 * Icon names a plugin may use, resolved to components.
 *
 * A plugin cannot ship a component, so it names an icon and this map decides
 * what that means. The list mirrors `ICONS` in
 * `backend/app/services/plugin_ui.py`, which refuses an unknown name at
 * install — so an entry reaching here always resolves, and the fallback is
 * there for a manifest edited in the database rather than for normal use.
 */

import {
  Activity,
  Beaker,
  Bot,
  Box,
  BarChart3,
  Check,
  Clock,
  Database,
  FileText,
  FlaskConical,
  Gauge,
  List,
  Package,
  Search,
  Sparkles,
  Table,
  Terminal,
  Zap,
} from 'lucide-react';

export const PLUGIN_ICONS: Record<
  string,
  React.ComponentType<{ className?: string }>
> = {
  activity: Activity,
  beaker: Beaker,
  bot: Bot,
  box: Box,
  chart: BarChart3,
  check: Check,
  clock: Clock,
  database: Database,
  file: FileText,
  flask: FlaskConical,
  gauge: Gauge,
  list: List,
  package: Package,
  search: Search,
  sparkles: Sparkles,
  table: Table,
  terminal: Terminal,
  zap: Zap,
};

export function pluginIcon(name: string) {
  return PLUGIN_ICONS[String(name || '').toLowerCase()] || Package;
}
