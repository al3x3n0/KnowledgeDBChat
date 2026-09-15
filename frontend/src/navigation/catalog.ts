/**
 * Every place this application can take you, as data.
 *
 * This was a literal inside `Layout.tsx`, which meant the navigation was the
 * same for everybody and could only be changed by editing the component. It
 * is a catalog now for two reasons: a person can reorder, hide and rename
 * what it offers (see `preferences.ts`), and a later slice can let a plugin
 * add to it.
 *
 * A destination is identified by its **key**, which is its route — including
 * the query string where that is what distinguishes two entries, as it does
 * for the two Admin tabs. Preferences are stored against these keys rather
 * than against labels, so renaming a page in your own navigation does not
 * orphan the preference that hid it.
 *
 * Nothing here decides what a user may *see*: `visibility` states the
 * condition, and `buildCatalog` applies it. Keeping that a declared flag
 * rather than an inline ternary is what lets the settings UI explain why an
 * entry is absent instead of simply not listing it.
 */

import {
  Activity,
  BarChart3,
  Bot,
  BookOpen,
  Brain,
  ClipboardCheck,
  Cpu,
  Database,
  FileCheck,
  FileText,
  FlaskConical,
  FolderGit2,
  GitBranch,
  GitPullRequest,
  Key,
  Layers,
  ListChecks,
  MessageCircle,
  Network,
  Presentation,
  Search,
  Server,
  Settings,
  Shield,
  Sigma,
  StickyNote,
  Wrench,
  Workflow,
  Zap,
} from 'lucide-react';

export type NavTo = string | { pathname: string; search?: string };

/** Why an entry might not be offered to this user. */
export type NavVisibility = 'always' | 'admin' | 'latex';

export interface NavItem {
  /** Stable identity: the route, with its search string when it has one. */
  key: string;
  name: string;
  to: NavTo;
  icon: React.ComponentType<{ className?: string }>;
  visibility?: NavVisibility;
}

export interface NavDoor {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  /** Rendered at the bottom, away from the work. */
  utility?: boolean;
  sections: NavItem[];
}

/** The key a destination is stored under. */
export function navKey(to: NavTo): string {
  if (typeof to === 'string') return to;
  return `${to.pathname}${to.search || ''}`;
}

function item(
  name: string,
  to: NavTo,
  icon: NavItem['icon'],
  visibility: NavVisibility = 'always'
): NavItem {
  return { key: navKey(to), name, to, icon, visibility };
}

/**
 * A door is one of the four things this application is for.
 *
 * The nav used to be eight groups of subsystems -- 31 destinations for an
 * admin, 23 for everyone else -- which asked you to know which service owned
 * a thing before you could find it. These four are named for what you are
 * doing: ask the corpus, draw on it, run the work, write it up. Settings is
 * the fifth door and holds the destinations that are configuration or
 * observability.
 */
export const NAV_CATALOG: NavDoor[] = [
  {
    id: 'chat',
    name: 'Chat',
    icon: MessageCircle,
    sections: [
      item('Chat', '/chat', MessageCircle),
      item('Search', '/search', Search),
    ],
  },
  {
    id: 'library',
    name: 'Library',
    icon: BookOpen,
    sections: [
      item('Documents', '/documents', FileText),
      item('Papers', '/papers', BookOpen),
      item('Reading Lists', '/reading-lists', ListChecks),
      item('Research Notes', '/research-notes', StickyNote),
      item('Memory', '/memory', Brain),
      item('Knowledge Graph', '/kg/global', Network),
      item('Templates', '/templates', FileCheck),
    ],
  },
  {
    // Where the work runs. Workflows belong here rather than in a department
    // of their own: a workflow is a run written down in advance.
    id: 'rnd',
    name: 'R&D',
    icon: Zap,
    sections: [
      item('Runs', '/autonomous-agents', Zap),
      item('Control Plane', '/agent-control-plane', Activity),
      item('Pipelines', '/pipelines', GitBranch),
      item('Workflows', '/workflows', Workflow),
      item('Agents', '/agent-builder', Bot),
    ],
  },
  {
    id: 'synthesis',
    name: 'Synthesis',
    icon: Layers,
    sections: [
      item('Synthesis', '/synthesis', Layers),
      item('Presentations', '/presentations', Presentation),
      item('LaTeX Studio', '/latex', Sigma, 'latex'),
      item('Repo Reports', '/repo-reports', FolderGit2),
      item('Draft Reviews', '/artifact-drafts', ClipboardCheck),
      item('Patch PRs', '/patch-prs', GitPullRequest),
    ],
  },
  {
    id: 'settings',
    name: 'Settings',
    icon: Settings,
    utility: true,
    sections: [
      item('Tools', '/tools', Wrench),
      item('AI Hub', '/ai-hub', Cpu),
      item('API Keys', '/api-keys', Key),
      item('MCP Config', '/mcp-config', Server),
      item('Usage', '/usage', BarChart3, 'admin'),
      item('Routing Observability', '/usage/routing', Activity, 'admin'),
      item('Routing Experiments', '/usage/experiments', FlaskConical, 'admin'),
      item('Admin', { pathname: '/admin', search: '?tab=overview' }, Shield, 'admin'),
      item('Admin Agents', { pathname: '/admin', search: '?tab=agents' }, Bot, 'admin'),
      item('KG Admin', '/admin/kg', Database, 'admin'),
      item('KG Audit', '/admin/kg/audit', Database, 'admin'),
      item('Preferences', '/settings', Settings),
    ],
  },
];

export interface CatalogContext {
  isAdmin: boolean;
  latexEnabled: boolean;
}

/** Which entry is offered to this user, before their own preferences apply. */
export function isPermitted(item: NavItem, ctx: CatalogContext): boolean {
  switch (item.visibility) {
    case 'admin':
      return ctx.isAdmin;
    case 'latex':
      // An admin can always reach it, because they are the one who turns it on.
      return ctx.latexEnabled || ctx.isAdmin;
    default:
      return true;
  }
}

/** The catalog this user is permitted to see. Preferences are applied after. */
export function buildCatalog(ctx: CatalogContext): NavDoor[] {
  return NAV_CATALOG.map((door) => ({
    ...door,
    sections: door.sections.filter((s) => isPermitted(s, ctx)),
  })).filter((door) => door.sections.length > 0);
}
