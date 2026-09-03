/**
 * Main layout component with sidebar navigation
 */

import React, { useMemo, useState } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import {
  MessageCircle, FileText, Brain, Settings, Menu, X, LogOut, Shield, User,
  Database, FileCheck, Workflow, Wrench, Presentation, Network, Search,
  Bot, BookOpen, ListChecks, BarChart3, Activity, FlaskConical, FolderGit2,
  Server, Key, Zap, Layers, Cpu, StickyNote, Sigma, GitPullRequest,
  ClipboardCheck
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import Button from './common/Button';
import AgentWidget from './agent/AgentWidget';
import NotificationBell from './notifications/NotificationBell';
import { useQuery } from 'react-query';
import { apiClient } from '../services/api';
import type { LatexStatusResponse, SystemHealth } from '../types';

type NavTo = string | { pathname: string; search?: string };
interface NavItem {
  name: string;
  to: NavTo;
  icon: React.ComponentType<{ className?: string }>;
}

/**
 * A door is one of the four things this application is for. Its sections are
 * the routes inside it.
 *
 * The nav used to be eight groups of subsystems -- 31 destinations for an
 * admin, 23 for everyone else -- which asked you to know which service owned
 * a thing before you could find it. These four are named for what you are
 * doing: ask the corpus, draw on it, run the work, write it up. Settings is
 * the fifth door and holds the twelve destinations that are configuration or
 * observability; ten of those twelve were already invisible to a non-admin,
 * which is the tell that they were never daily work.
 *
 * Every route is exactly where it was. Moving one between doors is a one-line
 * edit to the array below.
 */
interface NavDoor {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  /** Rendered at the bottom, away from the work. */
  utility?: boolean;
  sections: NavItem[];
}

const Layout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [navFilter, setNavFilter] = useState('');
  const navFilterRef = React.useRef<HTMLInputElement | null>(null);
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  // Global keyboard shortcuts for navigation
  // One shortcut per door, in the order the doors are drawn.
  //
  // These predated the four-door navigation and never caught up: Ctrl+3 went
  // to Memory — a page that now lives under Settings — while R&D and
  // Synthesis, two of the four doors and the bulk of the actual work here,
  // had no shortcut at all. Ctrl+K focuses the cross-door filter, which is the
  // fastest way to anything that is not a door.
  useKeyboardShortcuts([
    {
      key: '1',
      ctrlKey: true,
      handler: () => navigate('/chat'),
      description: 'Go to Chat',
    },
    {
      key: '2',
      ctrlKey: true,
      handler: () => navigate('/documents'),
      description: 'Go to Library',
    },
    {
      key: '3',
      ctrlKey: true,
      handler: () => navigate('/autonomous-agents'),
      description: 'Go to R&D',
    },
    {
      key: '4',
      ctrlKey: true,
      handler: () => navigate('/synthesis'),
      description: 'Go to Synthesis',
    },
    {
      key: 'k',
      ctrlKey: true,
      preventDefault: true,
      handler: () => {
        setSidebarOpen(true);
        // The rail may have just been revealed, so focus after it paints.
        requestAnimationFrame(() => navFilterRef.current?.focus());
      },
      description: 'Search every destination',
    },
    {
      key: ',',
      ctrlKey: true,
      handler: () => navigate('/settings'),
      description: 'Go to Settings',
    },
  ]);

  const isActiveNavItem = (item: NavItem) => {
    const pathname = typeof item.to === 'string' ? item.to : item.to.pathname;
    if (!location.pathname.startsWith(pathname)) return false;

    if (typeof item.to === 'string') return true;

    const desiredTab = item.to.search ? new URLSearchParams(item.to.search).get('tab') : null;
    if (!desiredTab) return true;
    const currentTab = new URLSearchParams(location.search).get('tab') || 'overview';
    return desiredTab === currentTab;
  };

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const { data: systemHealth } = useQuery<SystemHealth>(
    ['system-health-status'],
    () => apiClient.getSystemHealthStatus(),
    {
      enabled: !!user,
      refetchInterval: 15000,
      retry: 1,
    }
  );

  const { data: latexStatus } = useQuery<LatexStatusResponse>(
    ['latex-status'],
    () => apiClient.getLatexStatus(),
    {
      enabled: !!user,
      refetchInterval: 60000,
      retry: 1,
    }
  );

  const navDoors: NavDoor[] = useMemo(() => {
    const isAdmin = user?.role === 'admin';
    const showLatex = Boolean(latexStatus?.enabled) || isAdmin;

    const doors: NavDoor[] = [
      {
        id: 'chat',
        name: 'Chat',
        icon: MessageCircle,
        sections: [
          { name: 'Chat', to: '/chat', icon: MessageCircle },
          { name: 'Search', to: '/search', icon: Search },
        ],
      },
      {
        id: 'library',
        name: 'Library',
        icon: BookOpen,
        sections: [
          { name: 'Documents', to: '/documents', icon: FileText },
          { name: 'Papers', to: '/papers', icon: BookOpen },
          { name: 'Reading Lists', to: '/reading-lists', icon: ListChecks },
          { name: 'Research Notes', to: '/research-notes', icon: StickyNote },
          { name: 'Memory', to: '/memory', icon: Brain },
          { name: 'Knowledge Graph', to: '/kg/global', icon: Network },
          { name: 'Templates', to: '/templates', icon: FileCheck },
        ],
      },
      {
        // Where the work runs. Workflows belong here rather than in a
        // department of their own: a workflow is a run written down in
        // advance.
        id: 'rnd',
        name: 'R&D',
        icon: Zap,
        sections: [
          { name: 'Runs', to: '/autonomous-agents', icon: Zap },
          { name: 'Control Plane', to: '/agent-control-plane', icon: Activity },
          { name: 'Workflows', to: '/workflows', icon: Workflow },
          { name: 'Agents', to: '/agent-builder', icon: Bot },
        ],
      },
      {
        id: 'synthesis',
        name: 'Synthesis',
        icon: Layers,
        sections: [
          { name: 'Synthesis', to: '/synthesis', icon: Layers },
          { name: 'Presentations', to: '/presentations', icon: Presentation },
          ...(showLatex ? [{ name: 'LaTeX Studio', to: '/latex', icon: Sigma } as NavItem] : []),
          { name: 'Repo Reports', to: '/repo-reports', icon: FolderGit2 },
          { name: 'Draft Reviews', to: '/artifact-drafts', icon: ClipboardCheck },
          { name: 'Patch PRs', to: '/patch-prs', icon: GitPullRequest },
        ],
      },
      {
        id: 'settings',
        name: 'Settings',
        icon: Settings,
        utility: true,
        sections: [
          { name: 'Tools', to: '/tools', icon: Wrench },
          { name: 'AI Hub', to: '/ai-hub', icon: Cpu },
          { name: 'API Keys', to: '/api-keys', icon: Key },
          { name: 'MCP Config', to: '/mcp-config', icon: Server },
          ...(isAdmin
            ? [
                { name: 'Usage', to: '/usage', icon: BarChart3 },
                { name: 'Routing Observability', to: '/usage/routing', icon: Activity },
                { name: 'Routing Experiments', to: '/usage/experiments', icon: FlaskConical },
                { name: 'Admin', to: { pathname: '/admin', search: '?tab=overview' }, icon: Shield },
                { name: 'Admin Agents', to: { pathname: '/admin', search: '?tab=agents' }, icon: Bot },
                { name: 'KG Admin', to: '/admin/kg', icon: Database },
                { name: 'KG Audit', to: '/admin/kg/audit', icon: Database },
              ]
            : []),
          { name: 'Preferences', to: '/settings', icon: Settings },
        ] as NavItem[],
      },
    ];

    return doors.filter((d) => d.sections.length > 0);
  }, [latexStatus?.enabled, user?.role]);

  const allNavItems = useMemo(() => navDoors.flatMap((d) => d.sections), [navDoors]);

  const activeNavItem = useMemo(
    () => allNavItems.find(isActiveNavItem) || null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allNavItems, location.pathname, location.search]
  );

  const activeDoor = useMemo(() => {
    if (!activeNavItem) return navDoors[0] || null;
    return (
      navDoors.find((d) => d.sections.some((it) => it.name === activeNavItem.name)) ||
      navDoors[0] ||
      null
    );
  }, [activeNavItem, navDoors]);

  /**
   * The filter searches every section across every door, so a destination
   * stays reachable by name without knowing which door holds it -- the one
   * thing the old flat list was good at.
   */
  const filterResults = useMemo(() => {
    const q = navFilter.trim().toLowerCase();
    if (!q) return null;
    return navDoors
      .map((d) => ({ door: d, matches: d.sections.filter((it) => it.name.toLowerCase().includes(q)) }))
      .filter((r) => r.matches.length > 0);
  }, [navFilter, navDoors]);

  const handleQuickNav = (to: string) => {
    navigate(to);
    setSidebarOpen(false);
  };

  const degradedBanner = useMemo(() => {
    if (!systemHealth) return null;
    if (systemHealth.overall_status === 'healthy') return null;

    const unhealthy = Object.entries(systemHealth.services)
      .filter(([, s]) => s.status && s.status !== 'healthy')
      .map(([name, s]) => `${name}${s.error ? `: ${s.error}` : s.message ? `: ${s.message}` : ''}`);

    const title =
      systemHealth.overall_status === 'unhealthy' ? 'System degraded' : 'Limited functionality';
    const bg = systemHealth.overall_status === 'unhealthy' ? 'bg-red-50 border-red-200 text-red-900' : 'bg-yellow-50 border-yellow-200 text-yellow-900';

    return (
      <div className={`border-b px-4 py-2 text-sm ${bg}`}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="font-medium">{title}</div>
            {unhealthy.length > 0 && (
              <div className="text-xs mt-0.5 opacity-90">{unhealthy.join(' • ')}</div>
            )}
          </div>
          {user?.role === 'admin' && (
            <Link className="text-xs underline whitespace-nowrap" to={{ pathname: '/admin', search: '?tab=overview' }}>
              View system health
            </Link>
          )}
        </div>
      </div>
    );
  }, [systemHealth, user]);

  return (
    <div className="h-screen flex overflow-hidden bg-gray-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 flex z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        >
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm animate-fade-in" />
        </div>
      )}

      {/* Sidebar: doors on the left, the active door's sections beside them */}
      <div className={`
        fixed inset-y-0 left-0 flex w-[344px] bg-gray-50 shadow-level-3 md:shadow-none transform transition-transform duration-slow ease-enter z-50
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0 md:static md:inset-0
      `}>

        {/* Doors. The deepest plane in the app: everything else sits above it. */}
        <div className="w-[184px] flex-shrink-0 flex flex-col border-r border-gray-200 bg-gray-50">
          <div className="flex items-center h-16 px-4 border-b border-gray-200">
            <div className="flex items-center space-x-2 min-w-0 group">
              <Database className="w-6 h-6 text-primary-700 flex-shrink-0 transition-transform duration-base ease-ui group-hover:scale-110" />
              <span className="text-gray-900 font-semibold text-sm truncate tracking-tight">Knowledge DB</span>
            </div>
          </div>

          <nav className="flex-1 px-3 py-3 space-y-1 overflow-y-auto">
            {navDoors.filter((d) => !d.utility).map((door) => {
              const DoorIcon = door.icon;
              const isActive = activeDoor?.id === door.id;
              return (
                <Link
                  key={door.id}
                  to={door.sections[0].to}
                  onClick={() => setSidebarOpen(false)}
                  className={`
                    group relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
                    transition-all duration-fast ease-ui
                    ${isActive
                      ? 'bg-primary-500/10 text-primary-700 font-medium shadow-[inset_2px_0_0_0_theme(colors.primary.600)]'
                      : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900'}
                  `}
                >
                  <DoorIcon
                    className={`w-5 h-5 flex-shrink-0 transition-transform duration-fast ease-ui
                      ${isActive ? '' : 'group-hover:scale-110'}`}
                  />
                  <span className="truncate">{door.name}</span>
                </Link>
              );
            })}
          </nav>

          <div className="px-3 pb-2 space-y-1">
            {navDoors.filter((d) => d.utility).map((door) => {
              const DoorIcon = door.icon;
              const isActive = activeDoor?.id === door.id;
              return (
                <Link
                  key={door.id}
                  to={door.sections[0].to}
                  onClick={() => setSidebarOpen(false)}
                  className={`
                    group relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm
                    transition-all duration-fast ease-ui
                    ${isActive
                      ? 'bg-primary-500/10 text-primary-700 font-medium shadow-[inset_2px_0_0_0_theme(colors.primary.600)]'
                      : 'text-gray-500 hover:bg-gray-200 hover:text-gray-900'}
                  `}
                >
                  <DoorIcon
                    className={`w-5 h-5 flex-shrink-0 transition-transform duration-fast ease-ui
                      ${isActive ? '' : 'group-hover:scale-110'}`}
                  />
                  <span className="truncate">{door.name}</span>
                </Link>
              );
            })}
          </div>

          <div className="border-t border-gray-200 p-3">
            <div className="flex items-center gap-2 mb-2 min-w-0">
              <div className="flex-shrink-0">
                {user?.avatar_url ? (
                  <img className="w-7 h-7 rounded-full" src={user.avatar_url} alt={user.username} />
                ) : (
                  <div className="w-7 h-7 bg-gray-100 border border-gray-200 rounded-full flex items-center justify-center">
                    <User className="w-3.5 h-3.5 text-primary-700" />
                  </div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-gray-900 truncate">
                  {user?.full_name || user?.username}
                </p>
                <p className="text-[10px] text-gray-500 truncate">{user?.role}</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              fullWidth
              icon={<LogOut className="w-4 h-4" />}
              onClick={handleLogout}
            >
              Logout
            </Button>
          </div>
        </div>

        {/* Sections of the active door, or filter results across all of them.
            One plane above the door rail, so the two read as depth rather
            than as a single panel split by a line. */}
        <div className="flex-1 min-w-0 flex flex-col bg-gray-100">
          <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
            <span className="text-xs font-semibold tracking-wide uppercase text-gray-500 truncate">
              {filterResults ? 'Results' : activeDoor?.name}
            </span>
            <button
              className="md:hidden text-gray-700 hover:text-gray-900"
              onClick={() => setSidebarOpen(false)}
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="px-3 pt-3">
            <input
              ref={navFilterRef}
              value={navFilter}
              onChange={(e) => setNavFilter(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setNavFilter('');
                  e.currentTarget.blur();
                }
              }}
              aria-label="Search every destination"
              placeholder="Filter…"
              className="w-full px-3 py-2 text-sm rounded-md bg-gray-50 border border-gray-300
                shadow-[inset_0_1px_2px_0_rgb(0_0_0_/_0.35)]
                transition-all duration-fast ease-ui
                hover:border-gray-400
                focus:outline-none focus:ring-0 focus:border-primary-600 focus:shadow-accent-glow"
            />
          </div>

          <nav className="flex-1 px-3 py-3 overflow-y-auto">
            {filterResults ? (
              <div className="space-y-4">
                {filterResults.map(({ door, matches }) => (
                  <div key={door.id} className="space-y-1">
                    <div className="px-2 text-[10px] font-semibold tracking-wide uppercase text-gray-500">
                      {door.name}
                    </div>
                    {matches.map((item) => {
                      const Icon = item.icon;
                      return (
                        <Link
                          key={`${door.id}:${item.name}`}
                          to={item.to}
                          onClick={() => { setNavFilter(''); setSidebarOpen(false); }}
                          className={`
                            group flex items-center gap-3 px-3 py-2 text-sm rounded-md
                            transition-all duration-fast ease-ui
                            ${isActiveNavItem(item)
                              ? 'bg-primary-500/10 text-primary-700 font-medium shadow-[inset_2px_0_0_0_theme(colors.primary.600)]'
                              : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900 hover:translate-x-0.5'}
                          `}
                        >
                          <Icon className="w-4 h-4 flex-shrink-0" />
                          <span className="truncate">{item.name}</span>
                        </Link>
                      );
                    })}
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-1">
                {(activeDoor?.sections || []).map((item) => {
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.name}
                      to={item.to}
                      onClick={() => setSidebarOpen(false)}
                      className={`
                        group flex items-center gap-3 px-3 py-2 text-sm rounded-md
                        transition-all duration-fast ease-ui
                        ${isActiveNavItem(item)
                          ? 'bg-primary-500/10 text-primary-700 font-medium shadow-[inset_2px_0_0_0_theme(colors.primary.600)]'
                          : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900 hover:translate-x-0.5'}
                      `}
                    >
                      <Icon className="w-4 h-4 flex-shrink-0" />
                      <span className="truncate">{item.name}</span>
                    </Link>
                  );
                })}
              </div>
            )}
          </nav>

          {/* The three things started most often, kept one click away. */}
          <div className="px-3 pb-3 grid grid-cols-3 gap-2 border-t border-gray-200 pt-3">
            <button
              type="button"
              className="px-2 py-2 text-xs font-medium rounded-md bg-gray-200 text-gray-700 border border-gray-300
                transition-all duration-fast ease-ui active:translate-y-px
                hover:bg-gray-300 hover:text-gray-900 hover:border-gray-400 hover:shadow-level-1"
              onClick={() => handleQuickNav('/chat')}
            >
              Ask
            </button>
            <button
              type="button"
              className="px-2 py-2 text-xs font-medium rounded-md bg-gray-200 text-gray-700 border border-gray-300
                transition-all duration-fast ease-ui active:translate-y-px
                hover:bg-gray-300 hover:text-gray-900 hover:border-gray-400 hover:shadow-level-1"
              onClick={() => handleQuickNav('/documents')}
            >
              Ingest
            </button>
            <button
              type="button"
              className="px-2 py-2 text-xs font-medium rounded-md bg-gray-200 text-gray-700 border border-gray-300
                transition-all duration-fast ease-ui active:translate-y-px
                hover:bg-gray-300 hover:text-gray-900 hover:border-gray-400 hover:shadow-level-1"
              onClick={() => handleQuickNav('/autonomous-agents')}
            >
              Run
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {/* Top navigation */}
        <header className="bg-gray-100 border-b border-gray-200 shadow-level-1 h-16 flex items-center justify-between px-4 md:px-6 relative z-10">
          <button
            className="md:hidden text-gray-500 hover:text-gray-700"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="w-6 h-6" />
          </button>
          
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-semibold text-gray-900">
              {activeNavItem?.name || 'Knowledge Database'}
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            <NotificationBell />
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 min-h-0 overflow-hidden flex flex-col">
          {degradedBanner}
          <div className="flex-1 h-full min-h-0 overflow-auto flex flex-col">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Agent Widget */}
      <AgentWidget />
    </div>
  );
};

export default Layout;
