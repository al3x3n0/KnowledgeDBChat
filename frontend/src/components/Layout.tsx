/**
 * Main layout component with sidebar navigation
 */

import React, { useEffect, useMemo, useState } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import {
  MessageCircle,
  FileText,
  Brain,
  Settings,
  Menu,
  X,
  LogOut,
  Shield,
  User,
  Database,
  FileCheck,
  Workflow,
  Wrench,
  Presentation,
  Network,
  Search,
  Bot,
  BookOpen,
  ListChecks,
  BarChart3,
  Activity,
  FlaskConical,
  FolderGit2,
  Server,
  Key,
  Zap,
  Layers,
  Cpu,
  StickyNote,
  Sigma,
  GitPullRequest,
  ClipboardCheck,
  ChevronDown,
  ChevronRight,
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

interface NavGroup {
  id: string;
  name: string;
  defaultCollapsed?: boolean;
  items: NavItem[];
}

const SIDEBAR_GROUPS_STORAGE_KEY = 'kdb.sidebar.groups.v1';

const Layout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [navFilter, setNavFilter] = useState('');
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  // Global keyboard shortcuts for navigation
  useKeyboardShortcuts([
    {
      key: '1',
      ctrlKey: true,
      handler: () => navigate('/chat'),
      description: 'Navigate to Chat',
    },
    {
      key: '2',
      ctrlKey: true,
      handler: () => navigate('/documents'),
      description: 'Navigate to Documents',
    },
    {
      key: '3',
      ctrlKey: true,
      handler: () => navigate('/memory'),
      description: 'Navigate to Memory',
    },
    {
      key: ',',
      ctrlKey: true,
      handler: () => navigate('/settings'),
      description: 'Navigate to Settings',
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

  const navGroups: NavGroup[] = useMemo(() => {
    const isAdmin = user?.role === 'admin';
    const showLatex = Boolean(latexStatus?.enabled) || isAdmin;

    const groups: NavGroup[] = [
      {
        id: 'chat',
        name: 'Chat',
        defaultCollapsed: false,
        items: [{ name: 'Chat', to: '/chat', icon: MessageCircle }],
      },
      {
        id: 'knowledge',
        name: 'Knowledge',
        defaultCollapsed: false,
        items: [
          { name: 'Documents', to: '/documents', icon: FileText },
          { name: 'Search', to: '/search', icon: Search },
          { name: 'Knowledge Graph', to: '/kg/global', icon: Network },
          { name: 'Memory', to: '/memory', icon: Brain },
          { name: 'Templates', to: '/templates', icon: FileCheck },
        ],
      },
      {
        id: 'agents',
        name: 'Agents',
        defaultCollapsed: false,
        items: [
          { name: 'Autonomous Jobs', to: '/autonomous-agents', icon: Zap },
          { name: 'Workflows', to: '/workflows', icon: Workflow },
          { name: 'Agent Builder', to: '/agent-builder', icon: Bot },
          { name: 'Tools', to: '/tools', icon: Wrench },
        ],
      },
      {
        id: 'research',
        name: 'Research',
        defaultCollapsed: false,
        items: [
          { name: 'Papers', to: '/papers', icon: BookOpen },
          { name: 'Reading Lists', to: '/reading-lists', icon: ListChecks },
          { name: 'Research Notes', to: '/research-notes', icon: StickyNote },
        ],
      },
      {
        id: 'artifacts',
        name: 'Artifacts',
        defaultCollapsed: true,
        items: [
          ...(showLatex ? [{ name: 'LaTeX Studio', to: '/latex', icon: Sigma } as NavItem] : []),
          { name: 'Presentations', to: '/presentations', icon: Presentation },
          { name: 'Repo Reports', to: '/repo-reports', icon: FolderGit2 },
          { name: 'Synthesis', to: '/synthesis', icon: Layers },
          { name: 'Draft Reviews', to: '/artifact-drafts', icon: ClipboardCheck },
          { name: 'Patch PRs', to: '/patch-prs', icon: GitPullRequest },
        ],
      },
      {
        id: 'training',
        name: 'Training',
        defaultCollapsed: true,
        items: [
          { name: 'AI Hub', to: '/ai-hub', icon: Cpu },
          ...(isAdmin
            ? [
                { name: 'Usage', to: '/usage', icon: BarChart3 },
                { name: 'Routing Observability', to: '/usage/routing', icon: Activity },
                { name: 'Routing Experiments', to: '/usage/experiments', icon: FlaskConical },
              ]
            : []),
        ],
      },
      {
        id: 'admin',
        name: 'Admin',
        defaultCollapsed: true,
        items: [
          { name: 'API Keys', to: '/api-keys', icon: Key },
          { name: 'MCP Config', to: '/mcp-config', icon: Server },
          ...(isAdmin
            ? [
                { name: 'Admin', to: { pathname: '/admin', search: '?tab=overview' }, icon: Shield },
                { name: 'Agents', to: { pathname: '/admin', search: '?tab=agents' }, icon: Bot },
                { name: 'KG Admin', to: '/admin/kg', icon: Database },
                { name: 'KG Audit', to: '/admin/kg/audit', icon: Database },
              ]
            : []),
          { name: 'Settings', to: '/settings', icon: Settings },
        ],
      },
    ];

    return groups.filter((g) => g.items.length > 0);
  }, [latexStatus?.enabled, user?.role]);

  const allNavItems = useMemo(() => navGroups.flatMap((g) => g.items), [navGroups]);

  const activeNavItem = useMemo(
    () => allNavItems.find(isActiveNavItem) || null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allNavItems, location.pathname, location.search]
  );

  const activeGroupId = useMemo(() => {
    if (!activeNavItem) return null;
    const group = navGroups.find((g) => g.items.some((it) => it.name === activeNavItem.name));
    return group?.id || null;
  }, [activeNavItem, navGroups]);

  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>(() => {
    try {
      const raw = localStorage.getItem(SIDEBAR_GROUPS_STORAGE_KEY);
      if (raw) return JSON.parse(raw) as Record<string, boolean>;
    } catch {
      // ignore
    }
    const defaults: Record<string, boolean> = {};
    for (const g of navGroups) defaults[g.id] = Boolean(g.defaultCollapsed);
    if (activeGroupId) defaults[activeGroupId] = false;
    return defaults;
  });

  const toggleGroup = (groupId: string) => {
    setCollapsedGroups((prev) => {
      const next = { ...prev, [groupId]: !Boolean(prev[groupId]) };
      try {
        localStorage.setItem(SIDEBAR_GROUPS_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  };

  useEffect(() => {
    if (!activeGroupId) return;
    setCollapsedGroups((prev) => {
      if (prev[activeGroupId] === false) return prev;
      const next = { ...prev, [activeGroupId]: false };
      try {
        localStorage.setItem(SIDEBAR_GROUPS_STORAGE_KEY, JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, [activeGroupId]);

  const visibleGroups = useMemo(() => {
    const q = navFilter.trim().toLowerCase();
    if (!q) return navGroups;
    return navGroups
      .map((g) => ({
        ...g,
        items: g.items.filter((it) => it.name.toLowerCase().includes(q)),
      }))
      .filter((g) => g.items.length > 0);
  }, [navFilter, navGroups]);

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
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75" />
        </div>
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 flex flex-col w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out z-50
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0 md:static md:inset-0
      `}>
        {/* Sidebar header */}
        <div className="flex items-center justify-between h-16 px-4 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <Database className="w-8 h-8 text-primary-700" />
            <span className="text-primary-700 font-bold text-lg">Knowledge DB</span>
          </div>
          <button
            className="md:hidden text-gray-700 hover:text-gray-900"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-4 space-y-4 overflow-y-auto">
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                className="px-2 py-2 text-xs font-medium rounded-md bg-gray-100 text-gray-700 hover:bg-gray-200"
                onClick={() => handleQuickNav('/chat')}
              >
                New Chat
              </button>
              <button
                type="button"
                className="px-2 py-2 text-xs font-medium rounded-md bg-gray-100 text-gray-700 hover:bg-gray-200"
                onClick={() => handleQuickNav('/documents')}
              >
                Ingest
              </button>
              <button
                type="button"
                className="px-2 py-2 text-xs font-medium rounded-md bg-gray-100 text-gray-700 hover:bg-gray-200"
                onClick={() => handleQuickNav('/autonomous-agents')}
              >
                Run Job
              </button>
            </div>

            <div className="flex items-center gap-2">
              <input
                value={navFilter}
                onChange={(e) => setNavFilter(e.target.value)}
                placeholder="Filter navigation…"
                className="w-full px-3 py-2 text-sm rounded-md border border-gray-200 focus:outline-none focus:ring-2 focus:ring-primary-200 focus:border-primary-300"
              />
              {navFilter.trim() && (
                <button
                  type="button"
                  className="text-xs px-2 py-2 rounded-md border border-gray-200 text-gray-600 hover:bg-gray-50"
                  onClick={() => setNavFilter('')}
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {visibleGroups.map((group) => {
            const isCollapsed =
              navFilter.trim().length > 0
                ? false
                : collapsedGroups[group.id] !== undefined
                  ? Boolean(collapsedGroups[group.id])
                  : Boolean(group.defaultCollapsed);
            const hasActiveItem = group.items.some(isActiveNavItem);
            const GroupChevron = isCollapsed ? ChevronRight : ChevronDown;

            return (
              <div key={group.id}>
                <button
                  type="button"
                  className={`
                    w-full flex items-center justify-between px-2 py-1 text-xs font-semibold tracking-wide uppercase
                    ${hasActiveItem ? 'text-primary-700' : 'text-gray-500 hover:text-gray-700'}
                  `}
                  onClick={() => toggleGroup(group.id)}
                >
                  <span className="flex items-center gap-2">
                    <GroupChevron className="w-4 h-4" />
                    {group.name}
                  </span>
                  <span className="text-[10px] font-normal opacity-70">{group.items.length}</span>
                </button>

                {!isCollapsed && (
                  <div className="mt-2 space-y-1">
                    {group.items.map((item) => {
                      const isActive = isActiveNavItem(item);
                      const Icon = item.icon;

                      return (
                        <Link
                          key={`${group.id}:${item.name}`}
                          to={item.to}
                          className={`
                            flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200
                            ${
                              isActive
                                ? 'bg-primary-100 text-primary-700'
                                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                            }
                          `}
                          onClick={() => setSidebarOpen(false)}
                        >
                          <Icon className="w-5 h-5 mr-3" />
                          {item.name}
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* User section */}
        <div className="border-t border-gray-200 p-4">
          <div className="flex items-center space-x-3 mb-3">
            <div className="flex-shrink-0">
              {user?.avatar_url ? (
                <img 
                  className="w-8 h-8 rounded-full" 
                  src={user.avatar_url} 
                  alt={user.username}
                />
              ) : (
                <div className="w-8 h-8 bg-gray-100 border border-gray-200 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-primary-700" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {user?.full_name || user?.username}
              </p>
              <p className="text-xs text-gray-500 truncate">
                {user?.role}
              </p>
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

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top navigation */}
        <header className="bg-white shadow-sm border-b border-gray-200 h-16 flex items-center justify-between px-4 md:px-6">
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
        <main className="flex-1 overflow-auto">
          {degradedBanner}
          <Outlet />
        </main>
      </div>

      {/* Agent Widget */}
      <AgentWidget />
    </div>
  );
};

export default Layout;
