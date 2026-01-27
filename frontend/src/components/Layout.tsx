/**
 * Main layout component with sidebar navigation
 */

import React, { useMemo, useState } from 'react';
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
  FolderGit2,
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import Button from './common/Button';
import AgentWidget from './agent/AgentWidget';
import NotificationBell from './notifications/NotificationBell';
import { useQuery } from 'react-query';
import { apiClient } from '../services/api';
import type { SystemHealth } from '../types';

type NavTo = string | { pathname: string; search?: string };
interface NavItem {
  name: string;
  to: NavTo;
  icon: React.ComponentType<{ className?: string }>;
}

const Layout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
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

  const navigation: NavItem[] = [
    { name: 'Chat', to: '/chat', icon: MessageCircle },
    { name: 'Search', to: '/search', icon: Search },
    { name: 'Papers', to: '/papers', icon: BookOpen },
    { name: 'Reading Lists', to: '/reading-lists', icon: ListChecks },
    { name: 'Documents', to: '/documents', icon: FileText },
    { name: 'Knowledge Graph', to: '/kg/global', icon: Network },
    { name: 'Templates', to: '/templates', icon: FileCheck },
    { name: 'Presentations', to: '/presentations', icon: Presentation },
    { name: 'Repo Reports', to: '/repo-reports', icon: FolderGit2 },
    { name: 'Workflows', to: '/workflows', icon: Workflow },
    { name: 'Tools', to: '/tools', icon: Wrench },
    { name: 'Memory', to: '/memory', icon: Brain },
    { name: 'Settings', to: '/settings', icon: Settings },
  ];

  // Add admin navigation for admin users
  if (user?.role === 'admin') {
    navigation.push({
      name: 'Usage',
      to: '/usage',
      icon: BarChart3,
    });
    navigation.push({
      name: 'Admin',
      to: { pathname: '/admin', search: '?tab=overview' },
      icon: Shield,
    });
    navigation.push({
      name: 'Agents',
      to: { pathname: '/admin', search: '?tab=agents' },
      icon: Bot,
    });
    navigation.push({
      name: 'KG Admin',
      to: '/admin/kg',
      icon: Database,
    });
    navigation.push({
      name: 'KG Audit',
      to: '/admin/kg/audit',
      icon: Database,
    });
  }

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
        <div className="flex items-center justify-between h-16 px-4 bg-primary-600">
          <div className="flex items-center space-x-2">
            <Database className="w-8 h-8 text-white" />
            <span className="text-white font-bold text-lg">Knowledge DB</span>
          </div>
          <button
            className="md:hidden text-white hover:text-gray-200"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-4 space-y-2 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = isActiveNavItem(item);
            const Icon = item.icon;
            
            return (
              <Link
                key={item.name}
                to={item.to}
                className={`
                  flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200
                  ${isActive 
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
                <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
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
              {navigation.find(isActiveNavItem)?.name || 'Knowledge Database'}
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
