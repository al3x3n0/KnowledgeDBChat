import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

import { AuthProvider, useAuth } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';
import Layout from './components/Layout';
import LandingRedirect from './navigation/LandingRedirect';
import PluginViewPage from './plugins/PluginViewPage';
import LoadingSpinner from './components/common/LoadingSpinner';
import ErrorBoundary from './components/common/ErrorBoundary';

// Lazy load pages for code splitting
const LoginPage = lazy(() => import('./pages/LoginPage'));
const ChatPage = lazy(() => import('./pages/ChatPage'));
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'));
const MemoryPage = lazy(() => import('./pages/MemoryPage'));
const CampaignsPage = lazy(() => import('./pages/CampaignsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const AdminPage = lazy(() => import('./pages/AdminPage'));
const KGAdminPage = lazy(() => import('./pages/KGAdminPage'));
const KGAuditPage = lazy(() => import('./pages/KGAuditPage'));
const DocumentGraphPage = lazy(() => import('./pages/DocumentGraphPage'));
const TemplateFillPage = lazy(() => import('./pages/TemplateFillPage'));
const WorkflowsPage = lazy(() => import('./pages/WorkflowsPage'));
const PipelineStudioPage = lazy(() => import('./pages/PipelineStudioPage'));
const WorkflowEditorPage = lazy(() => import('./pages/WorkflowEditorPage'));
const ToolsPage = lazy(() => import('./pages/ToolsPage'));
const PresentationsPage = lazy(() => import('./pages/PresentationsPage'));
const GlobalGraphPage = lazy(() => import('./pages/GlobalGraphPage'));
const SearchPage = lazy(() => import('./pages/SearchPage'));
const PapersPage = lazy(() => import('./pages/PapersPage'));
const LatexStudioPage = lazy(() => import('./pages/LatexStudioPage'));
const ReadingListsPage = lazy(() => import('./pages/ReadingListsPage'));
const ResearchInboxPage = lazy(() => import('./pages/ResearchInboxPage'));
const ReadingListDetailPage = lazy(() => import('./pages/ReadingListDetailPage'));
const ContextPackPage = lazy(() => import('./pages/ContextPackPage'));
const UsagePage = lazy(() => import('./pages/UsagePage'));
const RoutingObservabilityPage = lazy(() => import('./pages/RoutingObservabilityPage'));
const RoutingExperimentsPage = lazy(() => import('./pages/RoutingExperimentsPage'));
const APIKeysPage = lazy(() => import('./pages/APIKeysPage'));
const MCPConfigPage = lazy(() => import('./pages/MCPConfigPage'));
const AgentBuilderPage = lazy(() => import('./pages/AgentBuilderPage'));
const AgentControlPlanePage = lazy(() => import('./pages/AgentControlPlanePage'));
const RepoReportsPage = lazy(() => import('./pages/RepoReportsPage'));
const AutonomousAgentsPage = lazy(() => import('./pages/AutonomousAgentsPage'));
const PatchPRsPage = lazy(() => import('./pages/PatchPRsPage'));
const ArtifactDraftsPage = lazy(() => import('./pages/ArtifactDraftsPage'));
const SynthesisPage = lazy(() => import('./pages/SynthesisPage'));
const AIHubPage = lazy(() => import('./pages/AIHubPage'));
const ResearchNotesPage = lazy(() => import('./pages/ResearchNotesPage'));

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Protected Route component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Public Route component (redirect to chat if already authenticated)
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (user) {
    // Via "/" rather than straight to /chat, so signing in lands where this
    // account is configured to land. Hardcoding it here would make the
    // preference hold for every route into the app except the front door.
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

// Loading fallback component
const PageLoader: React.FC = () => (
  <div className="min-h-screen flex items-center justify-center">
    <LoadingSpinner size="lg" />
  </div>
);

const AppRoutes: React.FC = () => {
  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        {/* Public routes */}
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Suspense fallback={<PageLoader />}>
                <LoginPage />
              </Suspense>
            </PublicRoute>
          }
        />

        {/* Protected routes */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route index element={<LandingRedirect />} />
          {/* Every contributed page lives under this one route shape. A
              manifest cannot choose its own path, or an installed plugin could
              claim /documents and shadow a first-party page. */}
          <Route path="p/:slug/:viewId" element={<PluginViewPage />} />
          <Route 
            path="chat" 
            element={
              <Suspense fallback={<PageLoader />}>
                <ChatPage />
              </Suspense>
            } 
          />
          <Route 
            path="chat/:sessionId" 
            element={
              <Suspense fallback={<PageLoader />}>
                <ChatPage />
              </Suspense>
            } 
          />
          <Route
            path="documents"
            element={
              <Suspense fallback={<PageLoader />}>
                <DocumentsPage />
              </Suspense>
            }
          />
          <Route
            path="search"
            element={
              <Suspense fallback={<PageLoader />}>
                <SearchPage />
              </Suspense>
            }
          />
          <Route
            path="papers"
            element={
              <Suspense fallback={<PageLoader />}>
                <PapersPage />
              </Suspense>
            }
          />
          <Route
            path="context-packs/:traceId"
            element={
              <Suspense fallback={<PageLoader />}>
                <ContextPackPage />
              </Suspense>
            }
          />
          <Route
            path="latex"
            element={
              <Suspense fallback={<PageLoader />}>
                <LatexStudioPage />
              </Suspense>
            }
          />
          <Route
            path="research/inbox"
            element={
              <Suspense fallback={<PageLoader />}>
                <ResearchInboxPage />
              </Suspense>
            }
          />
          <Route
            path="reading-lists"
            element={
              <Suspense fallback={<PageLoader />}>
                <ReadingListsPage />
              </Suspense>
            }
          />
          <Route
            path="reading-lists/:id"
            element={
              <Suspense fallback={<PageLoader />}>
                <ReadingListDetailPage />
              </Suspense>
            }
          />
          <Route
            path="documents/:documentId/graph" 
            element={
              <Suspense fallback={<PageLoader />}>
                <DocumentGraphPage />
              </Suspense>
            } 
          />
          <Route 
            path="memory" 
            element={
              <Suspense fallback={<PageLoader />}>
                <MemoryPage />
              </Suspense>
            } 
          />
          <Route
            path="settings"
            element={
              <Suspense fallback={<PageLoader />}>
                <SettingsPage />
              </Suspense>
            }
          />
          <Route
            path="usage"
            element={
              <Suspense fallback={<PageLoader />}>
                <UsagePage />
              </Suspense>
            }
          />

          <Route
            path="usage/routing"
            element={
              <Suspense fallback={<PageLoader />}>
                <RoutingObservabilityPage />
              </Suspense>
            }
          />

          <Route
            path="usage/experiments"
            element={
              <Suspense fallback={<PageLoader />}>
                <RoutingExperimentsPage />
              </Suspense>
            }
          />

          <Route
            path="api-keys"
            element={
              <Suspense fallback={<PageLoader />}>
                <APIKeysPage />
              </Suspense>
            }
          />
          <Route
            path="mcp-config"
            element={
              <Suspense fallback={<PageLoader />}>
                <MCPConfigPage />
              </Suspense>
            }
          />
          <Route
            path="agent-builder"
            element={
              <Suspense fallback={<PageLoader />}>
                <AgentBuilderPage />
              </Suspense>
            }
          />
          <Route
            path="agent-control-plane"
            element={
              <Suspense fallback={<PageLoader />}>
                <AgentControlPlanePage />
              </Suspense>
            }
          />
          <Route
            path="templates"
            element={
              <Suspense fallback={<PageLoader />}>
                <TemplateFillPage />
              </Suspense>
            }
          />
          <Route
            path="admin" 
            element={
              <Suspense fallback={<PageLoader />}>
                <AdminPage />
              </Suspense>
            } 
          />
          <Route 
            path="admin/kg" 
            element={
              <Suspense fallback={<PageLoader />}>
                <KGAdminPage />
              </Suspense>
            } 
          />
          <Route
            path="admin/kg/audit"
            element={
              <Suspense fallback={<PageLoader />}>
                <KGAuditPage />
              </Suspense>
            }
          />
          <Route
            path="kg/global"
            element={
              <Suspense fallback={<PageLoader />}>
                <GlobalGraphPage />
              </Suspense>
            }
          />
          <Route
            path="workflows"
            element={
              <Suspense fallback={<PageLoader />}>
                <WorkflowsPage />
              </Suspense>
            }
          />
          <Route
            path="pipelines"
            element={
              <Suspense fallback={<PageLoader />}>
                <PipelineStudioPage />
              </Suspense>
            }
          />
          <Route
            path="campaigns"
            element={
              <Suspense fallback={<PageLoader />}>
                <CampaignsPage />
              </Suspense>
            }
          />
          <Route
            path="tools"
            element={
              <Suspense fallback={<PageLoader />}>
                <ToolsPage />
              </Suspense>
            }
          />
          <Route
            path="presentations"
            element={
              <Suspense fallback={<PageLoader />}>
                <PresentationsPage />
              </Suspense>
            }
          />
          <Route
            path="artifact-drafts"
            element={
              <Suspense fallback={<PageLoader />}>
                <ArtifactDraftsPage />
              </Suspense>
            }
          />
          <Route
            path="repo-reports"
            element={
              <Suspense fallback={<PageLoader />}>
                <RepoReportsPage />
              </Suspense>
            }
          />
          <Route
            path="autonomous-agents"
            element={
              <Suspense fallback={<PageLoader />}>
                <AutonomousAgentsPage />
              </Suspense>
            }
          />
          <Route
            path="patch-prs"
            element={
              <Suspense fallback={<PageLoader />}>
                <PatchPRsPage />
              </Suspense>
            }
          />
          <Route
            path="synthesis"
            element={
              <Suspense fallback={<PageLoader />}>
                <SynthesisPage />
              </Suspense>
            }
          />
          <Route
            path="ai-hub"
            element={
              <Suspense fallback={<PageLoader />}>
                <AIHubPage />
              </Suspense>
            }
          />
          <Route
            path="research-notes"
            element={
              <Suspense fallback={<PageLoader />}>
                <ResearchNotesPage />
              </Suspense>
            }
          />
        </Route>

        {/* Workflow editor (full page, no layout) */}
        <Route
          path="/workflows/:id/edit"
          element={
            <ProtectedRoute>
              <Suspense fallback={<PageLoader />}>
                <WorkflowEditorPage />
              </Suspense>
            </ProtectedRoute>
          }
        />

        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
};

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <NotificationProvider>
            <Router>
              <div className="App">
              <AppRoutes />
              <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                // Literal hex, and it has to be: react-hot-toast renders into
                // its own portal with INLINE styles, so neither Tailwind
                // classes nor the `.bg-white` -> terminal-background override
                // in index.css reach it. The defaults here were a stock
                // #363636 box with #4ade80 icons -- a neutral grey card with
                // stock-green accents floating over a #0b0f10 surface with
                // #2dfc9a accents, which read as a different application.
                //
                // Values mirror tailwind.config.js: gray.100 lifts the toast
                // just off the gray.50 page, gray.300 gives it an edge, and
                // gray.900 is the palette's foreground. Keep them in step if
                // the theme moves.
                style: {
                  background: '#0f1516',
                  color: '#f6fff9',
                  border: '1px solid #1f2b2c',
                  // High-priority notifications are composed as
                  // `${title}\n${description}`, and HTML collapses that
                  // newline to a space -- the two ran together as one line.
                  // `pre-line` honours the break while still wrapping long
                  // descriptions, which `pre` would not.
                  whiteSpace: 'pre-line',
                },
                success: {
                  iconTheme: {
                    primary: '#2dfc9a',
                    secondary: '#0b0f10',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#0b0f10',
                  },
                },
              }}
            />
            </div>
          </Router>
        </NotificationProvider>
      </AuthProvider>
    </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;
