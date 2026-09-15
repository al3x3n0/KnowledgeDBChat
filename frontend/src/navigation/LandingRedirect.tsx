/**
 * Where "/" lands, for this account.
 *
 * It was `<Navigate to="/chat" replace />`, the same for everybody. A person
 * who lives in the Control Plane started every morning in Chat and navigated
 * away, which is a small tax paid every day.
 *
 * The preference has to be fetched before we can honour it, and redirecting
 * to the default first would be visible as a flash of the wrong page plus a
 * history entry nobody asked for. So this waits -- briefly, on a request the
 * sidebar is making anyway and shares through the query cache -- and falls
 * back to the default if it cannot be answered.
 */

import React from 'react';
import { Navigate } from 'react-router-dom';

import LoadingSpinner from '../components/common/LoadingSpinner';
import { useNavigation } from './useNavigation';

export const LandingRedirect: React.FC<{ fallback?: string }> = ({
  fallback = '/chat',
}) => {
  const { landing, isLoading } = useNavigation();

  if (isLoading) {
    return (
      <div className="min-h-[50vh] flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return <Navigate to={landing || fallback} replace />;
};

export default LandingRedirect;
