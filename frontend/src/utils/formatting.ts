/**
 * Utility functions for formatting data
 */

export const formatFileSize = (bytes?: number): string => {
  if (!bytes) return 'N/A';
  
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  
  return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};

export const formatDuration = (seconds: number): string => {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
};

export const formatPercentage = (value: number, total: number): string => {
  if (total === 0) return '0%';
  return `${Math.round((value / total) * 100)}%`;
};

export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};

export const capitalizeFirst = (text: string): string => {
  return text.charAt(0).toUpperCase() + text.slice(1);
};

export const formatSourceType = (sourceType: string): string => {
  switch (sourceType) {
    case 'gitlab':
      return 'GitLab';
    case 'confluence':
      return 'Confluence';
    case 'web':
      return 'Web Scraping';
    case 'file':
      return 'File Upload';
    default:
      return capitalizeFirst(sourceType);
  }
};

export const getStatusColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'active':
    case 'completed':
    case 'processed':
      return 'text-green-600';
    case 'degraded':
    case 'warning':
    case 'processing':
    case 'pending':
      return 'text-yellow-600';
    case 'unhealthy':
    case 'error':
    case 'failed':
    case 'inactive':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
};

export const getStatusBadgeColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'active':
    case 'completed':
    case 'processed':
      return 'bg-green-100 text-green-800';
    case 'degraded':
    case 'warning':
    case 'processing':
    case 'pending':
      return 'bg-yellow-100 text-yellow-800';
    case 'unhealthy':
    case 'error':
    case 'failed':
    case 'inactive':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};






