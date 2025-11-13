/**
 * Reusable progress bar component
 */

import React from 'react';
import clsx from 'clsx';

interface ProgressBarProps {
  value: number; // 0-100
  max?: number;
  showLabel?: boolean;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'primary' | 'success' | 'warning' | 'danger';
  indeterminate?: boolean;
  className?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  showLabel = false,
  label,
  size = 'md',
  variant = 'primary',
  indeterminate = false,
  className,
}) => {
  const percentage = indeterminate ? undefined : Math.min(Math.max((value / max) * 100, 0), 100);
  const displayLabel = label || (percentage !== undefined ? `${Math.round(percentage)}%` : '');

  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const variantClasses = {
    primary: 'bg-primary-600',
    success: 'bg-green-600',
    warning: 'bg-yellow-600',
    danger: 'bg-red-600',
  };

  return (
    <div className={clsx('w-full', className)}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium text-gray-700">
            {displayLabel}
          </span>
        </div>
      )}
      <div
        className={clsx(
          'w-full bg-gray-200 rounded-full overflow-hidden',
          sizeClasses[size]
        )}
        role="progressbar"
        aria-valuenow={indeterminate ? undefined : value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label || 'Progress'}
      >
        <div
          className={clsx(
            'h-full rounded-full transition-all duration-300 ease-out',
            variantClasses[variant],
            indeterminate && 'animate-pulse'
          )}
          style={
            indeterminate
              ? {
                  width: '30%',
                  animation: 'progress-indeterminate 1.5s ease-in-out infinite',
                }
              : percentage !== undefined
              ? {
                  width: `${percentage}%`,
                }
              : {}
          }
        />
      </div>
    </div>
  );
};

export default ProgressBar;

