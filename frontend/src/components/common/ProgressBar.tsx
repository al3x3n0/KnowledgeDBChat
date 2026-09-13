/**
 * The app's progress bar.
 *
 * The track is a well and the fill sits in it, lit from its own colour — on a
 * dark ground a flat bar reads as a painted rectangle, and the faint glow is
 * what makes it read as filled instead. The fill also carries a highlight
 * along its top edge, which is the same trick the surfaces use: light falls
 * from above, so the top of a raised thing catches it.
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

  // Each fill lights its own track slightly. Kept faint on purpose: a
  // progress bar reports, it does not announce.
  const glowClasses = {
    primary: 'shadow-[0_0_8px_-1px_rgb(24_161_97_/_0.6)]',
    success: 'shadow-[0_0_8px_-1px_rgb(22_163_74_/_0.6)]',
    warning: 'shadow-[0_0_8px_-1px_rgb(202_138_4_/_0.6)]',
    danger: 'shadow-[0_0_8px_-1px_rgb(220_38_38_/_0.6)]',
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
          'w-full rounded-full overflow-hidden bg-gray-200 border border-gray-300/60',
          'shadow-[inset_0_1px_2px_0_rgb(0_0_0_/_0.4)]',
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
            'h-full rounded-full transition-all duration-slow ease-enter relative',
            // The top-edge highlight, and a glow in the fill's own colour.
            'after:absolute after:inset-x-0 after:top-0 after:h-px',
            'after:bg-white/25 after:rounded-full',
            variantClasses[variant],
            glowClasses[variant],
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

