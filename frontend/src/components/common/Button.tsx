/**
 * The app's button.
 *
 * Four variants, and they are a hierarchy rather than a palette: `primary` is
 * the one action a screen wants you to take and is the only one that carries
 * the accent as a fill; `secondary` is everything else you might do; `ghost`
 * is for actions that live inside other content and should not compete with
 * it; `danger` is for the ones you cannot undo.
 *
 * Every variant answers the pointer in the same three ways — it lightens on
 * hover, it presses down half a pixel on click, and it takes the app's focus
 * ring on keyboard focus. That consistency is most of what makes a set of
 * buttons feel built rather than assembled.
 */

import clsx from 'clsx';
import { Loader2 } from 'lucide-react';
import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  fullWidth?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = 'primary',
      size = 'md',
      loading = false,
      icon,
      fullWidth = false,
      className,
      disabled,
      ...props
    },
    ref
  ) => {
    // `active:translate-y-px` is the whole trick behind a button feeling
    // physical: half a pixel of travel, gone in 120ms.
    const baseClasses =
      'inline-flex items-center justify-center font-medium rounded-md ' +
      'transition-all duration-fast ease-ui select-none ' +
      'active:translate-y-px ' +
      'disabled:opacity-40 disabled:cursor-not-allowed ' +
      'disabled:hover:translate-y-0 disabled:active:translate-y-0';

    const variantClasses = {
      // Accent-tinted rather than accent-filled. 536 of this app's 758 buttons
      // take the default variant, so a solid fill here would put a bright
      // green block on every toolbar in the product; tinting keeps the accent
      // meaningful and keeps the terminal look intact. The glow on hover is
      // what makes it feel lit rather than merely outlined.
      primary:
        'bg-primary-500/10 text-primary-700 border border-primary-500/60 ' +
        'hover:bg-primary-500/20 hover:border-primary-500 hover:shadow-accent-glow ' +
        'active:bg-primary-500/10',
      secondary:
        'bg-gray-100 text-gray-900 border border-gray-300 shadow-level-1 ' +
        'hover:bg-gray-200 hover:border-gray-400 hover:shadow-level-2 ' +
        'active:bg-gray-100 active:shadow-level-1',
      // No border and no shadow at rest: it should be invisible until wanted.
      ghost:
        'bg-transparent text-gray-700 border border-transparent ' +
        'hover:bg-gray-200 hover:text-gray-900',
      danger:
        'bg-transparent text-red-300 border border-red-500/70 ' +
        'hover:bg-red-500/10 hover:border-red-500 hover:text-red-200',
    };

    const sizeClasses = {
      sm: 'px-3 py-1.5 text-sm',
      md: 'px-4 py-2 text-sm',
      lg: 'px-6 py-3 text-base',
    };

    return (
      <button
        ref={ref}
        className={clsx(
          baseClasses,
          variantClasses[variant],
          sizeClasses[size],
          fullWidth && 'w-full',
          className
        )}
        disabled={disabled || loading}
        aria-busy={loading || undefined}
        {...props}
      >
        {loading ? (
          <Loader2 className="w-4 h-4 mr-2 animate-spin" aria-hidden="true" />
        ) : icon ? (
          <span className="mr-2 inline-flex shrink-0" aria-hidden="true">
            {icon}
          </span>
        ) : null}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;
