/**
 * The app's text input.
 *
 * A field should say three things without being read: where it is, whether it
 * has your attention, and whether it is wrong. So the resting state sits one
 * plane below its container (an input is a well, not a card), focus lifts it
 * and warms the border to the accent, and an error recolours the border and
 * the message together rather than only printing red text underneath.
 *
 * The id is generated once and kept. It used to be recomputed on every render,
 * which broke the label's `htmlFor` link the moment anything above re-rendered
 * — clicking the label stopped focusing the field, and a screen reader lost
 * the association.
 */

import clsx from 'clsx';
import React, { useId } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helpText?: string;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(({
  label,
  error,
  helpText,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className,
  id,
  ...props
}, ref) => {
  const generatedId = useId();
  const inputId = id || generatedId;
  const describedBy = error ? `${inputId}-error` : helpText ? `${inputId}-help` : undefined;

  return (
    <div className={clsx('space-y-1', fullWidth && 'w-full')}>
      {label && (
        <label 
          htmlFor={inputId} 
          className="block text-sm font-medium text-gray-700"
        >
          {label}
        </label>
      )}
      
      <div className="relative">
        {leftIcon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <span className="text-gray-400 text-sm">{leftIcon}</span>
          </div>
        )}
        
        <input
          ref={ref}
          id={inputId}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          className={clsx(
            // A well: darker than the surface it sits on, with the shadow
            // inset rather than cast.
            'block w-full rounded-md bg-gray-50 text-gray-900 border border-gray-300',
            'shadow-[inset_0_1px_2px_0_rgb(0_0_0_/_0.35)]',
            'transition-all duration-fast ease-ui',
            'hover:border-gray-400',
            // Focus lifts it out of the well and warms the edge.
            'focus:border-primary-600 focus:bg-gray-100 focus:shadow-accent-glow',
            'focus:outline-none focus:ring-0',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            error &&
              'border-red-500/70 hover:border-red-500 focus:border-red-500 ' +
              'focus:shadow-[0_0_0_1px_rgb(239_68_68_/_0.35),0_2px_12px_-2px_rgb(239_68_68_/_0.25)]',
            leftIcon && 'pl-10',
            rightIcon && 'pr-10',
            className
          )}
          {...props}
        />
        
        {rightIcon && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
            <span className="text-gray-400 text-sm">{rightIcon}</span>
          </div>
        )}
      </div>
      
      {error && (
        <p id={`${inputId}-error`} className="text-sm text-red-400 animate-rise-in">
          {error}
        </p>
      )}

      {helpText && !error && (
        <p id={`${inputId}-help`} className="text-sm text-gray-500">
          {helpText}
        </p>
      )}
    </div>
  );
});

Input.displayName = 'Input';

export default Input;

