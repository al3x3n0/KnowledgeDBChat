/**
 * Custom hook for managing keyboard shortcuts
 */

import { useEffect, useCallback } from 'react';

export interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  handler: (e: KeyboardEvent) => void;
  description?: string;
  preventDefault?: boolean;
}

interface UseKeyboardShortcutsOptions {
  enabled?: boolean;
  preventDefault?: boolean;
}

/**
 * Hook for managing keyboard shortcuts
 * 
 * @param shortcuts Array of keyboard shortcut configurations
 * @param options Configuration options
 * 
 * @example
 * ```tsx
 * useKeyboardShortcuts([
 *   {
 *     key: 'k',
 *     ctrlKey: true,
 *     handler: () => focusInput(),
 *     description: 'Focus input'
 *   }
 * ]);
 * ```
 */
export const useKeyboardShortcuts = (
  shortcuts: KeyboardShortcut[],
  options: UseKeyboardShortcutsOptions = {}
) => {
  const { enabled = true, preventDefault: defaultPreventDefault = true } = options;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!enabled) return;

      // Check if we're in an input, textarea, or contenteditable element
      const target = e.target as HTMLElement;
      const isInputElement =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable;

      // Find matching shortcut
      for (const shortcut of shortcuts) {
        const keyMatches = shortcut.key.toLowerCase() === e.key.toLowerCase();
        const ctrlMatches = shortcut.ctrlKey === undefined ? true : shortcut.ctrlKey === (e.ctrlKey || e.metaKey);
        const metaMatches = shortcut.metaKey === undefined ? true : shortcut.metaKey === e.metaKey;
        const shiftMatches = shortcut.shiftKey === undefined ? true : shortcut.shiftKey === e.shiftKey;
        const altMatches = shortcut.altKey === undefined ? true : shortcut.altKey === e.altKey;

        // Handle Ctrl/Cmd key (Ctrl on Windows/Linux, Cmd on Mac)
        const modifierMatches = shortcut.ctrlKey !== undefined
          ? (shortcut.ctrlKey && (e.ctrlKey || e.metaKey)) || (!shortcut.ctrlKey && !e.ctrlKey && !e.metaKey)
          : true;

        if (
          keyMatches &&
          modifierMatches &&
          metaMatches &&
          shiftMatches &&
          altMatches
        ) {
          // Check if shortcut should work in input fields
          // If shortcut has ctrlKey or metaKey, it should work everywhere
          // Otherwise, only work outside input fields
          if (isInputElement && !shortcut.ctrlKey && !shortcut.metaKey) {
            continue;
          }

          if (shortcut.preventDefault !== undefined ? shortcut.preventDefault : defaultPreventDefault) {
            e.preventDefault();
          }

          shortcut.handler(e);
          break; // Only handle first matching shortcut
        }
      }
    },
    [shortcuts, enabled, defaultPreventDefault]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown, enabled]);
};

/**
 * Helper function to get platform-specific modifier key name
 */
export const getModifierKey = (): 'Ctrl' | 'Cmd' => {
  return navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl';
};

/**
 * Helper function to format shortcut for display
 */
export const formatShortcut = (shortcut: Omit<KeyboardShortcut, 'handler'>): string => {
  const parts: string[] = [];
  
  if (shortcut.ctrlKey || shortcut.metaKey) {
    parts.push(getModifierKey());
  }
  
  if (shortcut.shiftKey) {
    parts.push('Shift');
  }
  
  if (shortcut.altKey) {
    parts.push('Alt');
  }
  
  parts.push(shortcut.key.toUpperCase());
  
  return parts.join(' + ');
};

