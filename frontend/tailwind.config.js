/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Terminal theme palette (dark UI, neon accents).
        // Note: We intentionally invert the usual "50 is light" convention so existing
        // usage like `bg-gray-50` + `text-gray-900` becomes dark-bg + light-text.
        primary: {
          50: '#07150f',
          100: '#0a2016',
          200: '#0e2f20',
          300: '#12422b',
          400: '#15623d',
          500: '#18a161',
          600: '#19c77b',
          700: '#2dfc9a',
          800: '#8dffbf',
          900: '#d9ffe9',
        },
        gray: {
          50: '#0b0f10',
          100: '#0f1516',
          200: '#162021',
          300: '#1f2b2c',
          400: '#415557',
          500: '#6c8482',
          600: '#9fb2ac',
          700: '#c8d6cf',
          800: '#e7f2ec',
          900: '#f6fff9',
        }
      },
      fontFamily: {
        // One family, two voices. `sans` carries prose and chrome; `mono` is
        // reserved for things that are literally values — measurements, ids,
        // timestamps, paths — so monospace means something instead of being
        // the default everything inherited.
        sans: ['IBM Plex Sans', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['IBM Plex Mono', 'Monaco', 'Consolas', 'monospace'],
      },
      // Elevation, for a dark UI.
      //
      // A black shadow on a near-black ground is invisible, so height here is
      // carried by the SURFACE first: each level is a lighter plane than the
      // one under it, with a hairline to catch the edge and a soft ambient
      // shadow underneath for separation. Use the `surface-*` utilities in
      // index.css rather than reaching for these three pieces by hand.
      boxShadow: {
        // Resting on the page: no lift, just a defined edge.
        'level-0': 'none',
        // A card or panel.
        'level-1': '0 1px 2px 0 rgb(0 0 0 / 0.40), 0 1px 3px 0 rgb(0 0 0 / 0.30)',
        // A dropdown, popover, or a card being hovered.
        'level-2': '0 4px 8px -2px rgb(0 0 0 / 0.50), 0 2px 4px -2px rgb(0 0 0 / 0.40)',
        // A modal or a drawer, above everything.
        'level-3': '0 16px 32px -8px rgb(0 0 0 / 0.65), 0 8px 16px -8px rgb(0 0 0 / 0.50)',
        // The accent's own glow, for a live or focused element. Deliberately
        // faint: it should register at the edge of vision, not announce itself.
        'accent-glow': '0 0 0 1px rgb(24 161 97 / 0.35), 0 2px 12px -2px rgb(24 161 97 / 0.25)',
      },
      transitionDuration: {
        // Three speeds, so timing is a decision rather than a default.
        // 120: something under the cursor answering. 180: a state change.
        // 240: something arriving or leaving.
        'fast': '120ms',
        'base': '180ms',
        'slow': '240ms',
      },
      transitionTimingFunction: {
        // Standard easing for anything that moves under a pointer, and a
        // gentler out-curve for things that enter.
        'ui': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'enter': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-dot': 'pulseDot 1.5s infinite',
        // Entrances, on the gentler curve.
        'rise-in': 'riseIn 240ms cubic-bezier(0.16, 1, 0.3, 1)',
        'scale-in': 'scaleIn 180ms cubic-bezier(0.16, 1, 0.3, 1)',
        // A live process, without the jitter of a spinner.
        'breathe': 'breathe 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 1.6s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pulseDot: {
          '0%, 20%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.2)', opacity: '0.7' },
          '80%, 100%': { transform: 'scale(1)', opacity: '1' },
        },
        riseIn: {
          '0%': { transform: 'translateY(6px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.97)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        breathe: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.55' },
        },
        // Sweeps a highlight across a skeleton placeholder.
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}






