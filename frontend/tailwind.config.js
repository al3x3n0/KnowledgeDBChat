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
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'Monaco', 'Consolas', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-dot': 'pulseDot 1.5s infinite',
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
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}






