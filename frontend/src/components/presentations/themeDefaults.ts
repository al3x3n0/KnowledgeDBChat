/**
 * Default theme configurations for presentation styles.
 * These match the backend PPTXBuilder styles.
 */

import { ThemeConfig, PresentationStyle } from '../../types';

export const STYLE_THEMES: Record<PresentationStyle, ThemeConfig> = {
  professional: {
    colors: {
      title_color: '#1a365d',
      accent_color: '#2e86ab',
      text_color: '#333333',
      bg_color: '#ffffff',
    },
    fonts: {
      title_font: 'Calibri, sans-serif',
      body_font: 'Calibri, sans-serif',
    },
    sizes: {
      title_size: 44,
      subtitle_size: 24,
      heading_size: 36,
      body_size: 20,
      bullet_size: 18,
    },
  },
  casual: {
    colors: {
      title_color: '#4a90d9',
      accent_color: '#ff6b6b',
      text_color: '#2d3a4a',
      bg_color: '#f8f9fa',
    },
    fonts: {
      title_font: 'Arial Rounded MT Bold, Arial, sans-serif',
      body_font: 'Arial, sans-serif',
    },
    sizes: {
      title_size: 48,
      subtitle_size: 26,
      heading_size: 38,
      body_size: 22,
      bullet_size: 20,
    },
  },
  technical: {
    colors: {
      title_color: '#007acc',
      accent_color: '#28a745',
      text_color: '#24292e',
      bg_color: '#ffffff',
    },
    fonts: {
      title_font: 'Consolas, monospace',
      body_font: 'Segoe UI, sans-serif',
    },
    sizes: {
      title_size: 40,
      subtitle_size: 22,
      heading_size: 32,
      body_size: 18,
      bullet_size: 16,
    },
  },
  modern: {
    colors: {
      title_color: '#2c3e50',
      accent_color: '#e74c3c',
      text_color: '#34495e',
      bg_color: '#ecf0f1',
    },
    fonts: {
      title_font: 'Segoe UI, sans-serif',
      body_font: 'Segoe UI, sans-serif',
    },
    sizes: {
      title_size: 46,
      subtitle_size: 24,
      heading_size: 36,
      body_size: 20,
      bullet_size: 18,
    },
  },
  minimal: {
    colors: {
      title_color: '#000000',
      accent_color: '#95a5a6',
      text_color: '#2c3e50',
      bg_color: '#ffffff',
    },
    fonts: {
      title_font: 'Helvetica, Arial, sans-serif',
      body_font: 'Helvetica, Arial, sans-serif',
    },
    sizes: {
      title_size: 42,
      subtitle_size: 22,
      heading_size: 34,
      body_size: 18,
      bullet_size: 16,
    },
  },
  corporate: {
    colors: {
      title_color: '#003d7a',
      accent_color: '#f5a623',
      text_color: '#333333',
      bg_color: '#ffffff',
    },
    fonts: {
      title_font: 'Arial, sans-serif',
      body_font: 'Arial, sans-serif',
    },
    sizes: {
      title_size: 44,
      subtitle_size: 24,
      heading_size: 36,
      body_size: 20,
      bullet_size: 18,
    },
  },
  creative: {
    colors: {
      title_color: '#9b59b6',
      accent_color: '#1abc9c',
      text_color: '#2c3e50',
      bg_color: '#fdfbf7',
    },
    fonts: {
      title_font: 'Georgia, serif',
      body_font: 'Verdana, sans-serif',
    },
    sizes: {
      title_size: 48,
      subtitle_size: 26,
      heading_size: 38,
      body_size: 20,
      bullet_size: 18,
    },
  },
  dark: {
    colors: {
      title_color: '#ffffff',
      accent_color: '#3db9d3',
      text_color: '#e0e0e0',
      bg_color: '#1e1e2e',
    },
    fonts: {
      title_font: 'Segoe UI, sans-serif',
      body_font: 'Segoe UI, sans-serif',
    },
    sizes: {
      title_size: 44,
      subtitle_size: 24,
      heading_size: 36,
      body_size: 20,
      bullet_size: 18,
    },
  },
};

/**
 * Get theme configuration for a presentation.
 * Falls back to style-based theme if no custom theme is provided.
 */
export function getThemeForPresentation(
  style: PresentationStyle,
  customTheme?: ThemeConfig | null
): ThemeConfig {
  if (customTheme) {
    return customTheme;
  }
  return STYLE_THEMES[style] || STYLE_THEMES.professional;
}
